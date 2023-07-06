# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
from tqdm import tqdm 
from datasets.oxford import TrainingTuple
# Import test set boundaries
import matplotlib.pyplot as plt 


FILENAME = "pd_northing_easting.csv"
POINTCLOUD_FOLS = "/Ouster/"


def construct_query_dict(df_centroids, base_path, filename, ind_nn_r, ind_r_r=50):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    # Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
    # Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)
    queries = {}
    for anchor_ndx in tqdm(range(len(ind_nn))):
        # print(df_centroids)
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        query = df_centroids.iloc[anchor_ndx]["file"]
        # Extract timestamp from the filename
        # print(query)
        scan_filename = os.path.split(query)[1]
        assert os.path.splitext(scan_filename)[1] == '.npy', f"Expected .npy file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])

        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]

        positives = positives[positives != anchor_ndx]

        
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)
        # print('\n############\n', anchor_ndx, '\n\n', positives, '\n\n', non_negatives)

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position=anchor_pos)

    file_path = os.path.join(base_path, filename)
    print(file_path)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

    return queries 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Baseline training dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')
    parser.add_argument('--save_path', type = str, required = True)
    parser.add_argument('--skip_nonjoint', action = 'store_true', default = False)
    parser.add_argument('--ind_nn_r', type = int, default = 3)
    parser.add_argument('--ind_r_r', type = int, default = 20)
    args = parser.parse_args()
    print('Dataset root: {}'.format(args.dataset_root))

    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root

    
    RUNS_FOLDERS = ["DCC", "Riverside"] # Change these two lines if you want to change the train / test splits
    EXCLUDED_FOLDERS = ["DCC_03", "Riverside_02"] # Change these two lines if you want to change the train / test splits 
    all_trees = []
    df_all_train = pd.DataFrame(columns=['file', 'northing', 'easting'])

    for RUNS_FOLDER in RUNS_FOLDERS:
        all_folders = sorted(os.listdir(os.path.join(base_path, 'MulRan', RUNS_FOLDER)))
        df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
        for folder in all_folders:
            if folder in EXCLUDED_FOLDERS:
                print("Excluding : {}".format(folder))
                continue 
            else:
                df_locations = pd.read_csv(os.path.join(base_path, 'MulRan', RUNS_FOLDER, folder, FILENAME), sep=',')
                df_locations['timestamp'] = 'MulRan/' + RUNS_FOLDER + '/' + folder + POINTCLOUD_FOLS + df_locations['timestamp'].astype(str) + '.npy'
                df_locations = df_locations.rename(columns = {'timestamp': 'file'})
                df_train = df_train.append(df_locations)
                df_all_train = df_all_train.append(df_locations)
                print(df_train)
        
        print(f"Train samples for {RUNS_FOLDER} : {len(df_train)}")
        if not args.skip_nonjoint:
            construct_query_dict(df_train, args.save_path, f"train_queries_MULRAN_{RUNS_FOLDER}.pickle", ind_nn_r = args.ind_nn_r, ind_r_r = args.ind_r_r)
    
    print("Number of training submaps: " + str(len(df_all_train['file'])))
    construct_query_dict(df_all_train, args.save_path, f"train_queries_MulRan.pickle", ind_nn_r=args.ind_nn_r, ind_r_r = args.ind_r_r)
    #         print(folder)
    #         df_locations = pd.read_csv(os.path.join(base_path, 'MulRan', RUNS_FOLDER, folder, FILENAME), sep=',')
    #         df_locations['timestamp'] = 'MulRan/' + RUNS_FOLDER + '/' + folder + POINTCLOUD_FOLS + df_locations['timestamp'].astype(str) + '.npy'
    #         df_train = df_locations.rename(columns={'timestamp': 'file'})
    #         print(df_train['file'])
    #         if folder not in EXCLUDED_FOLDERS:
    #             all_trees.append(df_train)
    #             df_all_train = df_all_train.append(df_train)
    #         else:
    #             print("EXCLUDING: {}".format(folder))
    #         construct_query_dict(df_train, os.path.join(base_path, 'pickles', 'MulRan', args.save_folder), f"train_queries_MULRAN_{folder}.pickle", ind_nn_r=args.ind_nn_r, ind_r_r = args.ind_r_r)

    #         # if not args.skip_nonjoint:
    #         #     print("Number of training submaps: " + str(len(df_train['file'])))
    #         #     construct_query_dict(df_train, os.path.join(base_path, 'pickles', 'MulRan', args.save_folder), f"train_queries_MULRAN_{folder}.pickle", ind_nn_r=args.ind_nn_r, ind_r_r = args.ind_r_r)
    
    # # Do all
    # print("Number of training submaps: " + str(len(df_all_train['file'])))
    # construct_query_dict(df_all_train, os.path.join(base_path, 'pickles', 'MulRan', args.save_folder), f"train_queries_MULRAN_JOINT_v2.pickle", ind_nn_r=args.ind_nn_r, ind_r_r = args.ind_r_r)