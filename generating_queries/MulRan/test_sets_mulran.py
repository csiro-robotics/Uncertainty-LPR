# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
from tqdm import tqdm 
import matplotlib.pyplot as plt 
from glob import glob

RUNS_FOLDER = "MulRan/"
FILENAME = "pd_northing_easting.csv"
POINTCLOUD_FOLS = "Ouster"
def construct_query_and_database_sets(data_root, folder, eval_thresh, time_thresh, save_path, filetype):
    
    # Load data, make distance and time trees
    print(data_root, folder, FILENAME)
    df_locations = pd.read_csv(os.path.join(data_root, folder[1:], FILENAME))
    df_locations['file'] = folder + '/' + POINTCLOUD_FOLS + '/' + df_locations['timestamp'].astype(str) + filetype
    
    print(df_locations[['timestamp','file']])
    df_locations['timestamp'] = df_locations['timestamp'] / 1e9
    distance_tree = KDTree(df_locations[['northing', 'easting']])
    time_tree = KDTree(df_locations['timestamp'].to_numpy().reshape(-1,1))

    # Make database
    database = {}
    for index, row in df_locations.iterrows():
        database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                              'easting': row['easting'], 'timestamp': row['timestamp']}
    
    # Get positive matches
    start_time = database[0]['timestamp']
    for key in tqdm(range(len(database.keys()))):
        # Get time index: Everything up to 90 seconds 
        if database[key]['timestamp'] - start_time < time_thresh:
            database[key]['tt'] = -1
            database[key]['revisit'] = False 
        else:
            database[key]['tt'] = next(x[0] for x in enumerate(df_locations['timestamp'].tolist()) 
                                    if x[1] > database[key]['timestamp'] - time_thresh) + 1
            coor = np.array([[database[key]['northing'], database[key]['easting']]])
            index_coor = distance_tree.query_radius(coor, r = eval_thresh)[0].tolist()
            index_coor = [x for x in index_coor if x < database[key]['tt']]
            if len(index_coor) != 0:
                database[key]['revisit'] = True 
            else:
                database[key]['revisit'] = False


    file_path = os.path.join(save_path, f"{folder.split('/')[-1]}_evaluation_database.pickle")
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'wb') as handle:
        pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", file_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation datasets')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')
    parser.add_argument('--eval_thresh', type = int, default = 10)
    parser.add_argument('--time_thresh', type = int, default = 90)
    parser.add_argument('--save_path', type = str, required = True)
    parser.add_argument('--filetype', type = str, default = '.npy')

    args = parser.parse_args()
    print('Dataset root: {}'.format(args.dataset_root))

    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root

    
    all_folders = glob(os.path.join(base_path, RUNS_FOLDER, '*', '*'))

    all_folders = [x.replace(base_path, '') for x in all_folders if 'Sejong' not in x] 

    for folder in all_folders:
        construct_query_and_database_sets(
            data_root = base_path,
            folder = folder,
            eval_thresh = args.eval_thresh,
            time_thresh = args.time_thresh,
            save_path = args.save_path,
            filetype = args.filetype
        )
