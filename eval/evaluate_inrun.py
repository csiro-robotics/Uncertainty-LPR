# Author: Jacek Komorowski
# Warsaw University of Technology

# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

from sklearn.neighbors import KDTree
import numpy as np
import os
import argparse
import torch
from tqdm import tqdm 
import MinkowskiEngine as ME
import random

from models.model_factory import model_factory
from torchpack.utils.config import configs 
from torch.utils.data import DataLoader

import scipy.spatial as spatial 
import pandas as pd 
import matplotlib.pyplot as plt 


def evaluate_dataset(model, device, database, silent = True):
    if 'r_thresh' in database.keys():
        _ = database.pop('r_thresh')
    # Get thresh 
    thresholds = np.linspace(0.001, 1.0, 1000)
    num_thresholds = len(thresholds)

    # Get database embeddings and locations 
    database_embeddings = get_latent_vectors(model, database, device)
    database_locations = np.array([[database[idx]['northing'], database[idx]['easting']] for idx in range(len(database_embeddings))])


    # Set up lists, variables
    num_thresholds = configs.eval.num_thresholds 
    thresholds = np.linspace(configs.eval.cd_thresh_min, configs.eval.cd_thresh_max, int(num_thresholds))
    num_true_positives = np.zeros(num_thresholds)
    num_false_positives = np.zeros(num_thresholds)
    num_true_negatives = np.zeros(num_thresholds)
    num_false_negatives = np.zeros(num_thresholds)



    
    num_revisits = 0
    num_correct_loc = 0
    
    # Iterate over embeddings
    for idx in tqdm(range(len(database_embeddings))):
        if database[idx]['tt'] == -1:
            continue 

    
        # Get neighbours, idx
        embedding_idx = database_embeddings[idx]
        location_idx = database_locations[idx] #torch.tensor([database_locations[idx]['northing'], database_locations[idx]['easting']])
    
        
        past_embeddings = database_embeddings[:database[idx]['tt']]
        past_locations = database_locations[:database[idx]['tt']]
        dist = spatial.distance.cdist(embedding_idx.reshape(1,-1), past_embeddings).reshape(-1)
        dist_sorted, idx_sorted = np.sort(dist), np.argsort(dist)
        min_dist = dist_sorted[0]
        
        locations_sorted = past_locations[idx_sorted] #torch.index_select(past_locations, dim = 0, index = idx_sorted)
        world_dist_sorted = spatial.distance.cdist(location_idx.reshape(1,-1), locations_sorted).reshape(-1)
        world_dist_sorted_in_range = world_dist_sorted <= configs.eval.cd_revisit_criteria

        is_revisit = database[idx]['revisit'] #np.any(world_dist_sorted_in_range) # Has it been here before?
        
        is_correct_loc = 0
        if is_revisit:
            num_revisits += 1
            if world_dist_sorted_in_range[0] == True:
                num_correct_loc += 1
                is_correct_loc = 1

        # Eval top-1 candidate for F1 Score
        for thresh_idx in range(num_thresholds):
            threshold = thresholds[thresh_idx]

            if min_dist < threshold: # Positive prediction
                if is_revisit == True:
                    num_true_positives[thresh_idx] += 1
                elif world_dist_sorted[0] > configs.eval.cd_not_revisit_criteria:
                    num_false_positives[thresh_idx] += 1
            else:  # Negative Prediction
                if is_revisit == True:
                    num_false_negatives[thresh_idx] += 1
                else:
                    num_false_positives[thresh_idx] += 1

    recall_1 = num_correct_loc / num_revisits

    F1max = 0.0
    AP = 0.0
    AP_count = 0
    Precisions, Recalls = [], []
    
    for thresh_idx in range(num_thresholds):
        nTrueNegative = num_true_negatives[thresh_idx]
        nFalseNegative = num_false_negatives[thresh_idx]
        nTruePositive = num_true_positives[thresh_idx]
        nFalsePositive = num_false_positives[thresh_idx]

        nTotalTestPlaces = nTrueNegative + nFalsePositive + nTruePositive + nFalseNegative

        Precision = 0
        Recall = 0
        Prev_Recall = 0
        F1 = 0.0

        if nTruePositive > 0:
            Precision = nTruePositive / (nTruePositive + nFalsePositive)
            Recall = nTruePositive / (nTruePositive + nFalseNegative)

            F1 = 2 * Precision * Recall / (Precision  + Recall)
            AP += (Recall - Prev_Recall) * Precision    
            AP_count += 1
            Prev_Recall = Recall

        if F1 > F1max:
            F1max = F1

        Precisions.append(Precision)
        Recalls.append(Recall)

    mAP = AP / AP_count

    if configs.eval.save_F1_graph == True:
        plt.title('F1 Graph')
        plt.plot(Recalls, Precisions, marker = '.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # plt.axis([0,1,0,1.1])
        # plt.xticks(np.arange(0, 1.01, step=0.1)) 
        plt.grid(True)
        plt.show()
    
    stats = {'Recall@1': recall_1*100, 'F1max': F1max*100}
    return stats

class EvalDataLoader():
    def __init__(self, dataset):
        self.set = dataset 
        self.dataset_path = configs.data.dataset_folder
        self.n_points = 4096

    
    def load_pc(self, filename):
        if '.bin' in filename:
            file_path = os.path.join(self.dataset_path, filename)
            pc = np.fromfile(file_path, dtype = np.float64)
            # coords are within -1..1 range in each dimension
            assert pc.shape[0] == self.n_points * 3, "Error in point cloud shape: {}".format(file_path)
            pc = np.reshape(pc, (pc.shape[0] // 3, 3))
            pc = torch.tensor(pc, dtype=torch.float)
        elif '.npy' in filename:
            file_path = os.path.join(self.dataset_path, filename)
            pc = np.load(file_path)[:,:3]
            assert pc.shape[0] == self.n_points, "Error in point cloud shape: {}".format(file_path)
            pc = torch.tensor(pc, dtype = torch.float)
        return pc 

    def __len__(self):
        return len(self.set)
    
    def __getitem__(self, idx):
        try:
            x = self.load_pc(self.set[idx]['query'])
        except:
            print(self.set.keys())
            print(len(self.set))

            print(idx)
            x = self.load_pc(self.set[idx]['query'])
        return x

def get_eval_dataloader(set):
    eval_dataset = EvalDataLoader(set)

    def collate_fn(data_list):
        clouds = [e for e in data_list]
        labels = [e[1] for e in data_list]
        batch = torch.stack(data_list, 0)
        
        coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=configs.model.mink_quantization_size)
                    for e in batch]
        bcoords = ME.utils.batched_coordinates(coords)
        feats = torch.ones((bcoords.shape[0], 1), dtype = torch.float32)
        batch = {'coords': bcoords, 'features': feats, 'cloud': batch}
        return batch 

    dataloader = DataLoader(
        eval_dataset,
        batch_size = 256,
        shuffle = False,
        collate_fn = collate_fn,
        num_workers = configs.train.num_workers
    )

    return dataloader

@torch.no_grad()
def get_latent_vectors(model, set, device):
    # Adapted from original PointNetVLAD code

    model.eval()
    embeddings_l = []
    # Create eval dataloader
    dataloader = get_eval_dataloader(set)

    for idx, batch in tqdm(enumerate(dataloader), total = len(set) // 256 + 1, desc = 'Getting Latent Vectors'):
        batch = {k:v.cuda() for k,v in batch.items()}
        embedding = model(batch)
        embedding = list(embedding)
        if configs.train.loss.normalize_embeddings:
            embedding = [torch.nn.functional.normalize(x, p=2, dim=-1) for x in embedding]
        for x in embedding:
            embeddings_l.append(x.detach().cpu().numpy())
    embeddings = np.vstack(embeddings_l)

    return embeddings
