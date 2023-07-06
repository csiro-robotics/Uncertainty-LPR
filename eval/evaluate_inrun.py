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
from eval.ue_metrics import get_uncertainty_results

import scipy.spatial as spatial 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import cosine_similarity


def get_embeddings(modelList, set, device, unc_method):
    #gets the embedding given every model in modelList

    mean_emb = []
    variance_emb = []
    loc_emb = []

    # uncertainty methods will have embeddings containing both mean and variance cosine sim
    if unc_method == 'STUN' or unc_method == 'pfe': 
        for model in modelList:
            mean, variance = get_latent_vectors_STUN(model, set, device)
            mean_emb.append(mean)
            variance_emb.append(variance)
            loc = np.array([[set[idx]['northing'], set[idx]['easting']] for idx in range(len(mean))])
            loc_emb.append(loc)
    else: # embeddings just have mean cosine sim
        for model in modelList:
            mean = get_latent_vectors(model, set, device) # B x D
            variance =  mean
            mean_emb.append(mean)
            variance_emb.append(variance)
            loc = np.array([[set[idx]['northing'], set[idx]['easting']] for idx in range(len(mean))])
            loc_emb.append(loc)

    return mean_emb, variance_emb, loc_emb


def evaluate_dataset(modelList, device, database, unc_method = '', silent = True):
    
    for model in modelList:
        model.eval()
    
    # Get thresh 
    if 'r_thresh' in database.keys():
        _ = database.pop('r_thresh')

    thresholds = np.linspace(0.001, 1.0, 1000)
    num_thresholds = len(thresholds)

    # Get database embeddings and locations
    mean, variance, loc = get_embeddings(modelList, database, device, unc_method)
    database_embeddings_all = mean 
    database_locations_all = loc 
    database_variances_all = variance 

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
    database_embeddings = database_embeddings_all[0] 
    database_locations = database_locations_all[0]

    allAccuracy = [] 
    allSimilarity = [] 
    allVariance = []
    allClosestNeighbour = []

    # For each database query
    for idx in tqdm(range(len(database_embeddings_all[0]))):
        # Do not consider most recent 90 seconds
        if database[idx]['tt'] == -1:
            continue 
    
        all_dist = []
        all_cos_sim = []
        all_world_dist_sorted = []
        all_top_world_dist = []

        all_vars = []

        # ENSEMBLES
        for mIdx in range(len(database_embeddings_all)):

            # Get neighbours, idx
            embedding_idx = database_embeddings_all[mIdx][idx]
            past_embeddings = database_embeddings_all[mIdx][:database[idx]['tt']]
            past_variances = database_variances_all[mIdx][:database[idx]['tt']]
           
            cos_sim = 1 - spatial.distance.cdist(embedding_idx.reshape(1,-1), past_embeddings, 'cosine').reshape(-1)
            dist = spatial.distance.cdist(embedding_idx.reshape(1,-1), past_embeddings).reshape(-1)

            all_dist += [dist]
            all_cos_sim += [cos_sim]
            all_vars += [past_variances]

        avg_dist = np.mean(all_dist, axis = 0)
        avg_cos_sim = np.mean(all_cos_sim, axis = 0)

        cos_asc_sorted, idx_acs_sorted = np.sort(avg_cos_sim), np.argsort(avg_cos_sim)
        cos_sorted, idx_sorted = cos_asc_sorted[::-1], idx_acs_sorted[::-1]

        #get the variance associated with top 1 prediction
        if unc_method == 'STUN':
            var_cos_sim = float(np.mean(database_variances_all[0][idx])) #query variance
        elif unc_method == 'pfe':
            #calculate similarity score between query and top database match
            aEmbed = database_embeddings_all[0][idx] #query embedding
            aVar = np.exp(database_variances_all[0][idx]) #query variance
            pEmbed = database_embeddings_all[0][idx_sorted[0]] #database embedding
            pVar = np.exp(all_vars[0][idx_sorted[0]]) #database variance

            meanDiffSq = (aEmbed-pEmbed)**2
            varSum = aVar + pVar
            logVarSum = np.log(varSum)
            similarity = -0.5*np.sum((meanDiffSq/(varSum+1e-10 ))+logVarSum)
            var_cos_sim = similarity
        else:
            var_cos_sim = np.var(all_cos_sim, axis = 0)[idx_sorted[0]]

        min_dist = avg_dist[idx_sorted[0]]
        max_cos = cos_sorted[0]
        prediction_id = int(idx_sorted[0])
        top1_var = var_cos_sim
        
        # ENSEMBLE DIFFERS HERE 
        location_idx = database_locations_all[0][idx] 
        past_locations = database_locations_all[0][:database[idx]['tt']]

        locations_sorted = past_locations[idx_sorted] 
            
        # Entire list of distances to place match
        avg_world_dist_sorted = spatial.distance.cdist(location_idx.reshape(1,-1), locations_sorted).reshape(-1)

        # Top matching relative distance
        place_candidate = past_locations[idx_sorted[0]]
        avg_top_world_dist = np.linalg.norm(location_idx - place_candidate) #the prediction

        world_dist_sorted_in_range = avg_world_dist_sorted <= configs.eval.cd_revisit_criteria

        #the ground truth
        is_revisit = database[idx]['revisit'] 

        allSimilarity.append(max_cos)
        allVariance.append(top1_var)
        if is_revisit: #GND-KNOWN

            if np.sum(world_dist_sorted_in_range) == 0:
                closest_neighbour = -1 #there was no TP
            else:
                correct_idxes = np.where(world_dist_sorted_in_range == 1)
                closest_neighbour = 1+correct_idxes[0][0] 

            num_revisits += 1
            if world_dist_sorted_in_range[0] == True: #PREDICT-KNOWN-CORRECT-TP
                num_correct_loc += 1
                allAccuracy.append(1)
            else: #PREDICT-KNOWN-INCORRECT-FP
                allAccuracy.append(0)
        else: #GND-UNKNOWN FP
            allAccuracy.append(-1)
            closest_neighbour = -1

        allClosestNeighbour.append(int(closest_neighbour))

        # Eval top-1 candidate for F1 Score
        for thresh_idx in range(num_thresholds):
            threshold = thresholds[thresh_idx]

            if min_dist < threshold: # Positive prediction
                if is_revisit == True:
                    num_true_positives[thresh_idx] += 1
                elif avg_world_dist_sorted[0] > configs.eval.cd_not_revisit_criteria:
                    num_false_positives[thresh_idx] += 1
            else:  # Negative Prediction
                if is_revisit == True:
                    num_false_negatives[thresh_idx] += 1
                else:
                    num_true_negatives[thresh_idx] += 1

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

    # mAP = AP / AP_count

    unc_results = get_uncertainty_results(allSimilarity,allAccuracy,False) 
    aupr_in = unc_results['auprIn']*100
    ausc = unc_results['auSC']*100
    auroc = unc_results['auroc']*100

    if configs.eval.save_F1_graph == True:
        plt.title('F1 Graph')
        plt.plot(Recalls, Precisions, marker = '.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.show()

    stats = {'Recall@1': recall_1*100, 'AuPR': aupr_in, 'F1max': F1max*100, 'auSC': ausc, 'AuROC': auroc,
        'Top1': {'Accuracy': allAccuracy, 'Similarity': allSimilarity, 'Closest Neighbour': allClosestNeighbour}}
    
    return stats

class EvalDataLoader():
    def __init__(self, dataset):
        self.set = dataset 
        self.dataset_path = configs.data.dataset_folder
        self.n_points = 4096
        if configs.data.load_mode == 1:
            self.load_pc = self.load_pc_1
        elif configs.data.load_mode == 2:
            self.load_pc = self.load_pc_2 
        else:
            raise ValueError(f'Error: load mode {configs.data.load_mode} not valid')

    # 4096 points
    def load_pc_1(self, filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        if filename[0] == '/':
            filename = filename[1:]
        file_path = os.path.join(self.dataset_path, filename)

        # if '.bin' in filename:
        #     pc = np.fromfile(file_path, dtype=np.float64)
        #     # coords are within -1..1 range in each dimension
        #     assert pc.shape[0] == self.n_points * 3, "Error in point cloud shape: {}".format(file_path)
        #     pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        #     pc = torch.tensor(pc, dtype=torch.float)
        # elif '.npy' in filename:
        #     pc = np.load(file_path)[:,:3]
        #     assert pc.shape[0] == self.n_points and pc.shape[1] == 3, 'Error in point cloud shape: {}'.format(file_path)
        #     pc = torch.tensor(pc, dtype = torch.float)
        if '.bin' in filename:
            # TODO FIX LATER 2 THINGS -- in 4096 loaders, asking for .bin when only .npy exist
            # getting a permission error
            if os.path.exists(file_path):
                pc = np.fromfile(file_path, dtype=np.float64)
                # coords are within -1..1 range in each dimension
                assert pc.shape[0] == self.n_points * 3, "Error in point cloud shape: {}".format(file_path)
                pc = np.reshape(pc, (pc.shape[0] // 3, 3))
                pc = torch.tensor(pc, dtype=torch.float)
            # if .bin asked for and .npy exists
            elif os.path.exists(file_path.replace(".bin", ".npy")):
                # permission denied on /datasets/work/d61-eif/source/incrementalPointClouds/MulRan/DCC/DCC_02/Ouster/1566533860991046993.npy
                if 'DCC/DCC_02/Ouster/1566533860991046993.npy' not in file_path.replace(".bin", ".npy"):
                    pc = np.load(file_path.replace(".bin", ".npy"))[:,:3]
                    assert pc.shape[0] == self.n_points and pc.shape[1] == 3, 'Error in point cloud shape: {}'.format(file_path.replace(".bin", ".npy"))
                    pc = torch.tensor(pc, dtype = torch.float)
                else:
                    pc = np.load(file_path.replace("1566533860991046993.bin", "1566534559484114597.npy"))[:,:3]
                    assert pc.shape[0] == self.n_points and pc.shape[1] == 3, 'Error in point cloud shape: {}'.format(file_path.replace("1566533860991046993.bin", "1566534559484114597.npy"))
                    pc = torch.tensor(pc, dtype = torch.float)
        elif '.npy' in filename:
            if 'DCC/DCC_02/Ouster/1566533860991046993.npy' not in file_path:
                pc = np.load(file_path)[:,:3]
                assert pc.shape[0] == self.n_points and pc.shape[1] == 3, 'Error in point cloud shape: {}'.format(file_path)
                pc = torch.tensor(pc, dtype = torch.float)
            else:
                pc = np.load(file_path.replace("1566533860991046993.npy", "1566534559484114597.npy"))[:,:3]
                assert pc.shape[0] == self.n_points and pc.shape[1] == 3, 'Error in point cloud shape: {}'.format(file_path)
                pc = torch.tensor(pc, dtype = torch.float)
        return pc

    def load_pc_2(self, filename):
        if filename[0] == '/':
            filename = filename[1:]
        file_path = os.path.join(self.dataset_path, filename)
        xyz = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:,:3]
        largest_dist = np.max(np.abs(xyz[:,:2]))
        xyz = torch.tensor(xyz)
        return xyz

    def __len__(self):
        return len(self.set)
    
    def __getitem__(self, idx):
        x = self.load_pc(self.set[idx]['query'])
        return x


def get_eval_dataloader(set):
    
    eval_dataset = EvalDataLoader(set)
    
    def collate_fn(data_list):
        clouds = [e for e in data_list]
        labels = [e[1] for e in data_list]
        
        if configs.data.load_mode == 1:
            batch = torch.stack(data_list, 0)
        elif configs.data.load_mode == 2:
            batch = data_list
        else:
            raise ValueError(f'Error: load mode {configs.data.load_mode} not valid')

        coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=configs.model.mink_quantization_size)
                    for e in batch]
        bcoords = ME.utils.batched_coordinates(coords)
        feats = torch.ones((bcoords.shape[0], 1), dtype = torch.float32)
        batch = {'coords': bcoords, 'features': feats, 'cloud': batch}
        return batch 
    
    dataloader = DataLoader(
        eval_dataset,
        batch_size = configs.eval.batch_size,
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

    for idx, batch in tqdm(enumerate(dataloader), total = len(set) // configs.eval.batch_size + 1, desc = 'Getting Latent Vectors'):
        batch = {k:v.cuda() for k,v in batch.items()}
        embedding = model(batch)
        embedding = list(embedding)
        if configs.train.loss.normalize_embeddings:
            embedding = [torch.nn.functional.normalize(x, p=2, dim=-1) for x in embedding]
        for x in embedding:
            embeddings_l.append(x.detach().cpu().numpy())
        del batch, embedding
    embeddings = np.vstack(embeddings_l)

    return embeddings


def get_latent_vectors_STUN(model, set, device):
    # Adapted from original PointNetVLAD code

    model.eval()
    embeddings_l = []
    var_l = []
    # Create eval dataloader
    dataloader = get_eval_dataloader(set)

    for idx, batch in tqdm(enumerate(dataloader), total = len(set) // configs.eval.batch_size + 1, desc = 'Getting Latent Vectors'):
        batch = {k:v.cuda() for k,v in batch.items()}
        embedding, var = model(batch)
        embedding = list(embedding)
        var = list(var)
        ## L2 NORMALIZE THE EMBEDDINGS
        if configs.train.loss.normalize_embeddings:
            embedding = [torch.nn.functional.normalize(x, p=2, dim=-1) for x in embedding]
        for x in embedding:
            embeddings_l.append(x.detach().cpu().numpy())
        for x in var:
            var_l.append(x.detach().cpu().numpy())
    embeddings = np.vstack(embeddings_l)
    vars = np.vstack(var_l)
    del embedding
    del var
    torch.cuda.empty_cache()
    return embeddings, vars

