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
from sklearn import metrics

import scipy.spatial as spatial

from misc.utils import MinkLocParams
from models.model_factory import model_factory
from eval.ue_metrics import get_uncertainty_results

from torchpack.utils.config import configs 
from torch.utils.data import DataLoader
import gc

def get_embeddings(modelList, set, device, unc_method):
    #gets the embedding given every model in modelList

    mean_emb = []
    variance_emb = []

    # stun and pfe will have embeddings containing both mean and variance cosine sim
    if unc_method == 'STUN' or unc_method == 'pfe': 
        for model in modelList:
            mean, variance = get_latent_vectors_STUN(model, set, device)
            mean_emb.append(mean)
            variance_emb.append(variance)
    else: # embeddings just have mean cosine sim
        for model in modelList:
            mean = get_latent_vectors(model, set, device) 
            variance =  mean 
            mean_emb.append(mean)
            variance_emb.append(variance)

    return mean_emb, variance_emb 


def evaluate_dataset(modelList, device, database_sets, query_sets, unc_method = '', silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(25)

    count = 0

    similarity = []
    top1_correct = []
    top1_similarity = []
    top_ClosestNeighbour = []
    one_percent_recall = []
    database_embeddings = []
    database_vars = []
    query_embeddings = []
    query_vars = []

    for model in modelList:
        model.eval()

    for set in tqdm(database_sets, disable=silent):
        mean, variance = get_embeddings(modelList, set, device, unc_method)
        database_embeddings.append(mean)
        database_vars.append(variance)

    for set in tqdm(query_sets, disable=silent):
        mean, variance = get_embeddings(modelList, set, device, unc_method)
        query_embeddings.append(mean)
        # stun uses mean of the variance for each query as uncertainty 
        query_vars.append(variance)

    allAccuracy = [] #was the top-1 correct or incorrect
    allMeanQueryVariance = [] #mean variance of each query

    allSimilarity = [] #the top-1 cosine similarity
    allDistance = [] #the top-1 euclidean distance
    allClosestNeighbour = []

    for m in range(len(query_sets)):
        for n in range(len(query_sets)):
            if m == n:
                continue
                
            #for each model, find the distance of the query vectors to all database vectors, then find top 25
            list_distances = []

            for sample_num, database_output in enumerate(database_embeddings[m]):
                queries_output = query_embeddings[n][sample_num]

                database_nbrs = KDTree(database_output)
                num_neighbors = len(database_output)

                model_distances = []
                for i in range(len(queries_output)):
                    # i is query element ndx

                    query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
                    true_neighbors = query_details[m]
                    if len(true_neighbors) == 0:
                        # account for queries with no true neighbors
                        model_distances += [np.zeros(num_neighbors)]
                        continue
  
                    distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
                    
                    distances = distances[0]
                    indices = indices[0]

                    newIdxes = [np.where(indices == x)[0][0] for x in range(len(indices))]
                    orderedDistances = distances[newIdxes] #return to the order of the database, rather than the order of closest distance

                    model_distances += [orderedDistances]
                list_distances += [model_distances] #add all the distances for this model to the list

            list_distances = np.array(list_distances)
            #find the average distance from the dropout for each query for all database vectors
            avg_distances = np.mean(list_distances, axis = 0)

            pair_recall, pair_similarity, pair_opr, top1_accuracy, top1_similarity, top1_distance, top1_meanQueryVariance, top_ClosestNeighbour = get_recall(m, n, avg_distances, database_sets, query_sets, database_embeddings, database_vars, query_embeddings, query_vars, unc_method)

            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

            # Correct or Incorrect
            allAccuracy += top1_accuracy
            # Variance of Query
            allMeanQueryVariance += top1_meanQueryVariance

            allDistance += top1_distance
            allSimilarity += top1_similarity

            allClosestNeighbour += top_ClosestNeighbour

    # allAccuracy       A x D   bool    np array of correct or incorrect top 1 place matches   
    # allDistance       A x D           np array of cosine similarity for top 1 place matches
    # allSimilarity     A x D           np array of euclidean distance for top 1 place matches
    # allMeanQueryVariance              np array of mean query variances used to estimate uncertainty

    unc_results = get_uncertainty_results(allSimilarity,allAccuracy,False) 
    aupr_in = unc_results['auprIn']*100
    ausc = unc_results['auSC']*100
    auroc = unc_results['auroc']*100

    ave_recall = recall / count
    ave_recall_1 = ave_recall[0]
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)

    stats = {'Recall@1%': ave_one_percent_recall, 'Recall@1': ave_recall_1, 'AuPR': aupr_in, 'auSC': ausc, 'AuROC': auroc,
        'Top1': {'Accuracy': allAccuracy, 'Similarity': allSimilarity, 'Closest Neighbour': allClosestNeighbour}}
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

    for idx, batch in enumerate(dataloader):
        batch = {k:v.cuda() for k,v in batch.items()}
        embedding = model(batch)
        embedding = list(embedding)
        ## L2 NORMALIZE THE EMBEDDINGS
        if configs.train.loss.normalize_embeddings:
            embedding = [torch.nn.functional.normalize(x, p=2, dim=-1) for x in embedding]
        for x in embedding:
            embeddings_l.append(x.detach().cpu().numpy())
    embeddings = np.vstack(embeddings_l)
    del embedding
    torch.cuda.empty_cache()
    return embeddings

def get_latent_vectors_STUN(model, set, device):
    # Adapted from original PointNetVLAD code

    model.eval()
    embeddings_l = []
    var_l = []
    # Create eval dataloader
    dataloader = get_eval_dataloader(set)

    for idx, batch in enumerate(dataloader):
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


def get_recall(m, n, allDistances, database_sets, query_sets, database_vectors, database_variance, query_vectors, queries_variance, unc_method):
    # Original PointNetVLAD code
    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(allDistances[m])/100.0)), 1)

    top1_accuracy = []
    top1_similarity = []
    top_closestNeighbour = []
    top1_distance = []
    top1_meanQueryVariance = []

    num_evaluated = 0

    database_output = database_vectors[m]
    database_output_variance = database_variance[m]
    queries_output = query_vectors[n]
    queries_output_variance = queries_variance[n]

    for i in range(len(allDistances)): #size 
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        # print(query_details.keys())
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue

        num_evaluated += 1

        distances = allDistances[i]
        orderedIndices = np.argsort(distances) #sorts the indices of the database in order of closest distances to furthest
        indices = orderedIndices[:25] #we want the top 25
        top1_index = indices[0] #number 1 top match

        for j in range(len(indices)):
            if indices[j] in true_neighbors:
                recall[j] += 1
                break

        all_cosine = []
        all_variance = []

        # Find number 1 top matching cosine similarity
        for mIdx in range(len(database_output)):

            query_embedding = queries_output[mIdx][i]
            query_embedding_variance = queries_output_variance[mIdx][i]
            
            top_database_embedding = database_output[mIdx][top1_index]

            cosine_similarity = np.dot(query_embedding, top_database_embedding)/(np.linalg.norm(query_embedding)*np.linalg.norm(top_database_embedding))
            
            # Find top 1 matching variance of database entry
            all_cosine += [cosine_similarity] 
            # Find mean of query variance embedding to get uncertainty
            meanVarianceQuery = np.mean(query_embedding_variance)
            all_variance += [meanVarianceQuery]
        
        # ACCOUNT FOR ENSEMBLES - average out cosine similarity and variance
        avg_cosine = np.mean(all_cosine) #np.var(all_cosine) for variance calculation
        if unc_method == 'STUN':
            avg_variance = all_variance[0]
        elif unc_method == 'pfe':
            #calculate similarity score between query and top database match
            aEmbed = query_embedding 
            # aVar = query_embedding_variance
            aVar = np.exp(query_embedding_variance)
            pEmbed = top_database_embedding #database embedding
            # pVar = database_output_variance[mIdx][top1_index] #database variance
            pVar = np.exp(database_output_variance[mIdx][top1_index])

            meanDiffSq = (aEmbed-pEmbed)**2
            varSum = aVar + pVar
            logVarSum = np.log(varSum)
            similarity = -0.5*np.sum((meanDiffSq/(varSum+1e-10 ))+logVarSum)
            avg_variance = similarity
        else:
            avg_variance = np.var(all_cosine)

        if top1_index in true_neighbors:
            top1_similarity_score.append(avg_cosine)

        correct = top1_index in true_neighbors

        top25_correct = [idx in true_neighbors for idx in indices]
        top25_correct = np.array(top25_correct)
        if np.sum(top25_correct) == 0: #there were no TP in top25 neighbours
            closest_neighbour = -1
        else:
            correct_idxes = np.where(top25_correct == 1)
            #this will grab the lowest idx that has a TP. index starts at 0, so we add a 1 to make it easier for us later on.
            closest_neighbour = 1+correct_idxes[0][0]

        top_closestNeighbour += [int(closest_neighbour)]
        top1_accuracy += [correct]
        top1_similarity += [avg_cosine.astype(float)]
        top1_distance += [distances[top1_index].astype(float)]
        top1_meanQueryVariance += [avg_variance.astype(float)]
    
        if len(list(set(indices[0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100

    return recall, top1_similarity_score, one_percent_recall, top1_accuracy, top1_similarity, top1_distance, top1_meanQueryVariance, top_closestNeighbour
