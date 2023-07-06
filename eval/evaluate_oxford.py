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

from misc.utils import MinkLocParams
from models.model_factory import model_factory

from torchpack.utils.config import configs 
from torch.utils.data import DataLoader





def evaluate_dataset(model, device, database_sets, query_sets, silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    similarity = []
    all_correct = []
    all_incorrect = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()


    for set in tqdm(database_sets, disable=False):
        database_embeddings.append(get_latent_vectors(model, set, device))

    for set in tqdm(query_sets, disable=False):
        query_embeddings.append(get_latent_vectors(model, set, device))

    for i in range(len(query_sets)):
        for j in range(len(query_sets)):
            if i == j:
                continue
            pair_recall, pair_similarity, pair_opr, correct, incorrect = get_recall(i, j, database_embeddings, query_embeddings, query_sets,
                                                                database_sets)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)
            for x in correct:
                all_correct.append(x)
            for x in incorrect:
                all_incorrect.append(x)

    print('\ncorrect predictions size : {}, cosine sim mean avg : {}'.format(len(all_correct), np.mean(all_correct)))
    print('incorrect predictions size : {}, cosine sim mean avg : {}\n'.format(len(all_incorrect), np.mean(all_incorrect)))


    ave_recall = recall / count
    ave_recall_1 = ave_recall[0]
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'Recall@1%': ave_one_percent_recall, 'Recall@1': ave_recall_1}
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

    for idx, batch in enumerate(dataloader):
        batch = {k:v.cuda() for k,v in batch.items()}
        embedding = model(batch)
        embedding = list(embedding)
        if configs.train.loss.normalize_embeddings:
            embedding = [torch.nn.functional.normalize(x, p=2, dim=1) for x in embedding]
        for x in embedding:
            embeddings_l.append(x.detach().cpu().numpy())
    embeddings = np.vstack(embeddings_l)

    return embeddings


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    # Original PointNetVLAD code
    
    # print(len(database_vectors), len(query_vectors))
    # print(len(database_vectors[0]), len(query_vectors[0]))
    # print(len(database_vectors[0][0]), len(query_vectors[0][0]))

    database_output = database_vectors[m]
    queries_output = query_vectors[n]


    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    correct = []
    incorrect = []

    num_evaluated = 0

    for i in range(len(queries_output)): #size 
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        # print(query_details.keys())
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                    correct.append(similarity)
                recall[j] += 1
                break
            else:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    incorrect.append(similarity)

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1
    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    
    return recall, top1_similarity_score, one_percent_recall, correct, incorrect

