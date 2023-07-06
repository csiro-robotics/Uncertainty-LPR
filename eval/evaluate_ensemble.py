# Evaluate an ensemble of models
# Option of saving cosine similarity grouped by correct and incorrect predictions to .txt file

from calendar import c
from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import torch
from torch.functional import _consecutive_return_inverse
import tqdm
import MinkowskiEngine as ME
import random
import json
import ast

from misc.utils import MinkLocParams
from models.model_factory import model_factory


def evaluate(modelList, device, params, silent=True):
    # Run evaluation on all eval datasets
    assert len(params.eval_database_files) == len(params.eval_query_files)

    stats = {}
    cosinesim = {}

    for database_file, query_file in zip(params.eval_database_files, params.eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp, correct, incorrect, ave_one_percent_recall = evaluate_dataset_ensembles(modelList, device, params, database_sets, query_sets, silent=silent)

        cosinesim[location_name] = {'recall': ave_one_percent_recall, 'correct': correct,'incorrect': incorrect}

        stats[location_name] = temp

    return stats, cosinesim

def evaluate_dataset_ensembles(modelList, device, params, database_sets, query_sets, silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    similarity = [] 
    all_correct = []
    all_incorrect = []
    one_percent_recall = []

    ensemble_database_embeddings = []
    ensemble_query_embeddings = []
    for modelIdx in range(len(modelList)):
        modelList[modelIdx].eval()

        database_embeddings = []
        query_embeddings = []

        for set in tqdm.tqdm(database_sets, disable=silent):
            database_embeddings.append(get_latent_vectors(modelList[modelIdx], set, device, params))

        for set in tqdm.tqdm(query_sets, disable=silent):
            query_embeddings.append(get_latent_vectors(modelList[modelIdx], set, device, params))

        ensemble_database_embeddings.append(database_embeddings)
        ensemble_query_embeddings.append(query_embeddings)

    for m in range(len(query_sets)):
        for n in range(len(query_sets)):
            if m == n:
                continue
                
            #for each model, find the distance of the query vectors to all database vectors, then find top 25
            ensemble_distances = []
            for modelIdx in range(len(modelList)):
                queries_output = ensemble_query_embeddings[modelIdx][n]
                database_output = ensemble_database_embeddings[modelIdx][m]

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
                ensemble_distances += [model_distances] #add all the distances for this model to the ensemble list

            ensemble_distances = np.array(ensemble_distances)
            #find the average distance from the ensemble for each query for all database vectors
            avg_distances = np.mean(ensemble_distances, axis = 0)

            pair_recall, pair_similarity, pair_opr, correct, incorrect = get_recall(m, n, avg_distances, query_sets,
                                                                database_sets, ensemble_database_embeddings, ensemble_query_embeddings)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

            # correct and incorrect prediction arrays to be plot in histogram
            for x in correct:
                all_correct.append(x)
            for x in incorrect:
                all_incorrect.append(x)

    print('\ncorrect predictions size : {}, cosine sim mean avg : {}'.format(len(all_correct), np.mean(all_correct)))
    print('incorrect predictions size : {}, cosine sim mean avg : {}\n'.format(len(all_incorrect), np.mean(all_incorrect)))

    ave_recall = recall / count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
            'average_similarity': average_similarity}
    return stats, all_correct, all_incorrect, ave_one_percent_recall

def get_recall(m, n, allDistances, query_sets, database_sets, ensemble_database_embeddings, ensemble_query_embeddings):
    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    database_output = allDistances[m]
    queries_output = query_sets[n]
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0

    # JUST FOR TOP 1 VAUES
    #for correct predictions
    correct = []
    incorrect = []

    for i in range(len(allDistances)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
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
                if j == 0: 
                    similarity = distances[indices[j]]
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break
        
        all_cosine = []
        # Find number 1 top matching cosine similarity
        for mIdx in range(len(ensemble_query_embeddings)):
            query_embedding = ensemble_query_embeddings[mIdx][n][i]
            top_database_embedding = ensemble_database_embeddings[mIdx][m][top1_index]
            cosine_similarity = np.dot(query_embedding, top_database_embedding)
            all_cosine += [cosine_similarity]

        ensemble_cosine = np.mean(all_cosine) #np.var(all_cosine) for variance calculation

        if top1_index in true_neighbors:
            correct.append(ensemble_cosine) #if top 1 match is correct predict
        else:
            incorrect.append(ensemble_cosine) #if top 1 match is incorrect predict

        if len(list(set(indices[0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, top1_similarity_score, one_percent_recall, correct, incorrect

def load_pc(file_name, params):
    # returns Nx3 matrix
    file_path = os.path.join(params.dataset_folder, file_name)
    pc = np.fromfile(file_path, dtype=np.float64)
    # coords are within -1..1 range in each dimension
    assert pc.shape[0] == params.num_points * 3, "Error in point cloud shape: {}".format(file_path)
    pc = np.reshape(pc, (pc.shape[0] // 3, 3))
    pc = torch.tensor(pc, dtype=torch.float)
    return pc


def get_latent_vectors(model, set, device, params):
    # Adapted from original PointNetVLAD code

    model.eval()
    embeddings_l = []
    for elem_ndx in set:
        x = load_pc(set[elem_ndx]["query"], params)

        with torch.no_grad():
            # coords are (n_clouds, num_points, channels) tensor
            coords = ME.utils.sparse_quantize(coordinates=x,
                                              quantization_size=params.model_params.mink_quantization_size)
            bcoords = ME.utils.batched_coordinates([coords])
            # Assign a dummy feature equal to 1 to each point
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
            feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
            batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

            embedding = model(batch)
            # embedding is (1, 1024) tensor
            #if params.normalize_embeddings:
            #    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize embeddings
            #else:
            params.normalize_embeddings = True #edited to ensure embeddings are normalised
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    return embeddings

def print_eval_stats(stats):
    #save 
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'], stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])



# Save cosine similarity to .txt file for easier plotting and visualisation without evaluating each time
def save_cosinesim_dict(num_models, ensemble_cosinesim):
    # read in .txt file containing cosine similiarity dict

    if os.path.isfile('cosinesim_dict.txt'):
        with open('cosinesim_dict.txt') as json_file:
            d = json.load(json_file)
            cosine_dict = ast.literal_eval(d)
    else: 
        cosine_dict = {}

    # create/replace ensemble values
    cosine_dict[str(num_models)] = ensemble_cosinesim

    # create/update txt file
    with open('cosinesim_dict.txt', 'w') as json_file:
        json_file.write(json.dumps(str(cosine_dict)))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Ensemble of 3-5 Models')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--num_models', type=int, required=True, help='How many models in the ensemble?')
    parser.add_argument('--model', type=str, required=True, help='Base model directory')
    parser.add_argument('--save_cosinesim', type=bool, required=False, default=False, help='Save cosine similarity to .txt file as dict? True/False (default False)')

    args = parser.parse_args()

    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    
    en_weights = [f'{args.model}{i+1}{".pth"}' for i in range(args.num_models)]

    print(f'Testing with {args.num_models} models in the ensemble')
    for m in en_weights:
        print('Weights: {}'.format(m))
    print('')

    params = MinkLocParams(args.config, args.model_config)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print('Device: {}'.format(device))

    # Evaluate each model and print
    models = [None for i in range(args.num_models)]
    for mIdx, m in enumerate(en_weights):
        models[mIdx] = model_factory(params)
        
        print('Loading weights: {}'.format(m))
        models[mIdx].load_state_dict(torch.load(m, map_location=device))

        models[mIdx].to(device)

    stats, cosinesim = evaluate(models, device, params, silent=False)
    print_eval_stats(stats)
    print()

    # to save cosine sim
    if args.save_cosinesim:
        save_cosinesim_dict(args.num_models,cosinesim)
