# Author: Jacek Komorowski
# Warsaw University of Technology

# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

from eval.evaluate_oxford import evaluate_dataset as eval_oxford 
from eval.evaluate_inrun import evaluate_dataset as eval_inrun  
import pandas as pd 
from torchpack.utils.config import configs
import os 
import argparse 
from misc.utils import load_pickle
from models.model_factory import model_factory
import torch 
import json
import numpy as np

def evaluate(modelList, unc_method = ''):
    assert len(configs.data.eval_database_files) == len(configs.data.eval_query_files)
    if not isinstance(modelList, list):
        modelList = [modelList]
    results_dict = {}
    seen_keys = []
    uncertainty_dict = {}

    shorterStats_total = {}

    for database_file, query_file in zip(configs.data.eval_database_files, configs.data.eval_query_files):
        location_name = database_file.split('.')[0].replace('_database', '').split('/')[-1]

        shorterStats = {}

        if query_file != None: # Oxford style eval
            database_sets = load_pickle(database_file)
            query_sets = load_pickle(query_file)
            stats = eval_oxford(modelList, 'cuda', database_sets, query_sets, unc_method)

            for field in ['Recall@1','AuPR','auSC','AuROC']:
                shorterStats[field] = stats[field]

            uncertainty_dict[location_name] = stats['Top1']

        else: # In-run (i.e. MulRan, KITTI) style eval **NO QUERY FILES**
            database_sets = load_pickle(database_file)
            stats = eval_inrun(modelList, 'cuda', database_sets, unc_method)

            for field in ['Recall@1','AuPR','auSC','AuROC']:
                shorterStats[field] = stats[field]
            
            uncertainty_dict[location_name] = stats['Top1']
    
        results_dict[location_name] = shorterStats 

    results_df = pd.DataFrame.from_dict(results_dict)
    results_df = results_df.transpose()
    return results_df, uncertainty_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--save_results', type = str, default = None, help = 'Path & filename to save csv of results to')
    parser.add_argument('--uncertainty_method', type=str, required=False, default='', help='default = Baseline MinkLoc3D Architecture. Options: STUN, pfe, dropout')
    parser.add_argument('--ensemble_models', type=int, required=False, default = 5, help='Number of models to be used for the ensemble')
    parser.add_argument('--dropout_passes', type=int, required=False, default = 5, help='Number of forward passes for mc dropout')
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)

    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights

    if args.uncertainty_method == '' or args.uncertainty_method == 'dropout': 
        model = model_factory(args.uncertainty_method)
        if args.weights is not None:
            assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
            print('Loading weights: {}'.format(args.weights))
            model.load_state_dict(torch.load(args.weights, map_location='cuda'))
        
        model.to('cuda')

        if args.uncertainty_method == '':
            modelList = [model]
        elif args.uncertainty_method == 'dropout':
            modelList = [model for i in range(args.dropout_passes)] #same model tested m times

    
    elif args.uncertainty_method == 'STUN':
        model = model_factory("stun_student")
        if args.weights is not None:
            assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
            print('Loading weights: {}'.format(args.weights))
            model.load_state_dict(torch.load(args.weights, map_location='cuda'))
        
        model.to('cuda')
        modelList = [model]

    elif args.uncertainty_method in ['pfe','PFE','probabilistic face embeddings']:
        args.uncertainty_method = 'pfe'

        model = model_factory("pfe")
        if args.weights is not None:
            assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
            print('Loading weights: {}'.format(args.weights))
            model.load_state_dict(torch.load(args.weights, map_location='cuda'))
        
        model.to('cuda')
        modelList = [model]

    elif args.uncertainty_method == 'ensemble': 
        modelList = []
        for mIdx in range(1, args.ensemble_models+1): # m models in the ensemble
            model = model_factory(args.uncertainty_method)
            if args.weights is not None:
                ensWeight = args.weights.replace('_1', '_'+str(mIdx))
                assert os.path.exists(ensWeight), 'Cannot open network weights: {}'.format(ensWeight)
                print('Loading weights: {}'.format(ensWeight))
                model.load_state_dict(torch.load(ensWeight, map_location='cuda'))
            
            model.to('cuda')
            modelList += [model]
    else:
        print('Uncertainty method not implemented')
        exit()

    stats, uncertainty_results = evaluate(modelList, args.uncertainty_method)
    print(f'\n{stats}')

    if configs.save_results != None and configs.save_results != 'None':
        assert os.path.exists(os.path.dirname(configs.save_results)), 'Error: Directory for configs.save_results does not exist'
        stats.to_csv(configs.save_results)
        print(f'\nSaved results to {configs.save_results}')

        uncertainty_dir = configs.save_results.replace('.csv', '')
        uncertainty_dir = uncertainty_dir.replace('/results/','/results_json/')
        with open(f'{uncertainty_dir}_uncertaintyData.json', 'w') as f:
            json.dump(uncertainty_results, f)
