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

def evaluate(model):
    assert len(configs.data.eval_database_files) == len(configs.data.eval_query_files)
    results_dict = {}
    seen_keys = []

    for database_file, query_file in zip(configs.data.eval_database_files, configs.data.eval_query_files):
        location_name = database_file.split('.')[0].replace('_database', '').split('/')[-1]
        if query_file != None: # Oxford style eval
            database_sets = load_pickle(database_file)
            query_sets = load_pickle(query_file)

            stats = eval_oxford(model, 'cuda', database_sets, query_sets)
            
        else: # In-run (i.e. MulRan, KITTI) style eval
            database = load_pickle(database_file)
            stats = eval_inrun(model, 'cuda', database)

        results_dict[location_name] = stats 

    results_df = pd.DataFrame.from_dict(results_dict)
    results_df = results_df.transpose()
    return results_df 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--save_results', type = str, default = None, help = 'Path & filename to save csv of results to')

    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)


    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('')

    model = model_factory()
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location='cuda'))
    
    model.to('cuda')
    stats = evaluate(model)
    print(stats)

    if args.save_results == None or args.save_results == 'None':
        assert os.path.exists(os.path.dirname(args.save_results)), 'Error: Directory for args.save_results does not exist'
        stats.to_csv(args.save_results)
