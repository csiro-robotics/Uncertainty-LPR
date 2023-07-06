# Author: Jacek Komorowski
# Warsaw University of Technology

import argparse
import torch

from training.trainer import do_train
from training.trainer_STUN import do_train_STUN
# from misc.utils import MinkLocParams
from datasets.dataset_utils import make_dataloaders

from torchpack.utils.config import configs 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Minkowski Net embeddings using BatchHard negative mining')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--uncertainty_method', type=str, required=False, default='', help='default = Baseline MinkLoc3D Architecture. Options: STUN')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)

    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)
    print(configs)
    print('Training config path: {}'.format(args.config))
    # print('Model config path: {}'.format(args.model_config))
    print('Debug mode: {}'.format(args.debug))
    print('Visualize: {}'.format(args.visualize))

    # params = MinkLocParams(args.config, args.model_config)
    # params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    dataloaders = make_dataloaders(debug=args.debug)

    if args.uncertainty_method in ['STUN','stun']:
        print('\n----------\nSTUN training\n----------\n')
        do_train_STUN(dataloaders, debug=args.debug, visualize=args.visualize)
    else:
        print("\n----------\nNo uncertainty training\n----------\n")
        do_train(dataloaders, debug=args.debug, visualize=args.visualize)
