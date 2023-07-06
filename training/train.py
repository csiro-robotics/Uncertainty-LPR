# Author: Jacek Komorowski
# Warsaw University of Technology

import argparse
import torch
import os

import training.trainer as trainer

from datasets.dataset_utils import make_dataloaders
from torchpack.utils.config import configs 

if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Train Minkowski Net embeddings using BatchHard negative mining')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--uncertainty_method', type=str, required=False, default='none', help='Uncertainty estimation method to be used. default=none. Options: STUN, PFE, MC Dropout')
    parser.add_argument('--teacher_net', type=str, required=False, default=os.path.join(ROOT_DIR, '../weights/minkloc_oxford.pth'), help='If using STUN, this is the teacher net model to be loaded. default = weights/minkloc_oxford.pth')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)

    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)
    print(f'\n{configs}\n')
    print('Training config path: {}'.format(args.config))
    print('Debug mode: {}'.format(args.debug))
    print('Visualize: {}'.format(args.visualize))

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    dataloaders = make_dataloaders(debug=args.debug)

    if args.uncertainty_method in ['STUN','stun']:
        print('\n-------------\nSTUN training\n-------------')
        print("Loading teacher net: {}\n-------------\n".format(args.teacher_net))
        trainer.do_train_STUN(args.teacher_net, dataloaders, debug=args.debug, visualize=args.visualize)
    elif args.uncertainty_method in ['pfe','PFE','probabilistic face embeddings']:
        print('\n------------\nPFE training\n------------\n')
        trainer.do_train_PFE(dataloaders, args.teacher_net, debug=args.debug, visualize=args.visualize)
    elif args.uncertainty_method in ['dropout']:
        print('\n------------\nDropout training\n------------\n')
        trainer.do_train(dataloaders, debug=args.debug, visualize=args.visualize)
    else:
        print("\n-----------------------\nNo uncertainty training\n-----------------------\n")
        trainer.do_train(dataloaders, debug=args.debug, visualize=args.visualize)
