# Author: Jacek Komorowski
# Warsaw University of Technology

# Train on Oxford dataset (from PointNetVLAD paper) using BatchHard hard negative mining.

import os
from datetime import datetime
import numpy as np
import torch
import pickle
import tqdm
import pathlib

from torch.utils.tensorboard import SummaryWriter

from eval.eval_wrapper import evaluate
from misc.utils import MinkLocParams, get_datetime
from models.loss import make_loss
from models.model_factory import model_factory

from torchpack.utils.config import configs 


def print_stats(stats, phase):
    if 'num_pairs' in stats:
        # For batch hard contrastive loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Pairs per batch (all/non-zero pos/non-zero neg): {:.1f}/{:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pairs'],
                       stats['pos_pairs_above_threshold'], stats['neg_pairs_above_threshold']))
    elif 'num_triplets' in stats:
        # For triplet loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Triplets per batch (all/non-zero): {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_triplets'],
                       stats['num_non_zero_triplets']))
    elif 'num_pos' in stats:
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   #positives/negatives: {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pos'], stats['num_neg']))

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if 'pos_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Pos loss: {:.4f}  Neg loss: {:.4f}'
        l += [stats['pos_loss'], stats['neg_loss']]
    if len(l) > 0:
        print(s.format(*l))


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def do_train_STUN(dataloaders, debug=False, visualize=False):
    # Create model class
    s = get_datetime()
    model = model_factory()
    model_name = 'model_' + configs.model.name + '_' + s
    print('Model name: {}'.format(model_name))

    print(configs.save_path)
    assert configs.save_path != None, 'Error: Please specify save path in input arguments by setting "save_path"'
    if not os.path.exists(configs.save_path):
        os.makedirs(configs.save_path)

    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # Move the model to the proper device before configuring the optimizer
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
    else:
        device = "cpu"

    print('Model device: {}'.format(device))

    loss_fn = make_loss()

    # Training elements
    if configs.train.optimizer.weight_decay is None or configs.train.optimizer.weight_decay == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.train.optimizer.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.train.optimizer.lr, weight_decay=configs.train.optimizer.weight_decay)

    if configs.train.optimizer.scheduler is None:
        scheduler = None
    else:
        if configs.train.optimizer.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.train.optimizer.epochs+1,
                                                                   eta_min=configs.train.optimizer.min_lr)
        elif configs.train.optimizer.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, configs.train.optimizer.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(configs.train.optimizer.scheduler))

    ###########################################################################
    # Initialize TensorBoard writer
    ###########################################################################

    now = datetime.now()
    logdir = os.path.join(configs.save_path, 'tf_logs')
    # logdir = os.path.join("../tf_logs", now.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(logdir)

    ###########################################################################
    #
    ###########################################################################

    is_validation_set = 'val' in dataloaders
    if is_validation_set:
        phases = ['train', 'val']
    else:
        phases = ['train']

    # Training statistics
    stats = {'train': [], 'val': [], 'eval': []}

    for epoch in range(1, configs.train.optimizer.epochs + 1):

        if debug == True and epoch > 3:
            break 

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_stats = []  # running stats for the current epoch

            count_batches = 0
            
            for idx, (batch, positives_mask, negatives_mask) in enumerate(dataloaders[phase]):
                if idx == 0:
                    pbar = tqdm.tqdm(total = len(dataloaders[phase]), desc = f'Epoch {epoch}: Loss 0.000')
                pbar.update(1)
                # batch is (batch_size, n_points, 3) tensor
                # labels is list with indexes of elements forming a batch
                count_batches += 1
                batch_stats = {}

                if debug and count_batches > 2:
                    break

                batch = {e: batch[e].to(device) for e in batch}

                n_positives = torch.sum(positives_mask).item()
                n_negatives = torch.sum(negatives_mask).item()
                if n_positives == 0 or n_negatives == 0:
                    # Skip a batch without positives or negatives
                    print('WARNING: Skipping batch without positive or negative examples')
                    continue

                optimizer.zero_grad()
                if visualize:
                    #visualize_batch(batch)
                    pass

                with torch.set_grad_enabled(phase == 'train'):
                    # Compute embeddings of all elements
                    embeddings = model(batch)
                    loss, temp_stats, _ = loss_fn(embeddings, positives_mask, negatives_mask)

                    temp_stats = tensors_to_numbers(temp_stats)
                    batch_stats.update(temp_stats)
                    batch_stats['loss'] = loss.item()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_stats.append(batch_stats)
                torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
                pbar.set_description(f'Epoch {epoch}: Loss {loss.item():.3f}')

            # ******* PHASE END *******
            if phase == 'train':
                pbar.close()
            # Compute mean stats for the epoch
            epoch_stats = {}
            for key in running_stats[0].keys():
                temp = [e[key] for e in running_stats]
                epoch_stats[key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(epoch_stats, phase)

        # ******* EPOCH END *******
        print('')

        if scheduler is not None:
            scheduler.step()

        loss_metrics = {'train': stats['train'][-1]['loss']}
        if 'val' in phases:
            loss_metrics['val'] = stats['val'][-1]['loss']
        writer.add_scalars('Loss', loss_metrics, epoch)

        if 'num_triplets' in stats['train'][-1]:
            nz_metrics = {'train': stats['train'][-1]['num_non_zero_triplets']}
            if 'val' in phases:
                nz_metrics['val'] = stats['val'][-1]['num_non_zero_triplets']
            writer.add_scalars('Non-zero triplets', nz_metrics, epoch)

        elif 'num_pairs' in stats['train'][-1]:
            nz_metrics = {'train_pos': stats['train'][-1]['pos_pairs_above_threshold'],
                          'train_neg': stats['train'][-1]['neg_pairs_above_threshold']}
            if 'val' in phases:
                nz_metrics['val_pos'] = stats['val'][-1]['pos_pairs_above_threshold']
                nz_metrics['val_neg'] = stats['val'][-1]['neg_pairs_above_threshold']
            writer.add_scalars('Non-zero pairs', nz_metrics, epoch)

        if configs.train.batch_expansion_th is not None:
            # Dynamic batch expansion
            epoch_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' not in epoch_train_stats:
                print('WARNING: Batch size expansion is enabled, but the loss function is not supported')
            else:
                # Ratio of non-zero triplets
                rnz = epoch_train_stats['num_non_zero_triplets'] / epoch_train_stats['num_triplets']
                if rnz < configs.train.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()

    print('')

    # Save final model weights
    # final_model_path = model_pathname + '_final.pth'
    final_model_path = os.path.join(configs.save_path, 'checkpoint_final.pth')
    torch.save(model.state_dict(), final_model_path)

    stats = {'train_stats': stats} #TODO

    # Evaluate the final model
    print('Beginning Evaluation!')
    model.eval()
    final_eval_stats = evaluate(model)
    print('Final model: ')
    print(final_eval_stats)
    stats_save_path = os.path.join(configs.save_path, 'final_results.csv')
    final_eval_stats.to_csv(stats_save_path)
    print(f'Saved results to {stats_save_path}')


#     print('Final model:')
#     print_eval_stats(final_eval_stats)
#     stats['eval'] = {'final': final_eval_stats}
#     print('')

#     # Pickle training stats and parameters
#     pickle_path = model_pathname + '_stats.pickle'
#     pickle.dump(stats, open(pickle_path, "wb"))

#     # Append key experimental metrics to experiment summary file
#     export_eval_stats("experiment_results.txt", final_eval_stats)


# # def export_eval_stats(file_name, eval_stats):
# #     s = ''
# #     ave_1p_recall_l = []
# #     ave_recall_l = []
# #     # Print results on the final model
# #     with open(file_name, "a") as f:
# #         for ds in ['oxford', 'university', 'residential', 'business']:
# #             ave_1p_recall = eval_stats[ds]['ave_one_percent_recall']
# #             ave_1p_recall_l.append(ave_1p_recall)
# #             ave_recall = eval_stats[ds]['ave_recall'][0]
# #             ave_recall_l.append(ave_recall)
# #             s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

# #         mean_1p_recall = np.mean(ave_1p_recall_l)
# #         mean_recall = np.mean(ave_recall_l)
# #         s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
# #         f.write(s)


# # def create_weights_folder():
# #     # Create a folder to save weights of trained models
# #     this_file_path = pathlib.Path(__file__).parent.absolute()
# #     temp, _ = os.path.split(this_file_path)
# #     weights_path = os.path.join(temp, 'weights')
# #     if not os.path.exists(weights_path):
# #         os.mkdir(weights_path)
# #     assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
# #     return weights_path
