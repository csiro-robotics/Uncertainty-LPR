# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import torch
from pytorch_metric_learning import losses, miners, reducers
from pytorch_metric_learning.distances import LpDistance
from torchpack.utils.config import configs 
#import torch.nn.functional as F


def make_loss(uncertainty_method):
    if uncertainty_method=="stun_student":
        # Stun uncertainty-aware loss function
        loss_fn = StunUncertaintyAwareLoss()
    elif uncertainty_method == 'pfe':
        loss_fn = PFELoss(False)
    elif configs.train.loss.name == 'BatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = BatchHardTripletLossWithMasks(configs.train.loss.margin, configs.train.loss.normalize_embeddings)
    elif configs.train.loss.name == 'BatchHardContrastiveLoss':
        loss_fn = BatchHardContrastiveLossWithMasks(configs.train.loss.pos_margin, configs.train.loss.neg_margin, configs.train.loss.normalize_embeddings)
    else:
        print('Unknown loss: {}'.format(configs.train.loss.name))
        raise NotImplementedError
    return loss_fn


class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings, positives_mask, negatives_mask):
        assert embeddings.dim() == 2
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets = self.mine(d_embeddings, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, embeddings, positives_mask, negatives_mask):
        # Based on pytorch-metric-learning implementation
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        self.max_pos_pair_dist = torch.max(hardest_positive_dist).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist).item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows
    

class BatchHardTripletLossWithMasks:
    def __init__(self, margin, normalize_embeddings):
        self.margin = margin
        self.normalize_embeddings = normalize_embeddings
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings, collect_stats=True)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=True, distance=self.distance,
                                                reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets


class BatchHardContrastiveLossWithMasks:
    def __init__(self, pos_margin, neg_margin, normalize_embeddings):
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings, collect_stats=True)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        # We use contrastive loss with squared Euclidean distance
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.ContrastiveLoss(pos_margin=self.pos_margin, neg_margin=self.neg_margin,
                                              distance=self.distance, reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'pos_pairs_above_threshold': self.loss_fn.reducer.reducers['pos_loss'].pos_pairs_above_threshold,
                 'neg_pairs_above_threshold': self.loss_fn.reducer.reducers['neg_loss'].neg_pairs_above_threshold,
                 'pos_loss': self.loss_fn.reducer.reducers['pos_loss'].pos_loss.item(),
                 'neg_loss': self.loss_fn.reducer.reducers['neg_loss'].neg_loss.item(),
                 'num_pairs': 2*len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets

class StunUncertaintyAwareLoss:

    def __call__(self, embeddings_teacher, embeddings, var):
        assert embeddings.dim() == 2
        assert embeddings_teacher.dim() == 2
        assert var.dim() == 2
        loss = self.stunloss(embeddings_teacher, embeddings, var)
        stats = {'loss': loss.item()}
        return loss, stats

    def stunloss(self, embeddings_teacher, embeddings, var):
        if configs.train.loss.normalize_embeddings == True:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings_teacher = torch.nn.functional.normalize(embeddings_teacher, p=2, dim=1)

        nominator = (embeddings_teacher - embeddings)**2
        denominator = 2*var + 1e-6
        regulator = 0.5 * torch.log(var)

        loss = torch.div(nominator, denominator) + regulator # B x D 
        loss = loss.sum(1).sum()

        return loss

class PFELoss:
    def __init__(self, normalize_embeddings):
        self.normalize_embeddings = normalize_embeddings
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings, collect_stats=True)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)

    def __call__(self, embeddings, logVar, positive_mask, negative_mask):
        assert embeddings.dim() == 2
        assert logVar.dim() == 2

        if configs.train.loss.normalize_embeddings == True:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        var = torch.exp(logVar)
   
        #find our triplet pairs
        anchors, positives, negatives = self.miner_fn(embeddings, positive_mask, negative_mask)

        #pfe only uses anchor + positive, extract these here
        anchorEmbeddings = embeddings[anchors]
        positiveEmbeddings = embeddings[positives]

        anchorVars = var[anchors]
        positiveVars = var[positives]
      
        #pass these to the pfe loss
        loss = self.pfeloss(anchorEmbeddings, positiveEmbeddings, anchorVars, positiveVars)
        
        stats = {'loss': loss.item()}
        return loss, stats

    def similarity_scores(self, aEmbed, pEmbed, aVar, pVar):
        #as in the paper, ignoring the constant term because it's irrelevant to the loss
        meanDiffSq = (aEmbed-pEmbed)**2
        varSum = aVar + pVar

        logVarSum = torch.log(varSum)
        similarity = -0.5*torch.sum((meanDiffSq/(varSum+1e-10 ))+logVarSum, dim = 1)

        return similarity

    def pfeloss(self, aEmbed, pEmbed, aVar, pVar):
        
        sim_scores = self.similarity_scores(aEmbed, pEmbed, aVar, pVar)

        #loss is the mean negative similarity score for all genuine pairs
        loss = torch.mean(-sim_scores)
   
        return loss