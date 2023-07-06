import numpy as np
import sklearn.metrics as metrics
from collections import Counter


def get_sparsification_curve(uncertainty, label, high_unc = True):
	###########################################################
	#Inputs:
	####### uncertainty: is a list of float uncertainty values. if high_unc = False, lower values = more uncertain. else if high_unc = True, higher values = more uncertain
	####### label: is a list of booleans that distinguishes between in-distribution/out-of-distribution, rather, what we expect to be certain and what we expect to be uncertain
	#######         this could be based on correct/incorrect or known/unknown. True = in distribution/known/correct, False = out distribution/unknown/incorrect
	####### high_unc: boolean indicating how to understand uncertainty values. if high_unc = True, higher values = more uncertain. 
	###########################################################
	uncertainty_np = np.array(uncertainty)
	
	if not high_unc:
		uncertainty_np = -uncertainty_np

	sorted_uncertainty = np.sort(uncertainty_np) #order from lowest to highest
	sorted_uncertainty = sorted_uncertainty[::-1] 
	
	total = len(uncertainty_np)

	#go through each score as a threshold
	rejectedCount = []
	all_error = []

	for score_threshold in [np.percentile(sorted_uncertainty, i) for i in range(0, 101, 1)][::-1]:
		#everything above the threshold is rejected
		rejects = uncertainty_np > score_threshold
		rejectedCount += [np.sum(rejects)/total]

		#error is number of FPs that remain and weren't rejected
		label_np = np.array(label)
		not_rejected = label_np[~rejects]
		error = np.sum(not_rejected != 1)/len(not_rejected)
		all_error += [error]

	all_error = np.array(all_error)
	rejectedCount = np.array(rejectedCount)

	return rejectedCount, all_error


def get_roc(uncertainty, label):
	in_unc = uncertainty[label == 1] 
	out_unc = uncertainty[label != 1]
	# in_unc = uncertainty[label] #uncertainty values for all in labels
	# out_unc = uncertainty[~label] #uncertainty values for all out labels

	# THRESHOLD FOR TPR AND FPR AND PRINT AUROC 
	sorted_uncertainty = np.sort(uncertainty) #order from lowest to highest
	sorted_uncertainty = sorted_uncertainty[::-1]
	fpr = []
	tpr = []

	#go through each score as a threshold
	for score_threshold in sorted_uncertainty:
		#in labels that correctly have a low uncertainty
		tp = np.sum(in_unc <= score_threshold) 
		#out labels that incorrectly have a low uncertainty
		fp = np.sum(out_unc <= score_threshold)

		tpr += [tp/len(in_unc)] #tpr is number of tp / number of pos
		fpr += [fp/len(out_unc)] #fpr is number of fp / number of neg

	tpr = np.array(tpr)
	fpr = np.array(fpr)

	return fpr, tpr


def get_min_ue(uncertainty, label, balance = False):
	in_unc = uncertainty[label == 1] 
	out_unc = uncertainty[label != 1]
	# in_unc = uncertainty[label] #uncertainty values for all in labels
	# out_unc = uncertainty[~label] #uncertainty values for all out labels


	thresholds = np.sort(uncertainty)
	uncertainty_error = []

	#assuming that higher is more uncertain
	for t in thresholds:
		#number of in scores that have high uncertainty
		in_error = np.sum(in_unc > t)/len(in_unc) ### essentially the false negative rate
		#number of out scores that have low uncertainty
		out_error = np.sum(out_unc <= t)/len(out_unc) #essentially false positive rate
		
		if balance:
			ue = 0.5*in_error + 0.5*out_error
		else:
			in_prop = len(in_unc)/(len(in_unc)+len(out_unc))
			out_prop = len(out_unc)/(len(in_unc)+len(out_unc))

			ue = in_prop*in_error + out_prop*out_error

		uncertainty_error += [ue]
	
	#return minimum possible uncertainty error
	return np.min(uncertainty_error)


def get_aupr(uncertainty, label):
	in_unc = uncertainty[label == 1] 
	out_unc = uncertainty[label != 1]
	# in_unc = uncertainty[label] #uncertainty values for all in labels
	# out_unc = uncertainty[~label] #uncertainty values for all out labels

	thresholds = np.sort(uncertainty)

	##############################################
	#precision recall where positive class is the in-data
	###############################################
	precisionIn = []
	recallIn = []
	#assuming that higher is more uncertain 
	for score_threshold in thresholds:
		#in labels that correctly have a low uncertainty
		tp = np.sum(in_unc <= score_threshold) 
		#out labels that incorrectly have a low uncertainty
		fp = np.sum(out_unc <= score_threshold)
		prec = tp / ( (tp+fp)+ 1e-6 )

		rec = tp/len(in_unc) #essentially TPR

		precisionIn += [prec]
		recallIn += [rec]

	auprIn = metrics.auc(recallIn, precisionIn)

	##############################################
	#precision recall where positive class is the out-data
	###############################################
	precisionOut = []
	recallOut = []
	for score_threshold in thresholds[::-1]:
		#out labels that correctly have a high uncertainty
		tp = np.sum(out_unc >= score_threshold)
		#in labels that incorrectly have a high uncertainty
		fp = np.sum(in_unc >= score_threshold)
		prec = tp / ( (tp+fp) + 1e-6 )

		rec = tp/len(out_unc)
	   
		precisionOut += [prec]
		recallOut += [rec]

	auprOut = metrics.auc(recallOut, precisionOut)

	return auprIn, auprOut


def get_uncertainty_results(uncertainty, label, high_unc = True):
	###########################################################
	#Inputs:
	####### uncertainty: is a list of float uncertainty values. if high_unc = False, lower values = more uncertain. else if high_unc = True, higher values = more uncertain
	####### label: is a list of booleans that distinguishes between in-distribution/out-of-distribution, rather, what we expect to be certain and what we expect to be uncertain
	#######         this could be based on correct/incorrect or known/unknown. True = in distribution/known/correct, False = out distribution/unknown/incorrect
	####### high_unc: boolean indicating how to understand uncertainty values. if high_unc = True, higher values = more uncertain. 
	###########################################################
	
	uncertainty = np.array(uncertainty)
	label = np.array(label)

	results = {}

	#change the magnitude if not low_unc. allows us flexible code below
	if not high_unc:
		uncertainty = -uncertainty
	
	fpr, tpr = get_roc(uncertainty, label)

	auroc = metrics.auc(fpr,tpr)

	min_ue_balanced = get_min_ue(uncertainty, label, True)
	min_ue = get_min_ue(uncertainty, label)

	auprIn, auprOut = get_aupr(uncertainty, label)


	rejRate, errorRate = get_sparsification_curve(uncertainty, label)
	auSC = metrics.auc(rejRate, errorRate)

	results['auroc'] = auroc
	results['min_ue_balanced'] = min_ue_balanced
	results['min_ue'] = min_ue
	results['auprIn'] = auprIn
	results['auprOut'] = auprOut
	results['sparsCurve'] = (rejRate, errorRate)
	results['auSC'] = auSC

	return results

