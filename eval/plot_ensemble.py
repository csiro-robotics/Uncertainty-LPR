from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import ast
import scipy.stats as stats
noise = np.random.normal(0, 1, (1000, ))
density = stats.gaussian_kde(noise)


# Plot recall against increasing number of models in an ensemble
def ensemble_recall(cosinesim):
    all_recall = [] # 2d array of recall vals for all models
    num_ensembles = 0
    font_size = 10

    for model_key, model_value in cosinesim.items():
        print('Ensemble with {} models'.format(model_key))
        model_recall = [] # 1d array of recall vals for all datasets
        num_ensembles += 1

        for dataset_key, dataset_value in model_value.items():
            # get recall val
            recall = round(dataset_value['recall'], 2)
            model_recall.append(recall)
            # print recall
            if dataset_key == 'oxford': print('{}\t\tAvg. top 1% recall: {}'.format(dataset_key, recall))
            else: print('{}\tAvg. top 1% recall: {}'.format(dataset_key, recall))
            
        all_recall.append(model_recall)
        print()

    # x axis and y axis
    all_recall = np.array(all_recall)
    x_axis = list(range(1, num_ensembles+1))

    plt.rcParams["figure.figsize"] = (13,7)
    plt.rcParams['font.size'] = str(font_size)
    plt.plot(x_axis,all_recall[:,0],'r-',label='oxford')
    plt.plot(x_axis,all_recall[:,3],'k-',label='university')
    plt.plot(x_axis,all_recall[:,2],'g-',label='residential')
    plt.plot(x_axis,all_recall[:,1],'b-',label='business')
    
    plt.tick_params(axis='x', labelsize= font_size)
    plt.tick_params(axis='y', labelsize= font_size)
    plt.title('MinkLoc3D Average Recall of Ensembles with 1-{} Models'.format(num_ensembles), fontsize=12)
    plt.ylabel('Average Recall @ 1%', fontsize=font_size)
    plt.xlabel('Number of Models in Ensemble', fontsize=font_size)
    plt.xticks(x_axis)
    plt.legend()

    plt.savefig('pics/ensemble-recall-1-10.png')
    plt.show()
 


# Plot cosine similarity histograms parted by correct and incorrect predictions
def cosinesim_histograms(cosinesim):

    # COMPARE EACH DATASET
    for i in range(4):

        iterate = [2,4,6,10]
        font_size = 7
        title_font_size = 10
        fig_size = (15,5)

        dataset = list(cosinesim['1'].keys())[i]

        # FOR COUNT HISTOGRAMS

        fig, axs = plt.subplots(1, len(iterate), sharex=True, sharey=True, figsize=fig_size) #only sharex=True for variance
        plt.rcParams['font.size'] = str(font_size)
        j = 0
        
        for num_models in iterate: 
            axs[j].hist(cosinesim[str(num_models)][dataset]['correct'], alpha=0.5, color='b', bins='auto', label='Correct Predictions')
            axs[j].hist(cosinesim[str(num_models)][dataset]['incorrect'], alpha=0.5, color='r', bins='auto', label='Incorrect Predictions')
            _,x,_ = axs[j].hist(cosinesim[str(num_models)][dataset]['correct'], histtype=u'step', color='b', bins='auto')
            _,x2, _ = axs[j].hist(cosinesim[str(num_models)][dataset]['incorrect'], histtype=u'step', color='r', bins='auto')
            axs[j].plot(x,density(x))
            axs[j].plot(x2,density(x2))
            axs[j].set_title('{} Model Ensemble'.format(num_models), weight='bold', size=font_size)
            axs[j].set_xlim((0,4e-5))
            axs[j].legend(loc="upper right")
            axs[j].tick_params(axis='x', labelsize= font_size)
            axs[j].tick_params(axis='y', labelsize= font_size)
            j += 1

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        #plt.xlabel("\nEnsemble Variance of Cosine Similarity between Query and Top 1 Database Match", fontsize=font_size)
        plt.xlabel("\nCosine Similarity between Query and Top 1 Database Match", fontsize=font_size)
        plt.ylabel("Count\n", fontsize=font_size)
        #plt.suptitle('MinkLoc3D Ensemble Variance of Cosine Similarity between Point Cloud Queries and Top Matching Database Entries for Correct and Incorrect Predictions : ' + r'$\bf{' + str(dataset) + '}\ dataset$', fontsize=title_font_size)
        plt.suptitle('MinkLoc3D Cosine Similarity between Point Cloud Queries and Top Matching Database Entries for Correct and Incorrect Predictions : ' + r'$\bf{' + str(dataset) + '}\ dataset$', fontsize=title_font_size)
        
        #plt.savefig('pics/cosinesim_variance_count_{}.png'.format(dataset))
        plt.savefig('pics/cosinesim_count_{}.png'.format(dataset))
        plt.show()
        

        # FOR DENSITY HISTOGRAMS
        
        fig, axs = plt.subplots(1, len(iterate), sharex=True, figsize=fig_size) #sharey=True, for mean
        plt.rcParams['font.size'] = str(font_size)
        j = 0

        for num_models in iterate: #later 8,10 models 
            axs[j].hist(cosinesim[str(num_models)][dataset]['correct'], density=True, stacked=True, alpha=0.5, color='b', bins='auto', label='Correct Predictions')
            axs[j].hist(cosinesim[str(num_models)][dataset]['incorrect'], density=True, stacked=True, alpha=0.5, color='r', bins='auto', label='Incorrect Predictions')
            _,x,_ = axs[j].hist(cosinesim[str(num_models)][dataset]['correct'], density=True, stacked=True, histtype=u'step', color='b', bins='auto')
            _,x2, _ = axs[j].hist(cosinesim[str(num_models)][dataset]['incorrect'], density=True, stacked=True, histtype=u'step', color='r', bins='auto')
            axs[j].plot(x,density(x))
            axs[j].plot(x2,density(x2))
            axs[j].set_title('{} Model Ensemble'.format(num_models), weight='bold', size=font_size)
            axs[j].set_xlim((0,4e-5))
            axs[j].legend(loc="upper right")
            axs[j].tick_params(axis='x', labelsize= font_size)
            axs[j].tick_params(axis='y', labelsize= font_size)
            j += 1

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        #plt.xlabel("\nEnsemble Variance of Cosine Similarity between Query and Top 1 Database Match", fontsize=font_size)
        plt.xlabel("\nCosine Similarity between Query and Top 1 Database Match", fontsize=font_size)
        plt.ylabel("Density\n", fontsize=font_size)
        #plt.suptitle('MinkLoc3D Ensemble Variance Cosine Similarity between Point Cloud Queries and Top Matching Database Entries for Correct and Incorrect Predictions : ' + r'$\bf{' + str(dataset) + '}\ dataset$', fontsize=title_font_size)
        plt.suptitle('MinkLoc3D Cosine Similarity between Point Cloud Queries and Top Matching Database Entries for Correct and Incorrect Predictions : ' + r'$\bf{' + str(dataset) + '}\ dataset$', fontsize=title_font_size)
        
        #plt.savefig('pics/cosinesim_variance_density_{}.png'.format(dataset))
        plt.savefig('pics/cosinesim_density_{}.png'.format(dataset))
        plt.show()

    

# Plot cosine similarity ROC curves and print AUROC scores
def cosinesim_roc(cosinesim):

    all_auroc = []
    # COMPARE EACH DATASET
    for i in range(4):
        
        iterate = [2,3,5,10] 
        font_size = 7
        title_font_size = 10
        fig_size = (13,4)

        dataset = list(cosinesim['1'].keys())[i]
        dataset_auroc = [] # 1d array of recall vals for all datasets
        
        fig, axs = plt.subplots(1, len(iterate), sharex=True, sharey=True, figsize=fig_size)
        plt.rcParams['font.size'] = str(font_size)
        
        j = 0 
        
        for num_models in range(1,len(list(cosinesim.keys()))+1): 
            correct = np.array(cosinesim[str(num_models)][dataset]['correct'])
            incorrect = np.array(cosinesim[str(num_models)][dataset]['incorrect'])
            all_cosine = np.concatenate((correct, incorrect)) 
            sorted_cosine = np.sort(all_cosine) #order from lowest cosine to highest cosine score
            fpr = []
            tpr = []

            #go through each score as a threshold
            for score_threshold in sorted_cosine:
                tp = np.sum(correct>=score_threshold) # > for mean and < for variance
                fp = np.sum(incorrect>=score_threshold)
                tpr.append(tp/len(correct))
                fpr.append(fp/len(incorrect))

            tpr = np.array(tpr)
            fpr = np.array(fpr)

            auroc = round(metrics.auc(fpr,tpr),4)
            dataset_auroc.append(auroc)
            
            if num_models in iterate:
                # plot roc curve
                axs[j].plot(fpr,tpr, label='AUROC = ' + r'$\bf{' + str(auroc) + '}$')
                
                # plot linear line to compare perfect fit
                x = [0,1]
                y = [0,1]
                axs[j].plot(x,y,'--k', alpha=0.5, label='random classifier')

                axs[j].set_title('{} Model Ensemble'.format(num_models), weight='bold', size=font_size)
                axs[j].legend(loc="lower right")
                axs[j].tick_params(axis='x', labelsize= font_size)
                axs[j].tick_params(axis='y', labelsize= font_size)
                j += 1

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("False Positive Rate", fontsize=font_size)
        plt.ylabel("True Positive Rate", fontsize=font_size)
        #plt.suptitle('MinkLoc3D ROC Curves for Ensemble Variance of Cosine Similarity of Correct and Incorrect Predictions : ' + r'$\bf{' + str(dataset) + '}\ dataset$', fontsize=title_font_size)
        plt.suptitle('ROC Curves for Cosine Similarity Separation Between Correct and Incorrect Predictions using MinkLoc3D : ' + r'$\bf{' + str(dataset) + '}\ dataset$', fontsize=title_font_size)
        

        print("AUROC Scores for {} dataset".format(dataset))
        for k in range(len(dataset_auroc)):
            print('{} Model Ensemble\t {}'.format(k+1,dataset_auroc[k]))
        print()
        all_auroc.append(dataset_auroc)

        #plt.savefig('pics/roc_variance_{}.png'.format(dataset))
        plt.savefig('pics/roc_{}.png'.format(dataset))
        plt.show()
        
    # Plot auroc performance
    all_auroc = np.array(all_auroc)
    num_ensembles = np.shape(all_auroc)[1]
    x_axis = list(range(1, num_ensembles+1)) # range(1,) for mean, range(2,) for variance

    font_size = 10
    plt.rcParams['font.size'] = str(font_size)
    plt.rcParams["figure.figsize"] = (13,7)
    plt.plot(x_axis,all_auroc[0,:],'r-',label='oxford') # [0,:] for mean, [0,1:] for variance
    plt.plot(x_axis,all_auroc[3,:],'k-',label='university')
    plt.plot(x_axis,all_auroc[2,:],'g-',label='residential')
    plt.plot(x_axis,all_auroc[1,:],'b-',label='business')
    
     
    plt.tick_params(axis='x', labelsize= font_size)
    plt.tick_params(axis='y', labelsize= font_size)
    #plt.title('MinkLoc3D AUROC Scores for Ensemble Variance of Cosine Similarity of Correct and Incorrect Predictions'.format(num_ensembles), fontsize=12)
    plt.title('MinkLoc3D AUROC Scores between Cosine Similarity Ensemble Variance of Correct and Incorrect Predictions of Ensembles with 1-{} Models'.format(num_ensembles), fontsize=12)
    
    plt.ylabel('AUROC Score', fontsize=font_size)
    plt.xlabel('Number of Models in Ensemble', fontsize=font_size)
    plt.xticks(x_axis)
    plt.legend()
    #plt.savefig('pics/ensemble-variance-auroc-1-10.png')
    plt.savefig('pics/ensemble-auroc-1-10.png')
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise Ensemble Findings')
    parser.add_argument('--plot_ensemble_recall', type=bool, required=False, default=False, help='Plot recall as number of models in ensemble changes? True/False (default False)')
    parser.add_argument('--plot_cosinesim_histograms', type=bool, required=False, default=False, help='Plot cosine similarity separated by correct and incorrect point cloud predictions? True/False (default False)')
    parser.add_argument('--plot_cosinesim_roc', type=bool, required=False, default=False, help='Plot ROC curves and AUROC scores separated by correct and incorrect point cloud predictions? True/False (default False)')

    args = parser.parse_args()

    # read in .txt file containing cosine similiarity dict
    with open('cosinesim_dict.txt') as json_file:
        d = json.load(json_file)
        cosinesim = ast.literal_eval(d)

    if args.plot_ensemble_recall: ensemble_recall(cosinesim)
    if args.plot_cosinesim_histograms: cosinesim_histograms(cosinesim)
    if args.plot_cosinesim_roc: cosinesim_roc(cosinesim)
