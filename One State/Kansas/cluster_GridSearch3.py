from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, FeatureAgglomeration, KMeans, MiniBatchKMeans, MeanShift, SpectralClustering
from sklearn import metrics
import itertools
import numpy as np

def cluster_GridSearch(data,model_param, perf_metric ={'Silhouette':None}, true_labels = None, verbose=False):
    ''' 
    PARAMETERS:

    data          Data for analysis

    model_param:  Dictionary containing model name as keys and values are a dictionary of the actual model and parameters.
                  
                  Example:
                  
                  model_param = {'KMeans':{'model':KMeans(),
                                           'n_clusters':[3,5,11,15,21,27,35,43],
                                           'n_jobs':[-2]},
                                 'DBSCAN':{'model':DBSCAN(),
                                           'eps':[0.2,0.3,0.4,0.5], 
                                           'min_samples':[5,10,20,40,80],
                                           'n_jobs':[-2]},
                                 'BIRCH':{'model':Birch(),
                                          'threshold':[0.1,0.2,0.3,0.4,0.5]}}

    perf_metric:  Dictionary with the key representing the performance metric and the values the parameters for it not inlcuding the data and cluster labels

                  Example:

                  perf_metric = {'Silhouette':{'sample_size':100,
                                               'random_state':5}}

    true_labels:  Optional list of actual labels which can be used when finding the optimal model

    verbose:      Print results for each model and parameter combination along with the performance metric value

    RETURNS:

    optimal_model: Dictionary of the optimal model is returned which contains the model and the performance metric value
    '''
    

    models = model_param.keys()
    optimal_model = {}
    i = 0
    metric_name = perf_metric.keys()[0]

    for model in models:
        
        p = model_param[model].copy()
        p.pop('model',None)
        
        keys, values = zip(*p.items())
    
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            for k, v in params.items():
                model_train = model_param[model]['model']
                setattr(model_train, k, v)

            model_train.fit(data)

            if metric_name  == 'Silhouette':
                try:
                    if perf_metric[metric_name]!=None:
                        metric = metrics.silhouette_score(data, model_train.labels_, **perf_metric[metric_name])
                    else:
                        metric = metrics.silhouette_score(data, model_train.labels_)
                except:
                    metric = np.nan
                if i == 0:
                    optimal_model[metric_name] = metric
                    optimal_model['Model'] = model
                    optimal_model['Parameters'] = params
                    optimal_model['Trained_Model'] = model_train
                else:
                    if optimal_model[metric_name]<metric:
                        optimal_model[metric_name] = metric
                        optimal_model['Model'] = model
                        optimal_model['Parameters'] = params
                        optimal_model['Trained_Model'] = model_train
            i+=1
            #else:
                #TODO: add other performance metrics with ground truth labels
                #metrics.homogeneity_score(true_labels, model_train.labels_)
                #metrics.completeness_score(true_labels, model_train.labels_)
                #metrics.v_measure_score(true_labels, model_train.labels_)
                #metrics.adjusted_rand_score(true_labels, model_train.labels_)
                #metrics.adjusted_mutual_info_score(true_labels, model_train.labels_)
            if verbose:
                print ('Model: {}'.format(model))
                print ('Parameters: {}'.format(params))
                print ('Clusters: {}'.format(len(set(model_train.labels_))))
                print ('{}: {}'.format(metric_name,metric))
                print 

    print ('OPTIMAL MODEL')
    print ('Model: {}'.format(optimal_model['Model']))
    print ('Parameters: {}'.format(optimal_model['Parameters']))
    print ('Clusters: {}'.format(len(set(optimal_model['Trained_Model'].labels_))))
    print ('{}: {}'.format(metric_name,optimal_model[metric_name]))
    
    return optimal_model