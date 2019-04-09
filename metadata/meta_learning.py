import os
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

import metadata.dataset_props as dprop

def get_trained_dataset_properties():
    return pd.read_csv('metadata/metadatafiles/dataset_props.csv')

def wasserstein_metric(vec1, vec2):
    return wasserstein_distance(vec1, vec2)

def iter_wasserstein(calc_df, ind_of_vec):
    calc_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    calc_df.fillna(0.0, inplace=True)
    dist_metric = []
    for x in calc_df.values:
        dist_metric.append(wasserstein_distance(x,calc_df.values[ind_of_vec,:]))
    return dist_metric

def get_new_dataset_properties(x, name='new_dataset'):
    x_props = pd.DataFrame(dprop.get_dataset_props(x), index=[0])
    x_props['dataset'] = name
    return x_props

def get_mdfile(x,y):
    x = pd.DataFrame(x)
    x['label'] = y ## cant be one-hot
    x_props = get_new_dataset_properties(x)

    all_props = get_trained_dataset_properties()

    calc_df = all_props.append(x_props, ignore_index=True)
    calc_df = calc_df.drop('dataset', axis=1)
    dist_metric = iter_wasserstein(calc_df, -1)
    
    print("wasserstein distance:", dist_metric)
    ind = list(np.argsort(dist_metric))
    ind.pop(0)
    meta_extract_order = all_props.ix[ind, 'dataset'].values
    meta_extract_order = [x[:-4] for x in meta_extract_order if x.endswith('.csv')]

    print("meta_extract_order:", meta_extract_order)
    
    for y in os.listdir('metadata/metadatafiles'):
        if meta_extract_order[0] in y:
            return 'metadata/metadatafiles/{}'.format(y)

