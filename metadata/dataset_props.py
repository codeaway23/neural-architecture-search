"""
This module contains functions to extract different dataset properties
"""
import datetime
import inspect
import json
import os
import random
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats


def calculate_distribution(data_frame1, data_frame2):
    """
    Calculate number of numerical and categorical columns in the data_frame
    :param data_frame1:
    :param data_frame2:
    :return:
    """
    print("Calculating distribution...")
    print("Start time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
    dict_percentage = dict()
    dict_percentage["numerical_percentage"] = None
    dict_percentage["categorical_percentage"] = None

    try:
        dict_percentage = dict()
        dict_percentage["numerical_percentage"] = ((data_frame1.shape[1]) / (data_frame2.shape[1])) * 100
        dict_percentage["categorical_percentage"] = 100 - dict_percentage["numerical_percentage"]
    except Exception as e:
        print("Exception in {}:{}".format(inspect.stack()[0][3], e))

    print("End time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
    print("")
    return dict_percentage


def calculate_kurtosis(data_frame):

    print("Calculating kurtosis...")
    print("Start time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))

    dict_kurt = dict()

    dict_kurt["kurtosis_max"] = None
    dict_kurt["kurtosis_min"] = None
    dict_kurt["kurtosis_average"] = None
    # dict_kurt["kurtosis_values"] = None
    dict_kurt["kurtosis_delta"] = None
    dict_kurt["kurtosis_percentage"] = None

    try:
        lis = list(data_frame.kurtosis())

        try:
            dict_kurt["kurtosis_max"] = max(lis)
        except Exception as e:
            print("Exception in {}:max:{}".format(inspect.stack()[0][3], e))

        try:
            dict_kurt["kurtosis_min"] = min(lis)
        except Exception as e:
            print("Exception in {}:min:{}".format(inspect.stack()[0][3], e))

        try:
            dict_kurt["kurtosis_average"] = np.mean(lis)
        except Exception as e:
            print("Exception in {}:average:{}".format(inspect.stack()[0][3], e))

        # try:
        #     dict_kurt["kurtosis_values"] = lis
        # except Exception as e:
        #     print("Exception in {}:values:{}".format(inspect.stack()[0][3], e))

        try:
            dict_kurt["kurtosis_delta"] = max(lis) - min(lis)
        except Exception as e:
            print("Exception in {}:delta:{}".format(inspect.stack()[0][3], e))

        try:
            dict_kurt["kurtosis_percentage"] = dict_kurt["kurtosis_average"] * 100
        except Exception as e:
            print("Exception in {}:percentage:{}".format(inspect.stack()[0][3], e))

    except Exception as e:
        print("Exception in {}:{}".format(inspect.stack()[0][3], e))

    print("End time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
    print("")
    return dict_kurt


def calculate_skew(data_frame):
    """
    It returns unbiased skew over requested axis Normalized by N-1
    :param data_frame: Pandas Dataframe
    :return: Dict of values
    """
    print("Calculating skew...")
    print("Start time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))

    dict_skew = dict()
    dict_skew["skew_max"] = None
    dict_skew["skew_min"] = None
    dict_skew["skew_average"] = None
    # dict_skew["skew_values"] = None
    dict_skew["skew_delta"] = None
    dict_skew["skew_percentage"] = None

    try:
        lis = list(data_frame.skew())

        try:
            dict_skew["skew_max"] = max(lis)
        except Exception as e:
            print("Exception in {}:max:{}".format(inspect.stack()[0][3], e))

        try:
            dict_skew["skew_min"] = min(lis)
        except Exception as e:
            print("Exception in {}:min:{}".format(inspect.stack()[0][3], e))

        try:
            dict_skew["skew_average"] = np.mean(lis)
        except Exception as e:
            print("Exception in {}:average:{}".format(inspect.stack()[0][3], e))

        # try:
        #     dict_skew["skew_values"] = lis
        # except Exception as e:
        #     print("Exception in {}:values:{}".format(inspect.stack()[0][3], e))

        try:
            dict_skew["skew_delta"] = max(lis) - min(lis)
        except Exception as e:
            print("Exception in {}:delta:{}".format(inspect.stack()[0][3], e))

        try:
            dict_skew["skew_percentage"] = dict_skew["skew_average"] * 100
        except Exception as e:
            print("Exception in {}:percentage:{}".format(inspect.stack()[0][3], e))

    except Exception as e:
        print("Exception in {}:{}".format(inspect.stack()[0][3], e))

    print("End time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
    print("")
    return dict_skew


def calculate_annova(data_frame):
    """
    Apply ANNOVA and calculate statistic and pvalue
    :param data_frame:
    :return:
    """
    print("Calculating annova...")
    print("Start time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))

    dict_annova = dict()
    dict_annova["annova_statistic"] = None
    dict_annova["annova_pvalue"] = None

    try:
        val = data_frame.values
        tmp_lis = stats.f_oneway(*val)
        dict_annova["annova_statistic"] = tmp_lis[0]
        dict_annova["annova_pvalue"] = tmp_lis[1]
    except Exception as e:
        print("Exception in {}:{}".format(inspect.stack()[0][3], e))

    print("End time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
    print("")
    return dict_annova


def calculate_correlation(data_frame):
    """
    Calculate correlation between columns
    :param data_frame: Pandas Dataframe
    :return: Dictionary of values
    """
    print("Calculating correlation...")
    print("Start time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))

    dict_corr = dict()
    dict_corr["corr_pos_average"] = None
    dict_corr["corr_neg_average"] = None
    dict_corr["corr_total_average"] = None

    try:
        correlations = data_frame.corr()

        flattend_data = correlations.values.flatten()
        all_values = flattend_data[~np.isnan(flattend_data)]

        pos_values = all_values[(all_values >= 0) & (all_values != 1.0) & (all_values != 1)]
        neg_values = all_values[all_values < 0]

        pos_mean = np.mean(pos_values)
        neg_mean = abs(np.mean(neg_values))

        if pos_mean != np.nan:
            dict_corr["corr_pos_average"] = pos_mean
        else:
            dict_corr["corr_pos_average"] = 0

        if neg_mean != np.nan:
            dict_corr["corr_neg_average"] = neg_mean
        else:
            dict_corr["corr_neg_average"] = 0

        if pos_mean + neg_mean != 0:
            dict_corr["corr_total_average"] = (pos_mean+neg_mean)/2
        else:
            dict_corr["corr_total_average"] = 0

    except Exception as e:
        print("Exception in {}:{}".format(inspect.stack()[0][3], e))

    print("End time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
    print("")
    return dict_corr


def calculate_levene(data_frame):
    """
    Apply levene test and calculate statistic and pvalue
    :param data_frame:
    :return:
    """
    print("Calculating levene...")
    print("Start time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))

    dict_levene = dict()
    dict_levene["levene_statistic"] = None
    dict_levene["levene_pvalue"] = None

    try:
        col = data_frame.values
        temp_lis = stats.levene(*col)

        dict_levene["levene_statistic"] = temp_lis[0]
        dict_levene["levene_pvalue"] = temp_lis[1]
    except Exception as e:
        print("Exception in {}:{}".format(inspect.stack()[0][3], e))

    print("End time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
    print("")
    return dict_levene


def calculate_anderson_darling(data_frame):
    """
    Anderson darling test
    :param data_frame:
    :return:
    """
    print("Calculating anderson darling...")
    print("Start time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))

    dict_anderson = dict()
    dict_anderson["norm"] = None
    dict_anderson['expon'] = None
    dict_anderson['logistic'] = None
    dict_anderson['gumbel'] = None
    dict_anderson['gumbel_l'] = None
    dict_anderson['gumbel_r'] = None

    try:

        type = ["norm", "expon", "logistic", "gumbel", "gumbel_l", "gumbel_r"]
        col = data_frame.columns

        size = data_frame.shape[0]
        thirty = size // 3
        to = 0
        for i in type:
            dict_anderson[i] = 0
        total = 0
        for i in range(0, size, thirty):
            to = 0

            start = i
            end = start + thirty
            prv = start
            # print(start,end)
            for j in range(start, end, 500):
                prv = j + 500
                to = to + 1
                if to > 33:
                    break
                val = random.randint(j, min(prv, end))
                if val > end:
                    val = start
                if val + 20 >= size:
                    break

                # print(val)
                for k in type:
                    count = 0
                    for x in col:
                        tmp = stats.anderson(data_frame[x].iloc[val:(val + 20)], k)
                        if tmp[0] < tmp[1][2]:
                            count = count + 1

                    dict_anderson[k] = dict_anderson[k] + (count / len(data_frame.columns))
            total = total + to

        # print(total)
        for i in type:
            dict_anderson[i] = dict_anderson[i] / total
            # print(dict_anderson[i])

    except Exception as e:
        print("Exception in {}:{}".format(inspect.stack()[0][3], e))

    print("End time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
    print("")
    return dict_anderson


def calculate_entropy(data_frame):
    """
    Calculate entropy for each column
    :param data_frame:
    :return:
    """
    print("Calculating entropy...")
    print("Start time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))

    data_frame = data_frame.select_dtypes(include=[np.number])
    data_frame.fillna(method="ffill", inplace=True)
    data_frame.fillna(method="bfill", inplace=True)

    dict_entropy = dict()
    dict_entropy["entropy_max"] = None
    dict_entropy["entropy_min"] = None
    dict_entropy["entropy_delta"] = None
    dict_entropy["entropy_average"] = None

    try:
       col = data_frame.columns
       list_entropy = []
       for i in col:

           if len(np.unique(data_frame[i])) < data_frame.shape[0] * 0.8:

               list_prob = np.array(data_frame[i].value_counts()) / data_frame.shape[0]
           else:
               tmp = pd.cut(data_frame[i], bins=5, labels=False)

               list_prob = np.array(tmp.value_counts()) / len(tmp)
           x = stats.entropy(list_prob)
           list_entropy.append(x)

       dict_entropy["entropy_max"] = max(list_entropy)
       dict_entropy["entropy_min"] = min(list_entropy)
       dict_entropy["entropy_delta"] = max(list_entropy) - min(list_entropy)
       dict_entropy["entropy_average"] = np.mean(list_entropy)
    except Exception as e:
       print("Exception in {}:{}".format(inspect.stack()[0][3], e))

    print("End time:", datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
    print("")

    return dict_entropy


def get_dataset_props(data_frame):
    """
    Extract all dataset properties
    :param data_frame:
    :return:
    """

    results = OrderedDict()

    try:
        """
        Check for numeric columns
        """
        columns = data_frame.dtypes.values

        print(data_frame.head())

        # Selecting subset of data_frame which have only numeric columns
        df = data_frame.select_dtypes(include=[np.number])

        # Applying condition when dataframe does not have any numeric columns
        if df.shape[1] * df.shape[0] == 0:
            return None

        # Filling null values with the method of forward filling and backward filling
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)

        # 1. Data distribution
        dict_percentage = calculate_distribution(df, data_frame)

        # 2. kurtosis
        dict_kurt = calculate_kurtosis(df)

        # 3. Skew
        dict_skew = calculate_skew(df)

        # 4. Correlation
        dict_corr = calculate_correlation(df)

        # 5. Anderson darling
        dict_anderson = calculate_anderson_darling(df)

        # 6. Levene
        dict_levene = calculate_levene(df)

        # 7. Entropy
        dict_entropy = calculate_entropy(df)

        # 8. ANNOVA
        dict_annova = calculate_annova(df)

        # Merge all dicts into one dictionary
        results.update(dict_percentage)
        results.update(dict_kurt)
        results.update(dict_skew)
        results.update(dict_corr)
        results.update(dict_anderson)
        results.update(dict_levene)
        results.update(dict_entropy)
        results.update(dict_annova)

        return results

    except Exception as e:
        print("Error:", e)
        return results


if __name__ == '__main__':
    
    # All datasets
    FILE_PATH = "../datasets"
    datasets = [FILE_PATH + "/" + x for x in os.listdir(FILE_PATH)]

    all_props = []

    for dataset in datasets:

        data_frame = pd.read_csv(dataset)
        dataset_props = get_dataset_props(data_frame=data_frame)

        print("Dataset Properties:", len(dataset_props.keys()), json.dumps(dataset_props))

        all_props.append(dataset_props)

    df = pd.DataFrame(all_props)
    df['dataset'] = os.listdir(FILE_PATH)
    df.to_csv("metadatafiles/dataset_props.csv")
