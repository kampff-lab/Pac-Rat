# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:13:59 2020

@author: KAMPFF-LAB-ANALYSIS3
"""
#delta = 1-4 Hz
#theta = 4-8 Hz
#alpha = 8-12 Hz
#beta = 12-30 Hz   


#####george code 
    
import numpy as np
import scipy.stats as sstat
from scipy.ndimage import measurements


def _calculate_cluster_based_test_statistic(dataset_1, dataset_2, min_area=1, cluster_alpha=0.05,
                                           sample_statistic='independent', cluster_statistic='maxsum'):
    if sample_statistic == 'independent':
        t_test_result = sstat.ttest_ind(dataset_1, dataset_2, axis=-1, equal_var=False)
    if sample_statistic == 'dependent':
        t_test_result = sstat.ttest_rel(dataset_1, dataset_2, axis=-1)

    masked_pvalues = np.zeros(t_test_result.pvalue.shape)
    masked_pvalues[t_test_result.pvalue < cluster_alpha] = 1

    cluster_labels, num = measurements.label(masked_pvalues)
    cluster_areas, _ = measurements._stats(masked_pvalues, cluster_labels, index=np.arange(cluster_labels.max() + 1))
    clusters_over_min_area = np.squeeze(np.argwhere(cluster_areas > min_area))

    cluster_labels_over_min_area = np.ones(cluster_labels.shape) * -1
    cluster_statistic_result = []
    
    if clusters_over_min_area.shape == ():
        clusters_over_min_area = [clusters_over_min_area]
    for c_index in np.arange(len(clusters_over_min_area)):
        cluster = clusters_over_min_area[c_index]
        if cluster_statistic == 'maxsum':
            cluster_statistic_result.append(np.sum(t_test_result.pvalue[cluster_labels == cluster]))
        if cluster_statistic == 'maxarea':
            cluster_statistic_result.append(cluster_areas[cluster])
        cluster_labels_over_min_area[cluster_labels == cluster] = c_index
    return cluster_statistic_result, cluster_labels_over_min_area


def _monte_carlo_independent_samples(dataset_1, dataset_2, num_of_top_clusters=1, num_permutations=1000,
                                    cluster_alpha=0.05, sample_statistic='independent', cluster_statistic='maxsum'):

    full_dataset = np.concatenate((dataset_1, dataset_2), axis=-1)

    num_of_trials_1 = dataset_1.shape[-1]
    
    results = np.zeros((num_of_top_clusters, num_permutations))

    for p in np.arange(num_permutations):
        random_samples_indices = np.random.choice(np.arange(full_dataset.shape[-1]), num_of_trials_1, replace=False)
        data_1 = full_dataset[:, :, random_samples_indices]
        remaining_samples = np.delete(np.arange(full_dataset.shape[-1]), random_samples_indices)
        data_2 = full_dataset[:, :, remaining_samples]

        unordered_result, _ = _calculate_cluster_based_test_statistic(data_1, data_2, min_area=0,
                                                                      cluster_alpha=cluster_alpha,
                                                                      sample_statistic=sample_statistic,
                                                                      cluster_statistic=cluster_statistic)

        ordered_result = np.sort(unordered_result)[::-1]
        '''
        try:
            results[:, p] = ordered_result[:num_of_top_clusters]
        except ValueError:
            print('Try increasing the minimum cluster area because there are more clusters found in the data than in'
                  ' the random perturbations')
            break
            return None
        '''
        results_to_add = np.min([len(ordered_result), num_of_top_clusters])
        for r in np.arange(results_to_add):
            results[r, p] = ordered_result[r]
        if p % 10 == 0:
            print(p)

    return results


def monte_carlo_significance_probability(dataset_1, dataset_2, num_permutations=1000, min_area=1, cluster_alpha=0.05,
                                         monte_carlo_alpha=0.01, sample_statistic='independent',
                                         cluster_statistic='maxsum'):

    cluster_based_statistic, cluster_labels_over_min_area = \
        _calculate_cluster_based_test_statistic(dataset_1, dataset_2, min_area=min_area,
                                                cluster_alpha=cluster_alpha,
                                                sample_statistic=sample_statistic,
                                                cluster_statistic=cluster_statistic)

    num_of_top_clusters = len(cluster_based_statistic)
    monte_carlo_results = \
    _monte_carlo_independent_samples(dataset_1, dataset_2, num_of_top_clusters=num_of_top_clusters,
                                     num_permutations=num_permutations, cluster_alpha=cluster_alpha,
                                     sample_statistic=sample_statistic, cluster_statistic=cluster_statistic)

    if monte_carlo_results is None:
        return None

    cluster_labels_under_alpha = np.ones(cluster_labels_over_min_area.shape) * -1
    p_values = []
    k = 0
    for c in np.arange(num_of_top_clusters):
        random = monte_carlo_results[c]
        result = cluster_based_statistic[c]
        try:
            num_of_random_larger_than_result = len(np.squeeze(np.where(random > result)))
        except TypeError:
            num_of_random_larger_than_result = 0
        if num_of_random_larger_than_result / num_permutations < monte_carlo_alpha:
            p_values.append(num_of_random_larger_than_result / num_permutations)
            cluster_labels_under_alpha[cluster_labels_over_min_area == c] = k
        k += 1

    return p_values, cluster_labels_under_alpha


#plotting fx (not used)
    
def show_significant_clusters_on_data(data, cluster_labels, lfp_probe_positions, window_time=10,
                                      colormap=None, markers=None, alpha=None, marker_color=None):
    if colormap is None:
        colormap = 'viridis'
    if markers is None:
        markers = '+'
    if alpha is None:
        alpha = 1
    interpolation = 'bilinear'
    f = plt.figure(1)
    ax = f.add_subplot(111)
    im=ax.imshow(avg, aspect='auto',
              
               interpolation=interpolation,
              cmap=colormap, alpha=alpha,origin='upper')
    cbar =f.colorbar(im, ax=ax,boundaries=np.linspace(0,2))

    #ax.hlines(y=brain_regions, xmin=-window_time, xmax=window_time)
    t = cluster_labels
    clusters = np.delete(np.unique(t), np.argwhere(np.unique(t) == -1))
    for c in clusters:
        points = np.argwhere(t == c).astype(np.float)
        points[:, 1] = (points[:, 1]) / (data.shape[1]) * 16 - 8
        points[:, 0] = np.abs(points[:, 0] - data.shape[0] + 0.5)
        points[:, 0] = points[:, 0] / data.shape[0] * \
                       (12) + (-6)
        if marker_color is not None:
            ax.scatter(points[:, 1], points[:, 0], marker=markers, color=marker_color)
        else:
            ax.scatter(points[:, 1], points[:, 0], marker=markers)



