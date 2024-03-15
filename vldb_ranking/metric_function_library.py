import math
from scipy.stats import percentileofscore
import numpy as np
import pandas as pd
from math import log2
"""
A library of functions desinged to work with the Rankees Ojects.

Note some of the functions may be from a previos version and need tweeking
"""

def disparity_single(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	# print(columns)
	if columns == None:
		return None
	#~ length = len(columns)
	keep = True
	return_stored = True
	selected,thresh = ranking_function.get_partial_ranking(rankees, k=k)
	if norm_funcs and rankees.norm_funcs != norm_funcs:
		# print('renormalizing')
		temp_norm = rankees.norm_funcs
		rankees.norm_funcs = norm_funcs
		keep=False
		return_stored= False
		# print(rankees.norm_funcs)
	# print(norm_funcs, rankees.norm_funcs)
	column = columns[0]
	selected_data = rankees.get_normalized(keep=keep, return_stored=return_stored).loc[selected][columns]
	all_data = rankees.get_normalized(keep=keep, return_stored=return_stored)[column]
	# print(selected_data['sex'])
	if not keep:
		rankees.norm_funcs = temp_norm
	total_centroid = all_data.value_counts(normalize=True)
	centroid = selected_data.value_counts(normalize=True)
	# print(centroid- total_centroid, total_centroid - centroid)
	# print(total_centroid, centroid)
	return (centroid + total_centroid).fillna(0) - 2 * total_centroid
def disparity_product(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	# print(columns)
	if columns == None:
		columns = rankees.get_ranking_columns()
	#~ length = len(columns)
	keep = True
	return_stored = True
	selected,thresh = ranking_function.get_partial_ranking(rankees, k=k)
	if norm_funcs and rankees.norm_funcs != norm_funcs:
		# print('renormalizing')
		temp_norm = rankees.norm_funcs
		rankees.norm_funcs = norm_funcs
		keep=False
		return_stored= False
		# print(rankees.norm_funcs)
	# print(norm_funcs, rankees.norm_funcs)
	selected_data = rankees.get_normalized(keep=keep, return_stored=return_stored).loc[selected , columns]
	all_data = rankees.get_normalized(keep=keep, return_stored=return_stored).loc[ : , columns]
	# print(selected_data['sex'])
	if not keep:
		rankees.norm_funcs = temp_norm
	def find_product(temp_columns, in_use, not_in_use):
		if not temp_columns:
			    #https://stackoverflow.com/questions/34157811/filter-a-pandas-dataframe-using-values-from-a-dict
				filter = {column:1 for column in in_use}
				filter.update({column:0 for column in not_in_use})
				sel = selected_data.loc[(selected_data[list(filter)] == pd.Series(filter)).all(axis=1)]
				all = all_data.loc[(all_data[list(filter)] == pd.Series(filter)).all(axis=1)]
				name = ''.join([('+' if item in in_use else '-') + item for item in columns])
				return [(name, len(sel) / len(selected_data) - len(all) / len(all_data))]
		else:
			col = temp_columns[0]
			rest = temp_columns[1:]
			return find_product(rest, in_use + (col, ), not_in_use) + find_product(rest, in_use, not_in_use + (col, ))
	# print(total_centroid, centroid)

	return pd.Series(dict(find_product(tuple(columns), (), ())))

def disparity_pairs(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	# print(columns)
	if columns == None:
		return None
	#~ length = len(columns)
	keep = True
	return_stored = True
	selected,thresh = ranking_function.get_partial_ranking(rankees, k=k)
	if norm_funcs and rankees.norm_funcs != norm_funcs:
		# print('renormalizing')
		temp_norm = rankees.norm_funcs
		rankees.norm_funcs = norm_funcs
		keep=False
		return_stored= False
		# print(rankees.norm_funcs)
	# print(norm_funcs, rankees.norm_funcs)
	column = columns[0]
	selected_data = rankees.get_normalized(keep=keep, return_stored=return_stored).loc[selected , list(set(columns))]
	all_data = rankees.get_normalized(keep=keep, return_stored=return_stored).loc[ : , list(set(columns))]
	temp_selected = selected_data.copy()
	temp_all = all_data.copy()
	for i in range(0, len(columns) - 1, 2):
		print (i, columns[i] + '_' + columns[i + 1])
		temp_selected[columns[i] + '_' + columns[i + 1]] = selected_data[columns[i]] * selected_data[columns[i+1]]
		temp_all[columns[i] +  '_' + columns[i + 1]] = all_data[columns[i]] * all_data[columns[i+1]]
	# print(temp_selected.columns)
	if not keep:
		rankees.norm_funcs = temp_norm
	total_centroid = temp_all.mean()
	centroid = temp_selected.mean()
	# print(centroid- total_centroid, total_centroid - centroid)
	# print(total_centroid, centroid)
	return (centroid + total_centroid).fillna(0) - 2 * total_centroid
def disparity_machine(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	# print(columns)
	if columns == None:
		columns = rankees.get_ranking_columns()
	#~ length = len(columns)
	keep = True
	return_stored = True
	selected,thresh = ranking_function.get_partial_ranking(rankees, k=k)
	if norm_funcs and rankees.norm_funcs != norm_funcs:
		# print('renormalizing')
		temp_norm = rankees.norm_funcs
		rankees.norm_funcs = norm_funcs
		keep=False
		return_stored= False
		# print(rankees.norm_funcs)
	# print(norm_funcs, rankees.norm_funcs)
	selected_data = rankees.get_normalized(keep=keep, return_stored=return_stored).loc[selected , columns]
	all_data = rankees.get_normalized(keep=keep, return_stored=return_stored).loc[ : , columns]
	# print(selected_data['sex'])
	if not keep:
		rankees.norm_funcs = temp_norm
	total_centroid = all_data.mean()
	centroid = selected_data.mean()
	# print(total_centroid, centroid)
	return centroid - total_centroid
def disparity(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	# print(columns)
	if columns == None:
		columns = rankees.get_ranking_columns()
	#~ length = len(columns)
	keep = True
	return_stored = True
	selected,thresh = ranking_function.get_partial_ranking(rankees, k=k)
	if norm_funcs and rankees.norm_funcs != norm_funcs:
		# print('renormalizing')
		temp_norm = rankees.norm_funcs
		rankees.norm_funcs = norm_funcs
		keep=False
		return_stored= False
		# print(rankees.norm_funcs)
	# print(norm_funcs, rankees.norm_funcs)
	selected_data = rankees.get_normalized(keep=keep, return_stored=return_stored).loc[selected , columns]
	all_data = rankees.get_normalized(keep=keep, return_stored=return_stored).loc[ : , columns]
	# print(selected_data['sex'])
	if not keep:
		rankees.norm_funcs = temp_norm
	total_centroid = all_data.mean()
	centroid = selected_data.mean()
	# print(total_centroid, centroid)
	return centroid - total_centroid
def kl_divergence(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	# print(columns)
	if columns == None:
		columns = rankees.get_ranking_columns()
	#~ length = len(columns)
	keep = True
	return_stored = True
	selected,thresh = ranking_function.get_partial_ranking(rankees, k=k)
	if norm_funcs and rankees.norm_funcs != norm_funcs:
		# print('renormalizing')
		temp_norm = rankees.norm_funcs
		rankees.norm_funcs = norm_funcs
		keep=False
		return_stored= False
		# print(rankees.norm_funcs)
	# print(norm_funcs, rankees.norm_funcs)
	selected_data = rankees.get_normalized(keep=keep, return_stored=return_stored).loc[selected , columns]
	all_data = rankees.get_normalized(keep=keep, return_stored=return_stored).loc[ : , columns]
	# print(selected_data['sex'])
	if not keep:
		rankees.norm_funcs = temp_norm
	s_plus_div_N = all_data.mean()
	s_plus_div_i = selected_data.mean()
	s_minus_div_N = 1 - s_plus_div_N
	s_minus_div_i = 1 - s_plus_div_i
	# print(total_centroid, centroid)
	kl_plus = s_plus_div_i * np.log2(s_plus_div_i / s_plus_div_N)
	kl_minus = s_minus_div_i * np.log2(s_minus_div_i / s_minus_div_N)
	print(kl_plus, kl_minus)
	print('****')
	print(disparity(rankees, ranking_function, columns, k, norm_funcs))
	no_nan = (kl_plus + kl_minus).fillna(1)
	print(no_nan)
	return no_nan
def false_positive(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	if columns == None:
		columns = rankees.get_ranking_columns()
	selected_nums = ranking_function.get_partial_ranking(rankees, k=k)[0]
	sn_set = set(selected_nums)
	unselected_nums = [num for num in rankees.clean_results.index.values if num not in sn_set]
	selected_data = rankees.get_normalized().loc[selected_nums , columns + ['two_year_recid']]
	unselected_data = rankees.get_normalized().loc[unselected_nums , columns + ['two_year_recid']]
	# exit(0)
	# total_data = rankees.get_normalized().loc[: , columns]
	# print(len(total_data))
	# print(len(selected_data))
	answer = {}
	# exit(0)

	overall_fp_rate = len(selected_data[selected_data['two_year_recid'] == 0])
	# print((1 - selected_data.two_year_recid.mean()) * selected_data.two_year_recid.count())
	# print()
	overall_fp_rate = overall_fp_rate / (overall_fp_rate + len(unselected_data[unselected_data['two_year_recid'] == 0]))
	# print(len(selected_data[selected_data['two_year_recid'] == 0]))
	# print(len(selected_data))
	# print(ranking_function.k)
	# print(overall_fp_rate)
	# exit(0)
	# exit(0)
	for column in columns:
		selected_1 = selected_data[selected_data[column] == 1]
		unselected_1 = unselected_data[unselected_data[column] == 1]
		column_fp_rate = len(selected_1[selected_1['two_year_recid'] == 0])
		if len(selected_1) == 0:
			answer[column] = 0
		elif column_fp_rate == 0 and len(unselected_1[unselected_1['two_year_recid'] == 0]) == 0:
			answer[column] = 0
		else:
			# print(column, column_fp_rate)
			answer[column] =  (column_fp_rate / ( len(unselected_1[unselected_1['two_year_recid'] == 0])+ column_fp_rate)) - overall_fp_rate
	# print(pd.Series(answer))
	# print(disparity(rankees, ranking_function, columns, k, norm_funcs))
	# print(answer)
	# print(overall_fp_rate)
	# print({key:answer[key] + overall_fp_rate for key in answer})
	return pd.Series(answer)

def disparate_impact(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	if columns == None:
		columns = rankees.get_ranking_columns()
	selected_nums = ranking_function.get_partial_ranking(rankees, k=k)[0]
	# unselected_nums = [num for num in rankees.clean_results.index.values if num not in selected_nums]
	selected_data = rankees.get_normalized().loc[selected_nums , columns]
	total_data = rankees.get_normalized().loc[: , columns]
	answer = {}
	for column in columns:
		selected_0 = selected_data[selected_data[column] == 0]
		selected_1 = selected_data[selected_data[column] == 1]
		total_0 = total_data[total_data[column] == 0]
		total_1 = total_data[total_data[column] == 1]
		if len(total_1) == 0 or len(total_0) == 0:
			answer[column] = 0
			continue
		if len(selected_1) != 0:
			p_0 = (len(selected_0) / len(total_0)) / (len(selected_1) / len(total_1))
		else:
			p_0 = 1
		if len(selected_0) != 0:
			p_1 = (len(selected_1) / len(total_1)) / (len(selected_0) / len(total_0))
		else:
			p_1 = 1
		if p_1 < p_0:
			answer[column] = p_1 - 1
		else:
			answer[column] = 1 - p_0
	# print(pd.Series(answer))
	# print(disparity(rankees, ranking_function, columns, k, norm_funcs))
	return pd.Series(answer)

def disparity_norm(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	return pd.Series(np.linalg.norm(disparity(rankees, ranking_function, columns, k, norm_funcs)))

# 	note this function requires columns and only makes sense in the 0-1 case
# the less selected one is considered the protected class

def impact_ratio(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	if columns == None:
		columns = rankees.get_ranking_columns()
	selected_nums = set(ranking_function.get_partial_ranking(rankees, k=k)[0])
	# unselected_nums = [num for num in rankees.clean_results.index.values if num not in selected_nums]
	selected_data = rankees.get_normalized().loc[selected_nums , columns]
	total_data = rankees.get_normalized().loc[: , columns]
	answer = {}
	for column in columns:
		selected_0 = selected_data[selected_data[column] == 0]
		selected_1 = selected_data[selected_data[column] == 1]
		total_0 = total_data[total_data[column] == 0]
		total_1 = total_data[total_data[column] == 1]
		if len(selected_1) != 0:
			p_0 = (len(selected_0) / len(total_0)) / (len(selected_1) / len(total_1))
		else:
			p_0 = 0
		if len(selected_0) != 0:
			p_1 = (len(selected_1) / len(total_1)) / (len(selected_0) / len(total_0))
		else:
			p_1 = 0
		answer[column] = min(p_0, p_1)
	return pd.Series(answer)
def max_kl(size_0, size_1, Q, N):
	i = 10
	if size_0 > size_1:
		zero_big = True
		temp = size_0
	else:
		zero_big = False
		temp = size_1
	mKL = 0
	while i < N:
		# print(prefix)
		if zero_big:
			num_0 = min(i, temp)
			num_1 = i - num_0
		else:
			num_1 = min(i, temp)
			num_0 = i - num_1
		P = (num_1/ i, num_0 / i)
		divergence = 0
		for j in range(len(P)):
			if P[j] > 0:
				divergence += P[j] * log2(P[j] / Q[j])
		mKL += divergence / log2(i)
		i += 10
	return mKL
def kl_divergence_old(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	if columns == None:
		columns = rankees.get_ranking_columns()[0]
	answer = {}
	sorted_nums = ranking_function.get_full_ranking(rankees)

	data = rankees.get_normalized()
	sorted_data = data.loc[sorted_nums , columns]
	N = len(sorted_nums)
	print('N: ', N)
	for column in columns:
		size_1 = len(data[data[column] == 1])
		size_0 = len(data[data[column] == 0])
		if size_0 == 0 or size_1 == 0:
			answer[column] = 0
			continue
		Q = (size_1 / N, size_0 / N)
		i = 10
		rKL = 0
		while i < N:
			prefix = data.loc[sorted_nums[:i], columns]
			# print(prefix)
			P = (len(prefix[prefix[column] == 1]) / i , len(prefix[prefix[column] == 0]) / i)
			divergence = 0
			for j in range(len(P)):
				if P[j] > 0:
					divergence += P[j] * log2(P[j] / Q[j])
			rKL += divergence / log2(i)
			i += 10
		print(column, rKL, size_0, size_1, Q, N)
		answer[column] = rKL / max_kl(size_0, size_1, Q, N)
	return pd.Series(answer)

def mean_difference(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	if columns == None:
		columns = rankees.get_ranking_columns()
	selected_nums = set(ranking_function.get_partial_ranking(rankees, k=k)[0])
	# unselected_nums = [num for num in rankees.clean_results.index.values if num not in selected_nums]
	selected_data = rankees.get_normalized().loc[selected_nums , columns]
	total_data = rankees.get_normalized().loc[: , columns]
	answer = {}
	for column in columns:
		selected_0 = selected_data[selected_data[column] == 0]
		selected_1 = selected_data[selected_data[column] == 1]
		total_0 = total_data[total_data[column] == 0]
		total_1 = total_data[total_data[column] == 1]
		p_0 = -((len(selected_0) / len(total_0)) - (len(selected_1) / len(total_1)))
		answer[column] = p_0
	return pd.Series(answer)
def simple_difference(rankees, ranking_function, columns, k=None, norm_funcs={}):
	selected, thresh = ranking_function.get_partial_ranking(rankees, k=k)
	selected_data = rankees.get_normalized().loc[selected , columns]
	all_data = rankees.get_normalized().loc[ : , columns]
	answer = {}
	for column in columns:
		all_1 = all_data[column] == 1
		selected_1 = selected_data[column] == 1
		p_select_1 = len(selected_data[selected_1]) / len(all_data[all_1])
		all_0 = all_data[column] == 0
		selected_0 = selected_data[column] == 0
		p_select_0 = len(selected_data[selected_0]) / len(all_data[all_0])
		answer[column] = min(p_select_0, p_select_1) / max(p_select_0, p_select_1)
	return pd.Series(answer)


def participation(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	if columns == None:
		columns = rankees.get_ranking_columns()
	#~ length = len(columns)
	selected, thresh = ranking_function.get_partial_ranking(rankees, k=k)
	selected_data = rankees.get_normalized().loc[selected , columns]
	#converts to binary above vs bellow threshhold and then applies the weights using pointwise multiplication
	binary_data = selected_data.gt(thresh).astype('float64')
	#used https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value
	#divides each row by the sum of itself
	pointwise_part = binary_data.div(binary_data.sum(axis=1), axis=0)
	#returns the mean participation of each column
	return pointwise_part.mean(axis=0)

def weighted_participation(rankees, ranking_function, columns=None, k=None,norm_funcs={}):
	if columns == None:
		columns = rankees.get_ranking_columns()
	#~ length = len(columns)
	selected, thresh = ranking_function.get_partial_ranking(rankees, k=k)
	weights = [ranking_function.weights[item] for item in columns]
	selected_data = rankees.get_normalized().loc[selected , columns]
	#converts to binary above vs bellow threshhold and then applies the weights using pointwise multiplication
	#note this last application of the weights is the only difference between paricipation and weighted participation
	weighted_binary_data = selected_data.gt(thresh).astype('float64').mul(weights)
	#used https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value
	#divides each row by the sum of itself
	pointwise_weighted_part = weighted_binary_data.div(weighted_binary_data.sum(axis=1), axis=0)
	#returns the mean participation of each column
	return pointwise_weighted_part.mean(axis=0)
def threshold(rankees, ranking_function, columns=None, k=None,norm_funcs={}):
	_, thresh = ranking_function.get_partial_ranking(rankees, k=k)
	return thresh
def disqualifying_power(rankees, ranking_function, columns=None, k=None,norm_funcs={}):
	#assumes weights sum to 1 and scores can't be higher than 1
	p_floor = param_floor(rankees, ranking_function, columns)
	data = rankees.get_normalized()
	disqualifying_power = {}
	for key in p_floor:
		disqualifying_power[key] = percentileofscore(data.loc[ : , key], p_floor[key])/100
	return pd.Series(disqualifying_power)
def param_floor(rankees, ranking_function, columns=None, k=None,norm_funcs={}):
	#assumes weights sum to 1 and scores can't be higher than 1
	_, thresh = ranking_function.get_partial_ranking(rankees, k=k)
	weights = ranking_function.weights
	p_floor = {}
	for key in weights:
		worst_score = thresh - (1 - weights[key])
		if worst_score < 0:
			worst_score = 0
		p_floor[key] = worst_score / weights[key]
	print(thresh)
	return pd.Series(p_floor)
def importance(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	if columns == None:
		columns = rankees.get_ranking_columns()
	#~ length = len(columns)
	selected,thresh = ranking_function.get_partial_ranking(rankees, k=k)
	selected_data = rankees.get_normalized().loc[selected , columns]
	#converts to binary above vs bellow threshhold and then applies the weights using pointwise multiplication
	binary_data = selected_data.gt(thresh).astype('float64')
	#used https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value
	#returns the mean participation of each column
	return binary_data.mean(axis=0)
#These are the functions with one output
#The API is takes in 2 lists of Rankees, rankees and selected
#the output is a single float
def normalized_centroid_distance(rankees, ranking_function, columns=None, k=None, norm_funcs={}):
	if columns == None:
		columns = rankees.get_ranking_columns()
	#~ length = len(columns)
	selected, thresh = ranking_function.get_partial_ranking(rankees, k=k)
	keep=True
	return_stored = True
	if norm_funcs and rankees.norm_funcs != norm_funcs:
		# print('renormalizing')
		temp_norm = rankees.norm_funcs
		rankees.norm_funcs = norm_funcs
		keep=False
		return_stored= False
	selected_data = rankees.get_normalized(keep=keep, return_stored=return_stored).loc[selected , columns]
	all_data = rankees.get_normalized(keep=keep).loc[ : , columns]
	if not keep:
		rankees.norm_funcs = temp_norm
	total_centroid = all_data.mean()
	centroid = selected_data.mean()
	#get the norm of each one
	pointwise_distance = selected_data.sub(centroid).apply(np.linalg.norm, axis = 1)
	selected_centroid_distance = pointwise_distance.mean()

	pointwise_distance = all_data.sub(total_centroid).apply(np.linalg.norm, axis = 1)
	total_centroid_distance = pointwise_distance.mean()

	#~ print(centroid)
	return pd.Series(selected_centroid_distance / total_centroid_distance)

	#~ print(total_centroid)
def centroid_distance(rankees, ranking_function, columns=None):
	if columns == None:
		columns = rankees.get_ranking_columns()
	#~ length = len(columns)
	selected, thresh = ranking_function.get_partial_ranking(rankees)
	selected_data = rankees.get_normalized().loc[selected , columns]
	centroid = selected_data.mean()
	#get the norm of each one
	pointwise_distance = selected_data.sub(centroid).apply(np.linalg.norm, axis = 1)
	#~ print(centroid)
	return pointwise_distance.mean()
