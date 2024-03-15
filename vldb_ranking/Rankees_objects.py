import pandas
import functools
import metric_function_library
from sys import argv
import cProfile
VERBOSE = False
def default_clean(data):
	num = float(data)
	if num > 1:
		return None
	return num
def to_float_clean(data):
	if data != 0 and not data:
		return None
	try:
		return float(data)
	except ValueError:
		return None
def default_drop(row):
	print(row)
	return True
def drop_missing(data, columns):
	if columns != None:
		return data.dropna(
	    axis=0,
	    how='any',
	    inplace=False,
	    subset=columns
	    )
	else:
		return data.dropna(
		    axis=0,
		    how='any',
		    inplace=False
		)
def half_normalizer(item):
	return item * 0.5
def weighted_sum_all(weights, row, columns_to_use):
	row = list(row)
	return sum([weights[i] * row[i] for i in range(len(row))])
def equal_weights(row, columns_to_use):
	row = dict(row)
	return sum([row[item] for item in columns_to_use]) / len(columns_to_use)
def weighted_sum_few(weights, row, columns_to_use):
	row = dict(row)
	# print(row, weights, columns_to_use)
	# print([weights[item] * row[item] for item in columns_to_use])
	return sum([weights[item] * row[item] for item in columns_to_use])
def selct_func(scoring_func, row, columns_to_use, k):
	return scoring_func(row, columns_to_use) >= k
def weighted_sum_few_generator(weights):
	return functools.partial(weighted_sum_few, weights)
def weighted_sum_all_generator(weights):
	return functools.partial(weighted_sum_all, weights)
def select_generator(weighted_func):
	return functools.partial(selct_func, weighted_func)
"""
This is the class representing the data, it is in charge of cleaning and normalizing the data.
The most important construction parameter (and the only one without defaults) is the source file which should be a CSV

The cleaning and normalizing functions are meant to be done on a per-column basis, but if we need fancier functions that combine data
from multiple sources that is also supported

It is recommended to pass ranking columns as the columns that are used while ranking since this speeds up the sorting considerabley
"""
class Rankees:
	def __init__(self, source_file, cleaning_functions={}, dropping_function=drop_missing, ranking_columns=None, norm_funcs = {}, full_frame_norm_func = None, dropping_columns = None, dtypes={}):
	#important columns are the ones we rank on
		if not dropping_columns:
			dropping_columns = ranking_columns
		self.source_file = source_file
		self.clean_results = self.clean_data(cleaning_functions, dropping_function, dropping_columns, dtypes)
		if ranking_columns:
			self.set_ranking_columns(ranking_columns)
		else:
			self.ranking_columns = None

		self.normalized_results = None
		self.norm_funcs = norm_funcs
		self.full_frame_norm_func = full_frame_norm_func
	def clean_data(self, cleaning_functions, dropping_function, columns, dtypes):
		data = pandas.read_csv(self.source_file, dtype=dtypes)
		print('done')
		#~ print(data.dtypes, data)
		for column in data.columns:
			if column in cleaning_functions:
				cleaning = cleaning_functions[column]
				data[column] = data[column].map(cleaning)
		print(f'columns is{data.columns}')
		data = dropping_function(data, columns)
		#print(data.dtypes, data)
		return data
	#This function is the main function that gets called from this class it returns cleaned normalized data
	def get_normalized(self, normalizing_funcs={}, full_frame_norm_func=None,keep=True, return_stored=True):
		# == operator is overloaded for dataframes, so this was the simplest way to check if the object is None or a dataframe
		if return_stored and type(self.normalized_results).__name__ == 'DataFrame':
			return self.normalized_results
		normalized_results = self.clean_results.copy(deep=True)
		if not normalizing_funcs and not full_frame_norm_func:
			  normalizing_funcs = self.norm_funcs
			  full_frame_norm_func = self.full_frame_norm_func
		for column in normalized_results.columns:
			if column in normalizing_funcs:
				norm_func = normalizing_funcs[column]
				normalized_results[column] = normalized_results[column].map(norm_func)
		if full_frame_norm_func:
			normalized_results = full_frame_norm_func(normalized_results, self.get_ranking_columns())
		if keep:
			self.normalized_results = normalized_results
		return normalized_results
	def set_ranking_columns(self, columns):
		self.ranking_columns = [column for column in columns if column in self.clean_results.columns]
	def get_ranking_columns(self):
		if not self.ranking_columns:
			self.ranking_columns = list(self.clean_results.columns)
		return self.ranking_columns
# class No_Sort_Ranking_function(Ranking_function):
# 	@Override
# 	def get_partial_ranking(self, rankees_object, k=None,keep=True, return_stored=True, store_thresh=True):
# 		if k != None:
# 			raise ValueError
# 			return None
#

"""
This class represents the ranking function, for now it just accepts scoring functions

If you just want a weighted sum function, simply give weights in the wieghts parameter and leave all others as default.

If another function is desired, set scoreing_func to true and pass the function to the func parameter.

See get_partial_ranking for a description of the most important function

"""
class Ranking_function:
	def __init__(self, func=None, select_func=False, compare_func=False, scoring_func=False, weights=None, k=0):
		#~ if not (select_func or compare_func or scoring_func):
			#~ print('No function given')
			#~ exit(0)
		if weights:
			if select_func:
				self.weights = weights
				if weights != 'equal':
					func = select_generator(weighted_sum_few_generator(weights))
				else:
					func = select_generator(equal_weights)
			else:
				scoring_func = True
				self.weights = weights
				if weights != 'equal':
					func = weighted_sum_few_generator(weights)
				else:
					func = equal_weights

		else:
			self.weights = None
		if select_func + compare_func + scoring_func != 1:
			print('wrong number of funcs give')
			exit(0)
		self.func = func
		self.k = k
		self.last_k = None
		self.partial = None
		self.full = None
		self.selection = None
		self.select_func = select_func
		self.compare_func = compare_func
		self.scoring_func = scoring_func
		self.thresh = None
	def get_full_ranking(self, rankees_object,  columns_to_use=None, keep=True, return_stored=True):
		if return_stored and self.full:
			return self.full
		if self.select_func:
			return None
		data = rankees_object.get_normalized(return_stored=True)
		full = list(data.index.values)
		if self.scoring_func:
			full = sorted(full, key=lambda rankee: self.func(data.loc[rankee], rankees_object.get_ranking_columns()), reverse=True)
		elif self.compare_func:
			full = sorted(full, key=functools.cmp_to_key(self.func), reverse=True)
		else:
			return None
		if keep:
			self.full = full
		return full
	def get_partial_ranking_select(self, rankees_object, k=None,keep=True, return_stored=True, store_thresh=True):
		if return_stored and self.partial and (k == self.last_k or (k == None and self.last_k == self.k)) and self.thresh:
			return self.partial, self.thresh
		if k == None:
			k = self.k
		data = rankees_object.get_normalized(return_stored=return_stored).loc[: , rankees_object.get_ranking_columns()]
		partial = list(data.index.values)
		#print(partial)
		if self.scoring_func:
			if VERBOSE:
				print('recalculating')
			# print(self.func(data.loc[0], rankees_object.get_ranking_columns()))
		elif self.compare_func:
			print('why here?')
		# full = sorted(partial, key=lambda rankee: self.func(data.loc[rankee], rankees_object.get_ranking_columns()), reverse=True)

		partial = [index for index in partial if self.func(data.loc[index],rankees_object.get_ranking_columns(), k)]
		if keep:
			self.partial = partial
			self.last_k = k
		if store_thresh and self.scoring_func:
			self.thresh = k
			return partial, self.thresh
		return partial, k
	#this is the main function run from this class.
	#it takes in a Rankees object and gives a partial ranking based on the scoring function
	#it is made to return the stored data if it exists to minimize resorting of the data when not needed
	#if you want the data resorted set return_stored to False
	#this is the only way to get the threshold (which is returned along with the ranking) to prevent old thresholds from being used.
	def get_partial_ranking(self, rankees_object, k=None,keep=True, return_stored=True, store_thresh=True):
		if self.select_func:
			return self.get_partial_ranking_select(rankees_object, k=k,keep=keep, return_stored=return_stored, store_thresh=store_thresh)
		if return_stored and self.partial and (k == self.last_k or (k == None and self.last_k == self.k)) and self.thresh:
			return self.partial, self.thresh

		elif return_stored and self.full:
			data = rankees_object.get_normalized(return_stored=return_stored).loc[: , rankees_object.get_ranking_columns()]
			if k == None:
				k = self.k
			partial = self.full[:max(round(len(rankees_object.get_normalized().index) * k), 1)]
			if keep:
				self.partial = partial
				self.last_k = k
			print
			self.thresh = self.func(data.loc[partial[-1]], rankees_object.get_ranking_columns())
			# print('thresh 0 {} thresh -1 {} thresh min {}'.format(
			# 	self.func(data.loc[partial[0]], rankees_object.get_ranking_columns()),
			#
			# 	self.thresh
			# ))
			return 	partial, self.thresh
		if self.select_func:
			return None
		if k == None:
			k = self.k
		data = rankees_object.get_normalized(return_stored=return_stored).loc[: , rankees_object.get_ranking_columns()]
		partial = list(data.index.values)
		#print(partial)
		if self.scoring_func:
			if VERBOSE:
				print('recalculating')
			# print(self.func(data.loc[0], rankees_object.get_ranking_columns()))
			full = sorted(partial, key=lambda rankee: self.func(data.loc[rankee], rankees_object.get_ranking_columns()), reverse=True)
		elif self.compare_func:
			print('why here?')
			full = sorted(partial, key=functools.cmp_to_key(self.func), reverse=True)
		partial = full[:max(round(len(data.index) * k), 1)]
		if keep:
			self.partial = partial
			self.full = full
			self.last_k = k
		if store_thresh and self.scoring_func:
			self.thresh = min([self.func(data.loc[rankee], rankees_object.get_ranking_columns()) for rankee in partial])
			return partial, self.thresh
		return partial
	# def get_thresh(self):
	# 	if self.thresh:
	# 		return self.thresh
	# 	else:
	# 		self.get_partial_ranking()
	def get_weights(self):
		if self.weights:
			return self.weights
		else:
			return None

"""
This class is the main way other files should interact with the system, it allows us to run metrics with a given Rankees object and ranking function

it mostly refrences other objects to do the work
"""
class Metric_computer:
	def __init__(self, rankees, ranking_function):
		self.rankees = rankees
		self.ranking_function = ranking_function
		self.last_result = Results(None)
	#this returns a list of results based on a list of functions passed to it
	def get_many_metrics_partial(self, metrics, k=None, columns=None, return_stored=True, norm_funcs={}):
		if return_stored == False:
			self.ranking_function.get_partial_ranking(self.rankees, return_stored=False)
		#~ #this line makes sure the results are current so the metric functions can retrieve the stored results
		#~ self.ranking_function.get_partial_ranking(self.rankees, k, return_stored=False)
		self.last_result = Results([metric(self.rankees, self.ranking_function, columns, k=k, norm_funcs=norm_funcs) for metric in metrics], k)
		return self.last_result.results
	#this returns a single result based on a  of functions passed to it
	def get_metric_partial(self, metric, k=None, columns=None, return_stored=True, norm_funcs={}):
		if return_stored == False:
			self.ranking_function.get_partial_ranking(self.rankees, return_stored=False)
		#~ #this line makes sure the results are current so the metric function can retrieve the stored results
		#~ self.ranking_function.get_partial_ranking(self.rankees, k, return_stored=False)
		self.last_result = Results(metric(self.rankees, self.ranking_function, columns, k=k, norm_funcs=norm_funcs), k)
		return self.last_result.results
	#in the future if we want a metric on all the data we can use these functions
	#~ def get_metric_full(self, metric, columns=None):
		#~ return Results(metric(self.rankees, self.ranking_function, columns), self.rankees.k)
	#~ def get_many_metrics_full(self, metrics, columns=None):
		#~ return Results([metric(self.rankees, self.ranking_function, columns) for metric in metrics])
	#returns the selected data
	def get_selected(self, k=None, return_stored=True):
		selected_nums, _ = self.ranking_function.get_partial_ranking(self.rankees, k, return_stored=return_stored)
		self.last_result = Results(self.rankees.clean_results.loc[selected_nums, :])
		return self.last_result.results
	def get_sorted(self, return_stored=True):
		selected_nums = self.ranking_function.get_full_ranking(self.rankees, return_stored=return_stored)
		self.last_result = Results(self.rankees.clean_results.loc[selected_nums, :])
		return self.last_result.results
	#returns the unselected data
	def get_normalized_sorted(self, return_stored=True):
		selected_nums = self.ranking_function.get_full_ranking(self.rankees, return_stored=return_stored)
		self.last_result = Results(self.rankees.get_normalized().loc[selected_nums, :])
		return self.last_result.results
	def get_normalized_selected(self, k=None, return_stored=True):
		selected_nums, _ = self.ranking_function.get_partial_ranking(self.rankees, k, return_stored=return_stored)
		self.last_result = Results(self.rankees.get_normalized().loc[selected_nums, :])
		return self.last_result.results
	def get_unselected(self, k=None):
		selected_nums = set(self.ranking_function.get_partial_ranking(self.rankees, k)[0])
		unselected_nums = [num for num in self.rankees.clean_results.index.values if num not in selected_nums]
		self.last_result = Results(self.rankees.clean_results.loc[unselected_nums, :])
		return self.last_result.results
	#returns all the data
	def get_all(self):
		self.last_result = Results(self.rankees.clean_results)
		return self.last_result.results
	def get_scores(self, return_stored=True):
		if not self.ranking_function.scoring_func:
			return None
		def temp(row):
			return self.ranking_function.func(row, self.rankees.get_ranking_columns())
		self.last_result = Results(self.rankees.get_normalized(return_stored=return_stored).apply(temp, axis=1))
		return self.last_result.results


#simple results class, not currently doing much but may be useful in the future.
class Results:
	def __init__(self, results, k=0):
		self.results = results
	def get_results(self):
		return self.results
	def __repr__(self):
		return str(self.results)

if __name__ == '__main__':
	#~ metrics = prep_7_th_grade_data('/res/users/amelie/NYCDOEData/AllNYC_7_16-17.csv')
	#~ funcs = [metric_function_library.weighted_participation, metric_function_library.participation, metric_function_library.param_floor, metric_function_library.disqualifying_power,
		#~ metric_function_library.centroid_distance, metric_function_library.normalized_centroid_distance, metric_function_library.disparity]
	#~ print(metrics.get_many_metrics_partial(funcs))
	#~ print(metrics.get_metric_partial(metric_function_library.disparity, columns = ['swd', 'poverty']))
	#~ exit(0)
	if argv[2] == 'e':
		cleaning_functions = {'Uniform_0-1':default_clean,  'Normal_0.5|0.05':default_clean,  'Normal_0.5|0.15':default_clean,  'Normal_0.75|0.05':default_clean}
		source_file = argv[1]
		rankees = Rankees(source_file,cleaning_functions)
		#~ print(rankees.get_normalized({'Uniform_0-1':half_normalizer}))
		weights = {'Uniform_0-1':0.25,  'Normal_0.5|0.05':0.25,  'Normal_0.5|0.15':0.3,  'Normal_0.75|0.05':0.2}
		ranker = Ranking_function(weights=weights, k=0.05)
		#~ print(ranker.get_partial_ranking(rankees, 2))
		metrics = Metric_computer(rankees, ranker)
		#~ print(metrics.get_selected())
		#~ print(metrics.get_unselected())
		print(metrics.get_metric_partial(metric_function_library.weighted_participation, 0.01))
		funcs = [metric_function_library.weighted_participation, metric_function_library.participation, metric_function_library.param_floor, metric_function_library.disqualifying_power,
					metric_function_library.centroid_distance, metric_function_library.normalized_centroid_distance, metric_function_library.disparity]
		print(metrics.get_many_metrics_partial(funcs))
		print([func.__name__ for func in funcs])
	else:
		metrics = prep_7_th_grade_data(argv[1])
		funcs = [metric_function_library.weighted_participation, metric_function_library.participation, metric_function_library.param_floor, metric_function_library.disqualifying_power,
			metric_function_library.centroid_distance, metric_function_library.normalized_centroid_distance, metric_function_library.disparity]
		print(metrics.get_many_metrics_partial(funcs))
		print(metrics.get_metric_partial(metric_function_library.disparity, columns = ['swd', 'poverty']))
