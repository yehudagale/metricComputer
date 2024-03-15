from Rankees_objects import *
import itertools
from scipy.stats import zscore
"""
this file contains functions designed to allow us to process the real data from the NYC school system using the RAnkees_objects objects
"""
#Cleaning fucntion for test scores
def score_cleaner(data):
	data = to_float_clean(data)
	if data and data >= 1 and data <= 4.5:
		return data
	else:
		return None
#cleaing function for grades
def grade_cleaner(data):
	if data == 'P':
		print('pass')
		data = 61
	elif data == 'F':
		data = 58
	data = to_float_clean(data)
	if data and data >= 54 and data <= 100:
		return data
	else:
		return None
def decile_cleaner(data):
	try:
		temp_data = int(data)
	except ValueError:
		return None
	if temp_data >= 1 and temp_data <= 10:
		return temp_data
	else:
		return None

def fourth_grade_cleaner(data):
	data = to_float_clean(data)
	if data and ((data >= 54 and data <= 100) or (data >= 1 and data <= 4)):
		return data
	else:
		return None
#normalizing function for the grades for school A in the paper
def points_grade_normalizer(grade):
		max_grade = 8.75
		if grade < 54.9:
			return 0
		if grade >= 95:
			return 8.75 / max_grade
		elif grade >= 90:
			return 7.75 / max_grade
		elif grade >= 85:
			return 6.75 / max_grade
		elif grade >= 80:
			return 5 / max_grade
		elif grade >= 65:
			return 3.5 / max_grade
		else:
			return 0
#normalizing function for the test scores for school A in the paper
def points_score_normalizer(score):
		max_score_points = 17.5
		if score >= 4.01:
			return 17.5 / max_score_points
		elif score >= 3.5:
			return 16 / max_score_points
		elif score >= 3:
			return 14.5 / max_score_points
		elif score >= 2.5:
			return 12 / max_score_points
		elif score >= 2:
			return 8 / max_score_points
		else:
			return 0 / max_score_points
#normalizing function for the absensces for school A in the paper
def points_absence_normalizer(days):
	max_points = 15
	if days <= 2:
		return 15 / max_points
	elif days <= 5:
		return 12 / max_points
	elif days <= 8:
		return 9 / max_points
	elif days <= 10:
		return 6 / max_points
	elif days <= 15:
		return 2 / max_points
	else:
		return 0 / max_points
#Seventh grade School A preparer
#takes in a source file and returns the three relevent Rankees objects
def points_school_7th(source_file):
		scores = ['ela_prof_rating', 'math_prof_rating']
		grades = ['ELA_grade', 'Math_grade', 'Science_grade', 'SocialStudies_grade']
		ranking_columns = scores + grades + ['total_abs']
		print(ranking_columns)
		cleaning_functions = {**{column:grade_cleaner for column in grades} , **{column:score_cleaner for column in scores}, 'total_abs':to_float_clean}
		normalizing_funcs = {'sex':M_F_to_0_1, **{column:points_grade_normalizer for column in grades}, **{column:points_score_normalizer for column in scores}, 'total_abs':points_absence_normalizer}
		dtypes = {}
		rankees = Rankees(source_file, cleaning_functions, ranking_columns=ranking_columns, norm_funcs = normalizing_funcs)
		print(rankees.get_ranking_columns())
		weights = {**{column:(8.75/100) for column in grades}, **{column:(17.5/100) for column in scores}, 'total_abs':(30/100)}
		ranker = Ranking_function(weights=weights, k=0.05)
		metrics = Metric_computer(rankees, ranker)
		return metrics, rankees, ranker
# a full frame normalizing function desinged to minimize disparity on sex poverty and disability
def percentile_normalizer_protected(frame, columns):
	protected = ['sex', 'poverty', 'swd']
	values = [[0,1], [0,1], [0,1]]
	#~ protected = ['poverty']
	#~ values = [[0,1]]
	percentiled = []
	for value in itertools.product(*values):
		#select the rows that match value
		temp_df = frame.copy(deep=True)
		truth_lists = []
		truth_series = temp_df[protected[0]] == value[0]
		for i, column in enumerate(protected):
			truth_series &= temp_df[column] == value[i]
		temp_df = temp_df.loc[truth_series]
		#if there are any such rows, replace the value with the rank
		if len(temp_df.index) > 0:
			temp_df[columns] = temp_df[columns].rank(pct=True)
			#~ print(temp_df)
			percentiled.append(temp_df.copy(deep=True))
	#return the concatonation of all the different ones
	return pandas.concat(percentiled)

def zscore_normalizer_protected(frame, columns):
	protected = ['sex']#, 'poverty', 'swd']
	values = [[0,1]]#, [0,1], [0,1]]
	zscored = []
	for value in itertools.product(*values):
		#select the rows that match value
		temp_df = frame.copy(deep=True)
		truth_lists = []
		truth_series = temp_df[protected[0]] == value[0]
		for i, column in enumerate(protected):
			truth_series &= temp_df[column] == value[i]
		temp_df = temp_df.loc[truth_series]
		#if there are any such rows, replace the value with the zscore
		if len(temp_df.index) > 0:
			temp_df[columns] = temp_df[columns].apply(zscore)
			zscored.append(temp_df.copy(deep=True))
	#return the concatonation of all the different ones
	return pandas.concat(zscored)
def zscore_nomalizer(frame, columns):
	temp_df = frame.copy(deep = True)
	temp_df[columns] = temp_df[columns].apply(zscore)
	return temp_df
def zscore_additive_protected(frame, columns):
	protected = ['sex', 'poverty', 'swd']
	values = [[0,1], [0,1], [0,1]]
	to_sum = []
	for column in protected:
		for value in [0, 1]:
			zscored = []
			temp_df = frame.copy(deep=True)
			temp_df = temp_df.loc[temp_df[column] == value]
			if len(temp_df.index) > 0:
				one_column = temp_df[columns].apply(zscore)
				zscored.append(one_column.copy(deep=True))
		to_sum.append(pandas.concat(zscored))
	total = to_sum[0]
	for sum_frame in to_sum[1:]:
		total.add(sum_frame)
	frame[columns] = total
	return frame
def points_adder(frame, columns):
	points = {'sex': -0.1430252033994217, 'swd': -0.19707760676649, 'poverty': -0.3076852418486825}
	for key in points:
		temp_df_0 = frame.loc[frame[key] == 0]
		temp_df_1 = frame.loc[frame[key] == 1]
		temp_df_1[columns] = temp_df_1[columns] - points[key]
		frame = pandas.concat([temp_df_0, temp_df_1])
	return frame
def score_normalizer(item):
	return (item) / 4.5
def grade_normalizer(item):
	return (item) / 100
# def partial
def fourth_grade_normalizer(item):
	if item > 10:
		return (item - 55) / 45
	else:
		return (item - 1) / 3
def M_F_to_0_1(item):
	convert = {'M':0, 'm':0, 'F':1, 'f':1, 'Male':0, 'Female':1}
	return convert[item]
def decile_score_normalizer(item):
	return item / 10
def prep_4_th_grade_data(fl, weights=None):
		scores = ['ela_prof_rating', 'math_prof_rating']
		grades = ['ELA_grade', 'Math_grade']
		ranking_columns = scores + grades
		print(ranking_columns)
		cleaning_functions = {**{column:fourth_grade_cleaner for column in grades} , **{column:score_cleaner for column in scores}}
		normalizing_funcs = {'sex':M_F_to_0_1, **{column:fourth_grade_normalizer for column in grades}, **{column:score_normalizer for column in scores}}
		source_file = fl
		dtypes = {}
		rankees = Rankees(source_file, cleaning_functions, ranking_columns=ranking_columns, norm_funcs = normalizing_funcs, full_frame_norm_func=zscore_normalizer_protected)
		print(rankees.get_ranking_columns())
		if weights:
			weights = {ranking_columns[i]:weights[i] for i in range(len(weights))}
		else:
			weights = {**{column:0.3 for column in grades}, **{column:0.2 for column in scores}}
		ranker = Ranking_function(weights=weights, k=0.05)
		metrics = Metric_computer(rankees, ranker)
		return metrics, rankees, ranker
def prep_7_th_grade_data(fl, weights=None):
		scores = ['ela_prof_rating', 'math_prof_rating']
		grades = ['ELA_grade', 'Math_grade']
		ranking_columns = scores + grades
		print(ranking_columns)
		cleaning_functions = {**{column:grade_cleaner for column in grades} , **{column:score_cleaner for column in scores}}
		normalizing_funcs = {'sex':M_F_to_0_1, **{column:grade_normalizer for column in grades}, **{column:score_normalizer for column in scores}}
		source_file = fl
		dtypes = {}
		rankees = Rankees(source_file, cleaning_functions, ranking_columns=ranking_columns, norm_funcs = normalizing_funcs)
		print(rankees.get_ranking_columns())
		if weights:
			weights = {ranking_columns[i]:weights[i] for i in range(len(weights))}
		else:
			weights = {**{column:0.3 for column in grades}, **{column:0.2 for column in scores}}
		ranker = Ranking_function(weights=weights, k=0.05)
		metrics = Metric_computer(rankees, ranker)
		return metrics, rankees, ranker
#School B preperation function
def prep_7_th_grade_single_column_data(fl, column):
		scores = ['ela_prof_rating', 'math_prof_rating']
		grades = ['CoreGPA', 'ELA_grade', 'Math_grade', 'Science_grade', 'SocialStudies_grade']
		ranking_columns = scores + grades
		print(ranking_columns)
		cleaning_functions = {**{column:grade_cleaner for column in grades} , **{column:score_cleaner for column in scores}}
		normalizing_funcs = {'sex':M_F_to_0_1, **{column:grade_normalizer for column in grades}, **{column:score_normalizer for column in scores}}
		dtypes = {}
		new_clean = {column:cleaning_functions[column]}
		new_norm = {column:normalizing_funcs[column]}
		rankees = Rankees(fl, new_clean, ranking_columns=[column], norm_funcs = normalizing_funcs)
		print(rankees.get_ranking_columns())
		weights = {column:1}
		ranker = Ranking_function(weights=weights, k=0.05)
		metrics = Metric_computer(rankees, ranker)
		return metrics, rankees, ranker

def prep_7_th_grade_WS_data(fl):
		scores = ['ela_prof_rating', 'math_prof_rating']
		grades = ['CoreGPA']
		ranking_columns = scores + grades
		print(ranking_columns)
		cleaning_functions = {**{column:grade_cleaner for column in grades} , **{column:score_cleaner for column in scores}}
		normalizing_funcs = {'sex':M_F_to_0_1, **{column:grade_normalizer for column in grades}, **{column:score_normalizer for column in scores}}
		dtypes = {}
		rankees = Rankees(fl, cleaning_functions, ranking_columns=ranking_columns, norm_funcs = normalizing_funcs)
		print(rankees.get_ranking_columns())
		weights = {**{column:(0.55) for column in grades}, **{column:(0.45/2) for column in scores}}
		ranker = Ranking_function(weights=weights, k=0.05)
		metrics = Metric_computer(rankees, ranker)
		return metrics, rankees, ranker
