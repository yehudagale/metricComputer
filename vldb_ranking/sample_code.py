from Real_data_presets import *
def prep_7_th_grade_simulated_data(fl, input_k=0.05):
		scores = ['Math_score' ,'ELA_score']
		ranking_columns = scores
		print(ranking_columns)
		cleaning_functions = {**{column:score_cleaner for column in scores}}
		normalizing_funcs = {**{column:score_normalizer for column in scores}}
		dtypes = {}
		rankees = Rankees(fl, cleaning_functions, ranking_columns=ranking_columns, norm_funcs = normalizing_funcs)
		print(rankees.get_ranking_columns())
		weights = {**{column:(1/len(scores)) for column in scores}}
		ranker = Ranking_function(weights=weights, k=input_k)
		metrics = Metric_computer(rankees, ranker)
		return metrics, rankees, ranker
metrics, _,_ = prep_7_th_grade_simulated_data('./student_info_with_demographics.csv')
#somtimes we want to get all the selected/unselected data
print(metrics.get_selected())
print(metrics.get_unselected())

#the main function is metrics.get_metric_partial which lets you compute a function comparing the selected and unselected data
#somtimes we want to specify columns
print(metrics.get_metric_partial(metric_function_library.disparity, columns=['sex', 'poverty']))
#somtimes we want the default (only the ranking columns)
funcs = [metric_function_library.importance, metric_function_library.participation, metric_function_library.param_floor]
for func in funcs:
	print(func.__name__)
	print(metrics.get_metric_partial(func))
#sometimes functions will only output a single point
print("normalized_centroid_distance")
print(metrics.get_metric_partial(metric_function_library.normalized_centroid_distance)) 
#we can easily change the percent selected:
print(metrics.get_metric_partial(metric_function_library.normalized_centroid_distance, k=0.5)) 


