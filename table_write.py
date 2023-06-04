### WRITE LATEX RESULTS TABLE

import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import os


import IPython
from utilities import read_tuple_file, get_conditional_filename_hashing
from experiment_parameters import get_experiment_info
from experiments_synthetic import get_modselalgo_tag#(modselalgo, shared_data, grid)


if __name__ == "__main__":





	num_timesteps = int(sys.argv[1])
	exp_types_original = str(sys.argv[2])
	exp_types = exp_types_original.split(",")
	num_experiments = int(sys.argv[3])
	original_modselalgos = str(sys.argv[4])
	modselalgos = original_modselalgos.split(",")



	#final_modselalgos_names = [get_modselalgo_tag(modselalgo, shared_data, grid) for modselalgo in modselalgos]



	path = os.getcwd()

	table_data_dir = "{}/paper_synthetic_exps/tables/T{}".format(path, num_timesteps)



	if not os.path.exists(table_data_dir):
		os.mkdir(table_data_dir)


	mean_data_matrix = np.zeros((len(exp_types), len(modselalgos)))
	std_data_matrix = np.zeros((len(exp_types), len(modselalgos)))

	#IPython.embed()

	for i,exp_type in enumerate(exp_types):
		results = []
		print("exp type {}".format(exp_type))

		for j,modselalgo in enumerate(modselalgos):

			experiment_info = get_experiment_info(exp_type)
			experiment_name = experiment_info["experiment_name"]

			exp_data_dir = "{}/paper_synthetic_exps/{}".format(path, experiment_name)
			#exp_data_dir = "{}/paper_synthetic_exps/{}".format(path, experiment_name)

			if not os.path.exists(table_data_dir):
				raise ValueError("{} does not exist".format(table_data_dir))


			exp_data_dir_T = "{}/T{}/final_means".format(exp_data_dir, num_timesteps)
			if not os.path.exists(exp_data_dir_T):
				raise ValueError("{} does not exist".format(exp_data_dir_T))


			mean_rewards_log_filename_stub = get_conditional_filename_hashing("final_mean_rewards_{}_{}_T{}".format(experiment_name, 
				modselalgo, num_timesteps))

			final_mean_rewards_log_filename = "{}/{}.txt".format(exp_data_dir_T, mean_rewards_log_filename_stub)


			#### Load data
			(mean, std) = read_tuple_file(final_mean_rewards_log_filename)

			# if len(exp_types) == 1:
			# 	mean_data_matrix[j] = mean
			# 	std_data_matrix[j] = std

			# else:
			mean_data_matrix[i,j] = mean
			std_data_matrix[i,j] = std



	column_names = modselalgos
	row_names = exp_types

	### Create the data matrix
	data_matrix = []
	colors = [["white"]*len(modselalgos)]*len(exp_types)
	for i in range(len(exp_types)):
		data_row = []
		max_value = -float("inf")
		max_value_index = 0
		for j in range(len(modselalgos)):
			# if len(exp_types) == 1:
			# 	data_row.append(r'{}\pm{}'.format(mean_data_matrix[j],std_data_matrix[j]))
			# 	if max_value < mean_data_matrix[j]:
			# 		max_value = mean_data_matrix[j]
			# 		max_value_index = j
			# else:
				data_row.append(r'${}\pm{}$'.format(mean_data_matrix[i,j],std_data_matrix[i,j]))
				if max_value < mean_data_matrix[i,j]:
					max_value = mean_data_matrix[i,j]
					max_value_index = j

		data_matrix.append(data_row)
		colors[i][max_value_index] = "red"


	# Create a figure and axis
	fig, ax = plt.subplots()

	# Create the table
	table = ax.table(cellText=data_matrix,
	                 cellColours=colors,
	                 cellLoc='center',
	                 colLabels=column_names,
	                 rowLabels=row_names,
	                 loc='center'
	                )

	# Modify the appearance of the table
	table.auto_set_font_size(False)
	table.set_fontsize(1)
	table.scale(1.2, 1.2)

	# Hide the axis and axis labels
	ax.axis('off')

	figure_name = get_conditional_filename_hashing("{}_{}".format(exp_types_original, original_modselalgos))
	
	figure_file_name = "{}/{}.pdf".format(table_data_dir, figure_name)
	# Show the plot
	plt.savefig(figure_file_name,  bbox_inches='tight')

	
	# IPython.embed()
	# raise ValueError("alskdf")


		

