### WRITE LATEX RESULTS TABLE

import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import os


import IPython
from utilities import read_dictionary_file, get_conditional_filename_hashing
from experiment_parameters import get_experiment_info



if __name__ == "__main__":




	num_timesteps = int(sys.argv[1])
	exp_types_original = str(sys.argv[2])
	exp_types = exp_types_original.split(",")
	num_experiments = int(sys.argv[3])
	modselalgos = str(sys.argv[4])
	modselalgos = modselalgos.split(",")
	# IPython.embed()
	# raise ValueError("asdflkm")

	table_code = "\\begin{tabular}{|l |"

	table_code += "c|"*len(modselalgos) + "}\n" 
	table_code += "\\hline \n"
	table_code +=  "Name "


	modsel_names = ""
	for modselalgo in modselalgos:
		modsel_names += modselalgo
		table_code += "& " + modselalgo + " "

	table_code += "\\\\\n \\hline \n"

	path = os.getcwd()


	table_data_dir = "{}/paper_synthetic_exps/tables/T{}".format(path, num_timesteps)

	if not os.path.exists(table_data_dir):
		os.mkdir(table_data_dir)






	for exp_type in exp_types:


		experiment_info = get_experiment_info(exp_type)

		experiment_name = experiment_info["experiment_name"]


		exp_data_dir = "{}/paper_synthetic_exps/{}".format(path, experiment_name)

		if not os.path.exists(exp_data_dir):
			raise ValueError("{} does not exist".format(exp_data_dir))


		exp_data_dir_T = "{}/T{}".format(exp_data_dir, num_timesteps)
		if not os.path.exists(exp_data_dir_T):
			raise ValueError("{} does not exist".format(exp_data_dir_T))


		mean_rewards_log_filename_stub = get_conditional_filename_hashing("final_mean_rewards_{}_{}_T{}".format(experiment_name, 
			modsel_names, num_timesteps))

		final_mean_rewards_log_filename = "{}/{}.txt".format(exp_data_dir_T, mean_rewards_log_filename_stub)
		

		#### Load data
		dictionary = read_dictionary_file(final_mean_rewards_log_filename)
		print("exp type {}".format(exp_type))
		print(dictionary)

		### Find the smallest mean
		smallest_mean = float("inf")
		smallest_modselalgo = ""
		for modselalgo in modselalgos:
			if dictionary[modselalgo][0] < smallest_mean:
				smallest_mean = dictionary[modselalgo][0]
				smallest_modselalgo = modselalgo



		table_code += experiment_name + " "
		for modselalgo in modselalgos:
			(mean, std) = dictionary[modselalgo]
			if modselalgo == smallest_modselalgo:
				table_code += "& {$\\bf " + str(mean) + "\\pm "+  str(std) + " $}" 
			else:
				table_code += "& {$" + str(mean) + "\\pm "+  str(std) + " $}" 

		table_code += "\\\\\n"


	table_code += "\\hline \n \\end{tabular}"


	resulting_filename_stub = get_conditional_filename_hashing("table_{}_{}".format(exp_types_original,modsel_names), tolerance = 40)
	#IPython.embed()
	resulting_filename = "{}/{}.tex".format(table_data_dir, resulting_filename_stub)


	with open(resulting_filename, "w") as f:
		f.write(table_code)
		f.close()

	
	# IPython.embed()
	# raise ValueError("alskdf")


		

