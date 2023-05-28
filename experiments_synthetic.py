import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

import ray

import IPython
from algorithmsmodsel import get_modsel_manager
#from bandit_algs import UCBalgorithm, LUCBalgorithm, EXP3
#from bandit_envs import BernoulliBandit, GaussianBandit, LinearBandit, LinearContextualBandit, SphereContextDistribution
from utilities import pickle_and_zip, unzip_and_load_pickle, produce_parallelism_schedule, write_dictionary_file, get_conditional_filename_hashing
from experiment_parameters import get_experiment_info



def run_experiments(bandit, arm_set_type, get_base_algorithms, num_timesteps, parameters,  
	modselalgo = "Corral", name = ""):
	

	if modselalgo in ["BalancingClassic", "DoublingDataDrivenBig", "EstimatingDataDrivenBig"]:
		dmin = 1
		per_parameter_putative_bonds = [2**i*dmin for i in range(int(np.log(num_timesteps)))]
		putative_bounds = per_parameter_putative_bonds*len(parameters)
		processed_parameters = []
		for p in parameters:
			processed_parameters += [p]*len(per_parameter_putative_bonds)

	else:
		processed_parameters = parameters
		putative_bounds = []


	base_algorithms = get_base_algorithms(processed_parameters)

	modsel_manager = get_modsel_manager(modselalgo,len(processed_parameters), num_timesteps, parameters = putative_bounds )


	rewards = []
	pseudo_rewards = []
	instantaneous_regrets = []
	probabilities = []


	per_algorithm_regrets = [[] for _ in range(len(processed_parameters))]

	arms_played = []#[0 for _ in range(num_arms)]
	parameter_pulls = [0 for _ in range(len(processed_parameters))]

	modsel_infos = []

	for t in range(num_timesteps):
		print("Timestep {}".format(t))
		modsel_sample_idx = modsel_manager.sample_base_index()
		probabilities.append(modsel_manager.get_distribution())
		parameter_pulls[modsel_sample_idx] += 1
		parameter = processed_parameters[modsel_sample_idx]

		print("Selected parameter {}".format(parameter))
		print("Modselalgo {}".format(modselalgo))
		print(name)

		base_algorithm = base_algorithms[modsel_sample_idx]

		context = bandit.get_context()
		arm_info = (arm_set_type, context)

		#IPython.embed()

		play_arm = base_algorithm.get_arm(parameter,arm_info = arm_info)
		


		#arms_played.append(play_arm)
		
		reward = bandit.get_reward(play_arm)
		rewards.append(reward)

		modsel_info = dict([])
		modsel_infos.append(modsel_info)

		base_algorithm.update_arm_statistics(play_arm, reward)
		
		mean_reward = bandit.get_arm_mean(play_arm)

		pseudo_rewards.append(mean_reward)

		
		modsel_manager.update_distribution(modsel_sample_idx, reward, modsel_info )


		instantaneous_regret = bandit.get_max_mean() - bandit.get_arm_mean(play_arm)
		instantaneous_regrets.append(instantaneous_regret)

		per_algorithm_regrets[modsel_sample_idx].append(instantaneous_regret)

	result = dict([])
	result["rewards"] = []#rewards
	result["pseudo_rewards"] = []#pseudo_rewards
	result["instantaneous_regrets"] = instantaneous_regrets
	result["arms_played"] = []#arms_played
	result["parameter_pulls"] = []# parameter_pulls
	result["probabilities"] = []#probabilities
	result["per_algorithm_regrets"] =[]# per_algorithm_regrets
	result["modsel_infos"] = []#modsel_infos


	return result



@ray.remote
def run_experiments_remote(bandit, arm_set_type, get_base_algorithms, num_timesteps, parameters,  
	modselalgo = "Corral", name = ""):
	return run_experiments(bandit, arm_set_type, get_base_algorithms, num_timesteps, parameters,  
	modselalgo = modselalgo, name = name)







if __name__ == "__main__":

	np.random.seed(1000)
	random.seed(1000)


	### RUN PARAMETERS
	MAX_PARALLELISM = 2000



	num_timesteps = int(sys.argv[1])
	exp_type = str(sys.argv[2])
	num_experiments = int(sys.argv[3])
	modselalgos = str(sys.argv[4])
	modselalgos = modselalgos.split(",")
	normalize = str(sys.argv[5]) == "True"
	# IPython.embed()
	# raise ValueError("asdflkm")


	### PLOT PARAMETERS
	plot_bbox = True
	USE_RAY = True
	FROM_FILE = False ### If this flag is true then it will try to load from file when possible. If False it will overwrite 
	SAVE_ALL_RUN_DATA = False


	plot_subsampling = 10


	timesteps_plot = np.arange(num_timesteps) + 1
	timesteps_plot = [0] + list(timesteps_plot[plot_subsampling-1::plot_subsampling])


	std_plot_multiplier = 2.0/np.sqrt(num_experiments)


	experiment_info = get_experiment_info(exp_type)

	experiment_name = experiment_info["experiment_name"]
	bandit = experiment_info["bandit"]
	get_base_algorithms = experiment_info["get_base_algorithms_func"]
	arm_set_type = experiment_info["arm_set_type"]
	experiment_tag = experiment_info["experiment_tag"]
	parameters = experiment_info["parameters"]
	plot_parameter_name = experiment_info["plot_parameter_name"]




	path = os.getcwd()
	exp_data_dir = "{}/paper_synthetic_exps/{}".format(path, experiment_name)

	if not os.path.exists(exp_data_dir):
		os.mkdir(exp_data_dir)

	exp_data_dir_T = "{}/T{}".format(exp_data_dir, num_timesteps)
	if not os.path.exists(exp_data_dir_T):
		os.mkdir(exp_data_dir_T)



	colors = ["red", "orange", "violet", "black", "brown", "yellow", "green", "gray", "cyan", "purple", 
	"darkkhaki", "salmon", "aquamarine", "sienna", "darkorchid", "mediumturquoise", "darkorange"]*10	

	normalization_visualization = 1.0/np.sqrt( np.arange(num_timesteps) + 1)
	normalization_visualization *= 1.0/np.log( np.arange(num_timesteps) + 2)



	#### RUN THE BASELINES
	baselines_results = []

	### Only run base learning experiments for the set of parameters. This avoids the self model-selection experiments to needlessly run multiple experiment copies.
	reduced_parameters = list(set(parameters))
	reduced_parameters.sort()


	log_filename = "data_base_{}_T{}".format(experiment_name,num_timesteps)
	
	if SAVE_ALL_RUN_DATA:
		all_base_data_log_filename = "alldata_{}_T{}".format(experiment_name, num_timesteps)

	baselines_all_data = []


	if FROM_FILE or os.path.exists("{}/{}.zip".format(exp_data_dir_T, log_filename)):
		baselines_results = unzip_and_load_pickle(exp_data_dir_T, log_filename, is_zip_file = True)


	else:
		for parameter, i in zip(reduced_parameters, range(len(reduced_parameters))):
				cum_regrets_all = []	
				#confidence_radius_pulls_all = []

				results = []
				if USE_RAY:
					parallelism_schedule = produce_parallelism_schedule(num_experiments, MAX_PARALLELISM)
					for batch in parallelism_schedule:
						partial_results = [run_experiments_remote.remote(bandit, arm_set_type, get_base_algorithms, num_timesteps, 
									[parameter],  modselalgo = "Corral", name = "{} base".format(experiment_name)) for _ in range(batch) ]
						partial_results = ray.get(partial_results)
						results += partial_results

				else:
					for _ in range(num_experiments):
						result = run_experiments(bandit, arm_set_type, get_base_algorithms, num_timesteps, 
							[parameter],  modselalgo = "Corral", name = "{} base".format(experiment_name)) ### Here we can use any modselalgo, it is dummy in this case.
						results.append(result)



				for result in results:
					rewards = result["rewards"]
					mean_rewards = result["pseudo_rewards"]
					instantaneous_regrets = result["instantaneous_regrets"]
					arm_pulls = result["arms_played"]


					cum_regrets_all.append(np.cumsum(instantaneous_regrets))




				mean_cum_regrets = np.mean(cum_regrets_all,0)
				std_cum_regrets = np.std(cum_regrets_all,0)


				### BASELINES RESULTS is not used
				baselines_results.append((mean_cum_regrets, std_cum_regrets))

				baselines_all_data.append((parameter, cum_regrets_all))


		if SAVE_ALL_RUN_DATA:
			pickle_and_zip(baselines_all_data, all_base_data_log_filename, exp_data_dir_T, is_zip_file = True, hash_filename  = False )
		
		pickle_and_zip(baselines_results, log_filename, exp_data_dir_T, is_zip_file = True, hash_filename  = False)


	final_mean_rewards_stds= dict([])
	modsel_names = ""
	for modselalgo, i in zip(modselalgos, range(len(modselalgos))):
	

		modsel_names += modselalgo
		modsel_cum_regrets_all = []	
		modsel_confidence_radius_pulls_all = []
		probabilities_all = []
		per_algorithm_regrets_stats = []

		log_filename = "data_{}_{}_T{}".format(modselalgo, experiment_name, num_timesteps)


		if SAVE_ALL_RUN_DATA:
			all_modsel_data_log_filename = "alldata_{}_{}_T{}".format(modselalgo, experiment_name, num_timesteps)

		if FROM_FILE or os.path.exists("{}/{}.zip".format(exp_data_dir_T, log_filename)): 
			(mean_modsel_cum_regrets, std_modsel_cum_regrets) = unzip_and_load_pickle(exp_data_dir_T, log_filename, is_zip_file = True, hash_filename  = False)

		else:

			results = []
			if USE_RAY:
				parallelism_schedule = produce_parallelism_schedule(num_experiments, MAX_PARALLELISM)
				for batch in parallelism_schedule:
					partial_results = [run_experiments_remote.remote(
								bandit, 
								arm_set_type, 
								get_base_algorithms,
								num_timesteps, 
								parameters,  
								modselalgo = modselalgo, 
								name = "{} modsel".format(experiment_name),
								) for _ in range(batch)]
					partial_results = ray.get(partial_results)
					results += partial_results
			else:
				for _ in range(num_experiments):

					result = run_experiments(
							bandit, 
							arm_set_type, 
							get_base_algorithms,
							num_timesteps, 
							parameters,  
							modselalgo = modselalgo, 
							name = "{} modsel".format(experiment_name),
							)
					results.append(result)


			for result in results:
				modsel_rewards = result["rewards"] 
				modsel_mean_rewards = result["pseudo_rewards"] 
				modsel_instantaneous_regrets = result["instantaneous_regrets"]
				modsel_arm_pulls = result["arms_played"] 
				modsel_confidence_radius_pulls = result["parameter_pulls"] 
				probabilities_modsel = result["probabilities"] 
				per_algorithm_regrets = result["per_algorithm_regrets"] 
				

				modsel_cum_regrets_all.append(np.cumsum(modsel_instantaneous_regrets))
				modsel_confidence_radius_pulls_all.append(modsel_confidence_radius_pulls)
				per_algorithm_regrets_stats.append(per_algorithm_regrets)


			mean_modsel_cum_regrets = np.mean(modsel_cum_regrets_all,0)
			std_modsel_cum_regrets = np.std(modsel_cum_regrets_all,0)
		
			if SAVE_ALL_RUN_DATA:
				pickle_and_zip(modsel_cum_regrets_all, all_modsel_data_log_filename, exp_data_dir_T, is_zip_file = True)
			
			pickle_and_zip((mean_modsel_cum_regrets, std_modsel_cum_regrets), log_filename, exp_data_dir_T, is_zip_file = True)



		final_mean_rewards_stds[modselalgo] = (int(mean_modsel_cum_regrets[-1]), int(2*std_modsel_cum_regrets[-1]/np.sqrt(num_experiments)))


		if normalize:
			mean_modsel_cum_regrets *= normalization_visualization
			std_modsel_cum_regrets *= normalization_visualization


		### PLOT THE MODEL SELECTION REGRETS
		if modselalgo == "EstimatingDataDriven":
			modselalgo_tag = r'$\mathrm{E}\mathrm{D}^2\mathrm{RB}$'
		elif modselalgo == "DoublingDataDriven":
			modselalgo_tag = r'$\mathrm{D}^3\mathrm{RB}$'
		elif modselalgo == "EstimatingDataDrivenBig":
			modselalgo_tag = r'$\mathrm{E}\mathrm{D}^2\mathrm{RB}$ Grid'
		elif modselalgo == "DoublingDataDrivenBig":
			modselalgo_tag = r'$\mathrm{D}^3\mathrm{RB}$ Grid'
		elif modselalgo == "BalancingClassic":
			modselalgo_tag = "RB Grid"
		else:
			modselalgo_tag = modselalgo


		### PLOTTING MODSEL REGRET
		mean_modsel_at_zero = mean_modsel_cum_regrets[0]
		std_modsel_at_zero = std_modsel_cum_regrets[0]
		subsampled_mean_modsel_cum_regrets = np.array([mean_modsel_at_zero] + list(mean_modsel_cum_regrets[plot_subsampling-1::plot_subsampling]))
		subsampled_std_modsel_cum_regrets = np.array([std_modsel_at_zero] +  list(std_modsel_cum_regrets[plot_subsampling-1::plot_subsampling]))


		plt.plot(timesteps_plot, subsampled_mean_modsel_cum_regrets, label = modselalgo_tag, color = colors[i], linewidth = 3, linestyle = "dashed" )
		plt.fill_between(timesteps_plot,subsampled_mean_modsel_cum_regrets -std_plot_multiplier*subsampled_std_modsel_cum_regrets, 
			subsampled_mean_modsel_cum_regrets +std_plot_multiplier*subsampled_std_modsel_cum_regrets, color = colors[i], alpha = .1   )



	mean_rewards_log_filename_stub = get_conditional_filename_hashing("final_mean_rewards_{}_{}_T{}".format(experiment_name, modsel_names, num_timesteps))

	final_mean_rewards_log_filename = "{}/{}.txt".format(exp_data_dir_T, mean_rewards_log_filename_stub)
	


	write_dictionary_file(final_mean_rewards_stds, final_mean_rewards_log_filename)


	### PLOT THE BASE LEARNER'S MEAN REGRETS.
	for parameter, baseline_result_tuple, color in zip(reduced_parameters, baselines_results, colors):
		mean_cum_regrets, std_cum_regrets = baseline_result_tuple
		if normalize:
					mean_cum_regrets *= normalization_visualization	
					std_cum_regrets *= normalization_visualization

		
		#plt.plot(np.arange(num_timesteps) + 1, mean_cum_regrets, label = "{}{}".format(plot_parameter_name, parameter), color = color )
		#plt.fill_between(np.arange(num_timesteps) + 1,mean_cum_regrets - std_plot_multiplier*std_cum_regrets,mean_cum_regrets + std_plot_multiplier*std_cum_regrets, alpha = .05 , color = color )
		
		mean_at_zero = mean_cum_regrets[0] 
		std_at_zero = std_cum_regrets[0]

		subsampled_mean_cum_regrets = np.array([mean_at_zero] + list(mean_cum_regrets[plot_subsampling-1::plot_subsampling]))
		subsampled_std_cum_regrets = np.array([std_at_zero] + list(std_cum_regrets[plot_subsampling-1::plot_subsampling]))

		plt.plot(timesteps_plot, subsampled_mean_cum_regrets, label = "{}{}".format(plot_parameter_name, parameter), color = color )
		plt.fill_between(timesteps_plot,subsampled_mean_cum_regrets - std_plot_multiplier*subsampled_std_cum_regrets,
			subsampled_mean_cum_regrets + std_plot_multiplier*subsampled_std_cum_regrets, alpha = .05 , color = color )

	
	plt.title("{}".format(experiment_tag), fontsize = 18)

	if plot_bbox:
		plt.legend(bbox_to_anchor = (1.0, 1.03), loc="upper left", fontsize = 13)
		#plt.tight_layout()

	else:		
		plt.legend( loc="upper left", fontsize = 9.8)

	plt.xlabel("Rounds", fontsize =13)
	#plt.ylim(0,1.5)
	
	if normalize:
		plot_name_stub = get_conditional_filename_hashing("norm_{}_{}_T{}".format(experiment_name, modsel_names,num_timesteps))
		plot_name = "{}/{}.pdf".format(exp_data_dir_T, plot_name_stub)

		plt.ylabel("Regret Scale", fontsize =13)
		if plot_bbox:
			plt.savefig(plot_name, bbox_inches='tight')
		else:
			plt.savefig(plot_name)

	else:
		plt.ylabel("Cumulative Regret", fontsize =13)
		plot_name_stub = get_conditional_filename_hashing("{}_{}_T{}".format(experiment_name, modsel_names,num_timesteps))
		plot_name = "{}/{}.pdf".format(exp_data_dir_T, plot_name_stub)

		if plot_bbox:
			plt.savefig(plot_name, bbox_inches='tight')
		else:
			plt.savefig(plot_name)

	plt.close("all")


		

