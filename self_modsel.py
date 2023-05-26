import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

import IPython
from algorithmsmodsel import CorralHyperparam, EXP3Hyperparam, UCBHyperparam, BalancingHyperparamDoublingDataDriven, get_modsel_manager
from bandit_algs import UCBalgorithm

from utilities import pickle_and_zip, unzip_and_load_pickle, produce_parallelism_schedule


from bandit_envs import BernoulliBandit, GaussianBandit

np.random.seed(1000)
random.seed(1000)


def test_MAB_modsel(means, stds, scalings, num_timesteps, confidence_radii,  
	modselalgo = "Corral", split = False, algotype = "bernoulli"):
	
	modsel_manager = get_modsel_manager(modselalgo,len(confidence_radii), num_timesteps )

	if algotype == "bernoulli":
		bandit = BernoulliBandit(means, scalings)
	elif algotype == "gaussian":
		bandit = GaussianBandit(means, stds)
	else:
		raise ValueError("unrecognized bandit type {}".format(algotype))

	num_arms = len(means)

	if split:
		ucb_algorithms = [UCBalgorithm(num_arms) for _ in range(len(confidence_radii))]
	else:
		ucb_algorithm = UCBalgorithm(num_arms)





	rewards = []
	mean_rewards = []
	instantaneous_regrets = []
	probabilities = []


	per_algorithm_regrets = [[] for _ in range(len(confidence_radii))]

	arm_pulls = [0 for _ in range(num_arms)]
	confidence_radius_pulls = [0 for _ in range(len(confidence_radii))]

	modsel_infos = []

	for t in range(num_timesteps):
		print("Timestep {}".format(t))
		modsel_sample_idx = modsel_manager.sample_base_index()
		probabilities.append(modsel_manager.get_distribution())
		confidence_radius_pulls[modsel_sample_idx] += 1
		confidence_radius = confidence_radii[modsel_sample_idx]
		print("Selected confidence radius {}".format(confidence_radius))
		if split:
			ucb_algorithm = ucb_algorithms[modsel_sample_idx]


		play_arm_index = ucb_algorithm.get_ucb_arm(confidence_radius)
		arm_pulls[play_arm_index] += 1
		reward = bandit.get_reward(play_arm_index)
		rewards.append(reward)

		modsel_info = dict([])
		#modsel_info["optimistic_reward_predictions"] = ucb_arm_value
		#modsel_info["pessimistic_reward_predictions"] = lcb_arm_value

		modsel_infos.append(modsel_info)

		ucb_algorithm.update_arm_statistics(play_arm_index, reward)
		
		mean_reward = bandit.get_arm_mean(play_arm_index)

		mean_rewards.append(mean_reward)

		##### NEED TO ADD THE UPDATE TO THE MODEL SEL ALGO

		modsel_manager.update_distribution(modsel_sample_idx, reward, modsel_info )



		instantaneous_regret = bandit.get_max_mean() - bandit.get_arm_mean(play_arm_index)
		instantaneous_regrets.append(instantaneous_regret)

		per_algorithm_regrets[modsel_sample_idx].append(instantaneous_regret)

	return rewards, mean_rewards, instantaneous_regrets, arm_pulls, confidence_radius_pulls, probabilities, per_algorithm_regrets, modsel_infos



def test_MAB_modsel_split(rewards_regrets_all, num_timesteps,  confidence_radius,
	modselalgo = "Corral"):
	
	num_base_learners = len(rewards_regrets_all)


	modsel_manager = get_modsel_manager(modselalgo,num_base_learners, num_timesteps )


	if algotype == "bernoulli":

		bandit = BernoulliBandit(means, scalings)
	elif algotype == "gaussian":
		bandit = GaussianBandit(means, stds)
	else:
		raise ValueError("unrecognized bandit type {}".format(algotype))

	num_arms = len(means)
	#empirical_means = [0 for _ in range(num_arms)]




	# play_arm_index = random.choice(range(num_arms))

	#ucb_algorithms = [UCBalgorithm(num_arms) for _ in range(num_base_learners)]

	

	rewards = []
	mean_rewards = []
	instantaneous_regrets = []
	probabilities = []


	#rewards_regrets_all

	base_learners_index = [0 for _ in range(num_base_learners)]

	#per_algorithm_regrets = [[] for _ in range(num_base_learners)]
	#arm_pulls = [0 for _ in range(num_arms)]
	#confidence_radius_pulls = [0 for _ in range(num_base_learners)]

	for t in range(num_timesteps):
		print("Timestep {}".format(t))
		modsel_sample_idx = modsel_manager.sample_base_index()
		probabilities.append(modsel_manager.get_distribution())

		
		#confidence_radius = confidence_radii[modsel_sample_idx]
		print("Selected confidence radius {}".format(confidence_radius))
		

		# play_arm_index, ucb_arm_value, lcb_arm_value = ucb_algorithm.get_ucb_arm(confidence_radius)
		# arm_pulls[play_arm_index] += 1
		

		reward = rewards_regrets_all[modsel_sample_idx][0][base_learners_index[modsel_sample_idx]]
		rewards.append(reward)

		#modsel_info = dict([])
		#modsel_info["optimistic_reward_predictions"] = ucb_arm_value
		#modsel_info["pessimistic_reward_predictions"] = lcb_arm_value

		#ucb_algorithm.update_arm_statistics(play_arm_index, reward)
		
		#mean_reward = bandit.get_arm_mean(play_arm_index)

		#mean_rewards.append(mean_reward)

		##### NEED TO ADD THE UPDATE TO THE MODEL SEL ALGO


		modsel_info = rewards_regrets_all[modsel_sample_idx][2][base_learners_index[modsel_sample_idx]]


		modsel_manager.update_distribution(modsel_sample_idx, reward, modsel_info )



		instantaneous_regret = rewards_regrets_all[modsel_sample_idx][1][base_learners_index[modsel_sample_idx]]
		instantaneous_regrets.append(instantaneous_regret)

		base_learners_index[modsel_sample_idx] += 1

		#per_algorithm_regrets[modsel_sample_idx].append(instantaneous_regret)

	return rewards, instantaneous_regrets, base_learners_index, probabilities






if __name__ == "__main__":

	num_timesteps = int(sys.argv[1])
	exp_type = str(sys.argv[2])
	num_experiments = int(sys.argv[3]) ### In this experiment this variable corresponds to the number of confidence radii



	std_multiplier = 2.0/np.sqrt(num_experiments)

	normalize = False
	#IPython.embed()
	if exp_type == "exp1":
		means = [.7, .8]
		stds = []
		scalings = []
		confidence_radius = 1 ## increase radii
		algotype = "bernoulli"
		experiment_name = "exp1"
		experiment_tag = r'$\mathrm{UCB}$'

	elif exp_type == "exp2":
		means = [.7, .8, .2, .9]
		stds = []
		scalings = []
		confidence_radius = .3 ## increase radii
		algotype = "bernoulli"
		experiment_name = "exp2"
		experiment_tag = r'$\mathrm{UCB}$'


	elif exp_type == "exp3":
		means = [.7, .9]
		stds = [.1, .1]
		scalings = []
		confidence_radius = 2 ## increase radii
		algotype = "gaussian"
		experiment_name = "exp3"
		experiment_tag = r'$\mathrm{UCB}$'


	elif exp_type == "exp4":
		means = [.7, .9]
		stds = [1, 1]
		scalings = []
		confidence_radius = 2 ## increase radii
		algotype = "gaussian"
		experiment_name = "exp4"
		experiment_tag = r'$\mathrm{UCB}$'

	elif exp_type == "exp5":
		means = [.7, .9]
		stds = [3, 3]
		scalings = []
		confidence_radius = 2 ## increase radii
		algotype = "gaussian"
		experiment_name = "exp5"
		experiment_tag = r'$\mathrm{UCB}$'


	elif exp_type == "exp6":
		means = [.7, .9]
		stds = [6, 6]
		scalings = []
		confidence_radius = 2 ## increase radii
		algotype = "gaussian"
		experiment_name = "exp6"
		experiment_tag = r'$\mathrm{UCB}$'

	elif exp_type == "exp7":
		means = [.5, 1, .2, .1, .6]
		stds = [6, 6, 6, 6, 6]
		scalings = []
		confidence_radius = 2 ## increase radii
		algotype = "gaussian"
		experiment_name = "exp7"
		experiment_tag = r'$\mathrm{UCB}$'

	elif exp_type == "exp7selfmodsel":
		means = [.5, 1, .2, .1, .6]
		stds = [1, 1, 1, 1, 1]
		scalings = []
		confidence_radius = 0 ## increase radii
		algotype = "gaussian"
		experiment_name = "exp7selfmodsel"
		experiment_tag = "Experiment 1" #r'$\mathrm{UCB}$'


	elif exp_type == "exp8":
		means = [.5, 1, .2, .1, .6]
		stds = [6, 6, 6, 6, 6]
		scalings = []
		confidence_radius = 3 ## increase radii
		algotype = "gaussian"
		experiment_name = "exp8"
		experiment_tag = r'$\mathrm{UCB}$'


	else:
		raise ValueError("experiment type not recognized {}".format(exp_type))

	confidence_radii = [confidence_radius]*num_experiments



	exp_data_dir = "./paper_self_modsel/{}".format(experiment_name)
	exp_info = "means - {} \n conf_radii - {}".format(means, confidence_radii)

	if not os.path.exists(exp_data_dir):
		os.mkdir(exp_data_dir)

	exp_data_dir_T = "{}/T{}".format(exp_data_dir, num_timesteps)
	if not os.path.exists(exp_data_dir_T):
		os.mkdir(exp_data_dir_T)

	# per_experiment_data = "{}/detailed".format(exp_data_dir_T)
	# if not os.path.exists(per_experiment_data):
	# 	os.mkdir(per_experiment_data)

	with open('{}/info.txt'.format(exp_data_dir), 'w') as f:
	    f.write(exp_info)


	colors = ["red", "orange", "violet", "black", "brown", "yellow", "green", "gray"]	
	#modselalgos = ["DoublingDataDriven", "EstimatingDataDriven"]#["UCB", 'BalancingSharp',  "EXP3", "Corral", 'BalancingDoResurrectClassic','BalancingDoResurrectDown', "BalancingDoResurrect"]#"BalancingDoubling"]# "BalancingDoubling"]#"BalancingDoResurrect", "BalancingSharp", "UCB", "EXP3", "Corral" ]

	modselalgos_lists = [["DoublingDataDriven", "EstimatingDataDriven"]]#, "Corral"]]

	normalization_visualization = 1.0/np.sqrt( np.arange(num_timesteps) + 1)
	normalization_visualization *= 1.0/np.log( np.arange(num_timesteps) + 2)




	individual_lines_colors = "black"
	mean_line_color = "blue"
	modsel_color = "red"
	#### RUN THE BASELINES
	

	data_to_save = []


	cum_regrets_all = []	
	rewards_regrets_all = []		

	baselines_results = []
	counter = 0



	for confidence_radius in confidence_radii:
			#confidence_radius_pulls_all = []
			

			rewards, mean_rewards, instantaneous_regrets, arm_pulls,_, _, _, modsel_infos = test_MAB_modsel(means, stds, scalings, num_timesteps, 
					[confidence_radius],  modselalgo = "Corral", algotype = algotype) ### Here we can use any modselalgo, it is dummy in this case.


			instance_cum_regrets = np.cumsum(instantaneous_regrets)

			cum_regrets_all.append(instance_cum_regrets)
			rewards_regrets_all.append((rewards, instantaneous_regrets, modsel_infos))

			# mean_cum_regrets = np.mean(cum_regrets_all,0)
			# std_cum_regrets = np.std(cum_regrets_all,0)

			# mean_cum_regrets *= normalization_visualization
			# std_cum_regrets *= normalization_visualization

			baselines_results.append(np.cumsum(instantaneous_regrets))

			#plt.plot(np.arange(num_timesteps) + 1, mean_cum_regrets, label = "radius {}".format(confidence_radius), color = color )
			#plt.fill_between(np.arange(num_timesteps) + 1,mean_cum_regrets - .5*std_cum_regrets,mean_cum_regrets + .5*std_cum_regrets, alpha = .2 , color = color )

			instance_plot_reg = instance_cum_regrets
			
			data_to_save.append(("run {}".format(counter), instance_cum_regrets))

			if normalize:
				instance_plot_reg *= normalization_visualization




			#### PLOT
			if counter == 0:
				plt.plot(np.arange(num_timesteps) + 1, instance_plot_reg, color = individual_lines_colors, linewidth = .5, label = "Instance Regret")
			else:
				plt.plot(np.arange(num_timesteps) + 1, instance_plot_reg, color = individual_lines_colors, linewidth = .5)

			counter  += 1


	### PLOT the mean reward with confidence interval
	mean_cum_regrets = np.mean(cum_regrets_all, 0)
	std_cum_regrets = np.std(cum_regrets_all, 0)

	data_to_save.append(("mean cum_regret", mean_cum_regrets  ))

	data_to_save.append(("std cum regret", std_cum_regrets))


	pickle_and_zip(data_to_save, "{}_T{}".format(experiment_name,num_timesteps), exp_data_dir_T, is_zip_file = True)

	if normalize:
		mean_cum_regrets *= normalization_visualization
		std_cum_regrets *= normalization_visualization


	plt.plot(np.arange(num_timesteps) + 1, mean_cum_regrets, color = mean_line_color  , linewidth = 3, linestyle = "--", label = "Expected Regret")
	plt.fill_between(np.arange(num_timesteps)+1, mean_cum_regrets - std_multiplier*std_cum_regrets, mean_cum_regrets + std_multiplier*std_cum_regrets, alpha = .2, color = mean_line_color)
	
	plt.title("{} Base Learners".format(experiment_tag), fontsize = 18)

	plt.legend(loc="upper left", fontsize = 13)
	plt.xlabel("Rounds", fontsize =13)
	if normalize:
		plt.ylabel("Regret Rate", fontsize =13)
		plt.savefig("{}/norm_{}_T{}.pdf".format(exp_data_dir_T, experiment_name,num_timesteps))

	else:
		plt.ylabel("Cumulative Regret", fontsize =13)
		plt.savefig("{}/{}_T{}.pdf".format(exp_data_dir_T, experiment_name,num_timesteps))
			
	plt.close("all")


	### USE the same realizations as above to run our model selection algorithms.
	for modselalgos in modselalgos_lists:	
		modselalgos_name = ""

		for modselalgo, i in zip(modselalgos, range(len(modselalgos))):
				modselalgos_name += modselalgo
				modsel_cum_regrets_all = []	
				modsel_confidence_radius_pulls_all = []
				probabilities_all = []
				#per_algorithm_regrets_stats = []
				for _ in range(num_experiments):
					modsel_rewards, modsel_instantaneous_regrets, modsel_base_learners_index, probabilities_modsel = test_MAB_modsel_split(rewards_regrets_all, num_timesteps,  confidence_radius,
						modselalgo = modselalgo)
					
					modsel_cum_regrets_all.append(np.cumsum(modsel_instantaneous_regrets))
					modsel_confidence_radius_pulls_all.append(modsel_base_learners_index)
					#probabilities_all.append(probabilities_modsel)
					#per_algorithm_regrets_stats.append(per_algorithm_regrets)



				mean_modsel_cum_regrets = np.mean(modsel_cum_regrets_all,0)
				std_modsel_cum_regrets = np.std(modsel_cum_regrets_all,0)




				#### Normalized Visualization
				if normalize:
					mean_modsel_cum_regrets *= normalization_visualization
					std_modsel_cum_regrets *= normalization_visualization

			

				if modselalgo == "EstimatingDataDriven":
					modselalgo_tag = r'$\mathrm{E}\mathrm{D}^2\mathrm{RB}$'
				elif modselalgo == "DoublingDataDriven":
					modselalgo_tag = r'$\mathrm{D}^3\mathrm{RB}$'
				else:
					modselalgo_tag = modselalgo


				#IPython.embed()
				plt.plot(np.arange(num_timesteps) + 1, mean_modsel_cum_regrets, label = modselalgo_tag, color = colors[i], linewidth = 3, linestyle = "--")

				plt.fill_between(np.arange(num_timesteps) + 1,mean_modsel_cum_regrets -std_multiplier*std_modsel_cum_regrets, mean_modsel_cum_regrets +std_multiplier*std_modsel_cum_regrets, 
					color = colors[i], alpha = .2   )

				#mean_modsel_confidence_radius_pulls = np.mean( modsel_confidence_radius_pulls_all, 0)


		counter = 0
		#IPython.embed()
		for confidence_radius, baseline_result in zip(confidence_radii, baselines_results):
			#mean_cum_regrets, std_cum_regrets = baseline_result_tuple

			baseline_result_plot = baseline_result				
			if normalize:
				baseline_result_plot =  baseline_result*normalization_visualization
			


			if counter == 0:
				plt.plot(np.arange(num_timesteps) + 1, baseline_result_plot, color = individual_lines_colors, linewidth = .5, label = "Instance Regret" )
			else:
				plt.plot(np.arange(num_timesteps) + 1, baseline_result_plot, color = individual_lines_colors, linewidth = .5)

			counter += 1


		plt.title("{}".format(experiment_tag), fontsize = 18)
		plt.legend(loc="upper left", fontsize = 12)
		plt.xlabel("Rounds", fontsize =13)
		#plt.ylim(bottom = 0, top = 500)
		if normalize:
			plt.ylabel("Regret Rate", fontsize = 13)
			plt.savefig("{}/norm_{}_{}_T{}.pdf".format(exp_data_dir_T,experiment_name, modselalgos_name,num_timesteps))
		else:
			plt.ylabel("Cumulative Regret", fontsize = 13)
			plt.savefig("{}/{}_{}_T{}.pdf".format(exp_data_dir_T,experiment_name, modselalgos_name,num_timesteps))

		plt.close("all")


			

				

