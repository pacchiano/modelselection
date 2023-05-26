import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

import IPython
from algorithmsmodsel import CorralHyperparam, EXP3Hyperparam, UCBHyperparam, BalancingHyperparamDoublingDataDriven, get_modsel_manager
from bandit_algs import UCBalgorithm

from bandit_envs import BernoulliBandit, GaussianBandit

from self_modsel import test_MAB_modsel

np.random.seed(1001)
random.seed(1001)



if __name__ == "__main__":

	num_timesteps = int(sys.argv[1])
	exp_type = str(sys.argv[2])
	num_experiments = int(sys.argv[3]) ### In this experiment this variable corresponds to the number of confidence radii

	dmin = 2

	normalize = True
	#IPython.embed()
	if exp_type == "exp1":
		means = [.7, .8]
		stds = []
		scalings = []
		confidence_radius = 1 ## increase radii
		algotype = "bernoulli"
		experiment_name = "exp1"
		experiment_tag = "MAB"

	elif exp_type == "exp2":
		means = [.7, .8, .2, .9]
		stds = []
		scalings = []
		confidence_radius = .3 ## increase radii
		algotype = "bernoulli"
		experiment_name = "exp2"
		experiment_tag = "MAB"


	elif exp_type == "exp3":
		means = [.7, .9]
		stds = [.1, .1]
		scalings = []
		confidence_radius = 2 ## increase radii
		algotype = "gaussian"
		experiment_name = "exp3"
		experiment_tag = "MAB"


	elif exp_type == "exp4":
		means = [.7, .9]
		stds = [1, 1]
		scalings = []
		confidence_radius = 2 ## increase radii
		algotype = "gaussian"
		experiment_name = "exp4"
		experiment_tag = "MAB"

	elif exp_type == "exp5":
		means = [.7, .9]
		stds = [3, 3]
		scalings = []
		confidence_radius = 2 ## increase radii
		algotype = "gaussian"
		experiment_name = "exp5"
		experiment_tag = "MAB"


	elif exp_type == "exp6":
		means = [.7, .9]
		stds = [6, 6]
		scalings = []
		confidence_radius = 2 ## increase radii
		algotype = "gaussian"
		experiment_name = "exp6"
		experiment_tag = "MAB"

	elif exp_type == "exp7":
		means = [.5, 1, .2, .1, .6]
		stds = [6, 6, 6, 6, 6]
		scalings = []
		confidence_radius = 2 ## increase radii
		algotype = "gaussian"
		experiment_name = "exp7"
		experiment_tag = "MAB"

	elif exp_type == "exp8":
		means = [.5, 1, .2, .1, .6]
		stds = [6, 6, 6, 6, 6]
		scalings = []
		confidence_radius = 3 ## increase radii
		algotype = "gaussian"
		experiment_name = "exp8"
		experiment_tag = "MAB"


	else:
		raise ValueError("experiment type not recognized {}".format(exp_type))

	confidence_radii = [confidence_radius]*num_experiments



	exp_data_dir = "./supporting_plots/{}".format(experiment_name)
	if not os.path.exists(exp_data_dir):
		os.mkdir(exp_data_dir)

	exp_data_dir_T = "{}/T{}".format(exp_data_dir, num_timesteps)
	if not os.path.exists(exp_data_dir_T):
		os.mkdir(exp_data_dir_T)


	pure_doubling_color = "red"
	adaptive_regret_color = "orange"


	#colors = ["red", "orange", "violet", "black", "brown", "yellow", "green", "gray"]	
	#modselalgos = ["DoublingDataDriven", "EstimatingDataDriven"]#["UCB", 'BalancingSharp',  "EXP3", "Corral", 'BalancingDoResurrectClassic','BalancingDoResurrectDown', "BalancingDoResurrect"]#"BalancingDoubling"]# "BalancingDoubling"]#"BalancingDoResurrect", "BalancingSharp", "UCB", "EXP3", "Corral" ]

	#modselalgos_lists = [["DoublingDataDriven", "EstimatingDataDriven", "Corral"]]

	normalization_visualization = 1.0/np.sqrt( np.arange(num_timesteps) + 1)
	#normalization_visualization *= 1.0/np.log( np.arange(num_timesteps) + 2)




	individual_lines_colors = "black"
	mean_line_color = "blue"
	#modsel_color = "red"
	#### RUN THE BASELINES
	


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
			

			doubling_regrets = []
			adaptive_regrets = []
			doubling_coeff = dmin
			adaptive_coeff = dmin
			for cum_reg, i in zip(instance_plot_reg, range(len(instance_plot_reg))):
				doubling_coeff = max(doubling_coeff, cum_reg/np.sqrt(i+1))
				doubling_regrets.append(doubling_coeff*np.sqrt(i+1))

				adaptive_coeff = max(dmin, cum_reg/np.sqrt(i+1))
				adaptive_regrets.append(adaptive_coeff*np.sqrt(i+1))

				print("Coefficients ", doubling_coeff, " ", adaptive_coeff)

			doubling_regrets = np.array(doubling_regrets)
			adaptive_regrets = np.array(adaptive_regrets)



			if normalize:
				instance_plot_reg *= normalization_visualization
				doubling_regrets *= normalization_visualization
				adaptive_regrets *= normalization_visualization



			#### PLOT
			if counter == 0:
				plt.plot(np.arange(num_timesteps) + 1, instance_plot_reg, color = individual_lines_colors, linewidth = .5, label = "True Regret Coeff.")
				plt.plot(np.arange(num_timesteps) + 1, adaptive_regrets, color = adaptive_regret_color, linewidth = 3, linestyle = "--", label = "Regret Coeff.")
				plt.plot(np.arange(num_timesteps) + 1, doubling_regrets, color = pure_doubling_color, linewidth = 3, linestyle = "--",label = "Monotonic Regret Coeff.")

			else:
				plt.plot(np.arange(num_timesteps) + 1, instance_plot_reg, color = individual_lines_colors, linewidth = .5)
				plt.plot(np.arange(num_timesteps) + 1, doubling_regrets, color = pure_doubling_color, linewidth = .5)
				plt.plot(np.arange(num_timesteps) + 1, adaptive_regrets, color = adaptive_regret_color, linewidth = .5)

			counter  += 1

	### PLOT the mean reward with confidence interval
	mean_cum_regrets = np.mean(cum_regrets_all, 0)
	std_cum_regrets = np.std(cum_regrets_all, 0)
	if normalize:
		mean_cum_regrets *= normalization_visualization
		std_cum_regrets *= normalization_visualization


	# plt.plot(np.arange(num_timesteps) + 1, mean_cum_regrets, color = mean_line_color  , linewidth = 3, linestyle = "--", label = "Expected Regret")
	# plt.fill_between(np.arange(num_timesteps)+1, mean_cum_regrets - .5*std_cum_regrets, mean_cum_regrets + .5*std_cum_regrets, alpha = .2, color = mean_line_color)
	

	plt.legend(loc="lower right", fontsize = 13)
	plt.xlabel("Rounds", fontsize =13)
	if normalize:
		plt.title("Regret Coefficients Evolution", fontsize = 18)

		plt.ylabel("Regret Coefficients", fontsize =13)
		plt.savefig("{}/norm_{}_T{}.pdf".format(exp_data_dir_T, experiment_name,num_timesteps))

	else:
		plt.title("Regret Functions Evolution", fontsize = 18)

		plt.ylabel("Regret Functions", fontsize =13)
		plt.savefig("{}/{}_T{}.pdf".format(exp_data_dir_T, experiment_name,num_timesteps))
			
	plt.close("all")




			

				

