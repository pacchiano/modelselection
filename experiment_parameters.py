import numpy as np

from algorithmsmodsel import CorralHyperparam, EXP3Hyperparam, UCBHyperparam, BalancingHyperparamDoublingDataDriven, get_modsel_manager
from bandit_algs import UCBalgorithm, LUCBalgorithm, EXP3

from bandit_envs import BernoulliBandit, GaussianBandit, LinearBandit, LinearContextualBandit, SphereContextDistribution
from utilities import pickle_and_zip, unzip_and_load_pickle, produce_parallelism_schedule, write_dictionary_file




def get_experiment_info(exp_type):




	if exp_type == "experiment1":
		means = [.5, 1, .2, .1, .6]
		#stds = [6, 6, 6, 6, 6]

		stds = [1, 1, 1, 1, 1]
		scalings = []
		#confidence_radii = [2]*10 ## increase radii
		confidence_radii = [0]*10 ## increase radii

		experiment_name = "experiment1"
		bandit = GaussianBandit(means, stds)
		num_arms = len(means)

		def get_base_algorithms(parameters):
			base_algorithms = [UCBalgorithm(num_arms) for _ in range(len(parameters))]
			return base_algorithms

		exp_info = "means - {} \n conf_radii - {}".format(means, confidence_radii)
		arm_set_type = "MAB"
		experiment_tag = "Experiment 1"#r'$\mathrm{UCB}$'
		parameters = confidence_radii
		plot_parameter_name = "Base c="


	elif exp_type == "experiment2":
		means = [.5, 1, .2, .1, .6]
		#stds = [6, 6, 6, 6, 6]
		stds = [1, 1, 1, 1, 1]

		scalings = []
		#confidence_radii = [0, 4, 6, 20] ## increase radii
		confidence_radii = [0, 4, 6, 20] ## increase radii
		experiment_name = "experiment2"
		bandit = GaussianBandit(means, stds)
		num_arms = len(means)

		def get_base_algorithms(parameters):
			base_algorithms = [UCBalgorithm(num_arms) for _ in range(len(parameters))]
			return base_algorithms

		exp_info = "means - {} \n conf_radii - {}".format(means, confidence_radii)
		arm_set_type = "MAB"
		experiment_tag = "Experiment 2"#r'$\mathrm{UCB}$'
		parameters = confidence_radii
		plot_parameter_name = "Base c="




	elif exp_type == "experiment3":
		dimension = 10
		theta_star = np.arange(dimension)/np.linalg.norm(np.arange(dimension))*5
		#confidence_radii = [.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		
		confidence_radii = [0, .16, 2.5, 5, 25	] ## increase radii
		experiment_name = "experiment3"
		bandit = LinearBandit(theta_star, arm_set = "sphere", std_scaling = 1)
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(dimension, dimension) for _ in range(len(parameters))]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, confidence_radii)
		arm_set_type = "sphere"
		experiment_tag = "Experiment 3"#r'$\mathrm{LinTS}$'
		parameters = confidence_radii
		plot_parameter_name = "Base c="



	elif exp_type == "experiment4":
		dimension = 10
		theta_star = np.arange(dimension)
		confidence_radii = [0, .16, 2.5, 5, 25	]#[.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		experiment_name = "experiment4"
		context_size = 10
		context_distribution = SphereContextDistribution(dimension, context_size)
		bandit = LinearContextualBandit(theta_star, context_distribution, context_size = context_size)
		#IPython.embed()
		#raise ValueError("asdlfkm")
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(dimension, dimension) for _ in range(len(parameters))]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, confidence_radii)
		arm_set_type = "contextual"
		experiment_tag = "Experiment 4"# r'$\mathrm{LinTS}$ Contextual'
		parameters = confidence_radii
		plot_parameter_name = "Base c="


	elif exp_type == "experiment5":
		max_dimension = 15
		dstar = 5
		theta_star = np.arange(max_dimension)#/np.linalg.norm(np.arange(dstar))
		theta_star[dstar:] = 0
		confidence_radii = [2] ## increase radii
		experiment_name = "experiment5"
		bandit = LinearBandit(theta_star, arm_set = "sphere", std_scaling = 1)
		dimensions = [2, 5, 10, 15]
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(d, max_dimension) for d in parameters]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, max_dimension)
		arm_set_type = "sphere"
		experiment_tag = "Experiment 5"#r'$\mathrm{LinTS}$'
		parameters = dimensions
		plot_parameter_name = "Base d="




	elif exp_type == "experiment6":
		max_dimension = 15
		dstar = 5
		theta_star = np.arange(max_dimension)/np.linalg.norm(np.arange(dstar))
		theta_star[dstar:] = 0
		confidence_radii = [2] ## increase radii
		experiment_name = "experiment6"
		context_size = 10
		context_distribution = SphereContextDistribution(max_dimension, context_size)
		bandit = LinearContextualBandit(theta_star, context_distribution, context_size = context_size)
		dimensions = [2, 5, 10, 15]

		#IPython.embed()
		#raise ValueError("asdlfkm")
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(d, max_dimension) for d in parameters]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, confidence_radii)
		arm_set_type = "contextual"
		experiment_tag = "Experiment 6"#r'$\mathrm{LinTS}$ Contextual'
		parameters = dimensions
		plot_parameter_name = "Base d="





	### Generic multiple 
	elif exp_type == "experiment7":
		means = [.1, .2, .5, .8]
		stds = []
		scalings = []
		confidence_radii = [0, .08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		experiment_name = "experiment7"
		bandit = BernoulliBandit(means, scalings)
		num_arms = len(means)

		def get_base_algorithms(parameters):
			base_algorithms = [UCBalgorithm(num_arms) for _ in range(len(parameters))]
			return base_algorithms

		exp_info = "means - {} \n conf_radii - {}".format(means, confidence_radii)
		arm_set_type = "MAB"
		experiment_tag = "Experiment A"#r'$\mathrm{UCB}$'
		parameters = confidence_radii
		plot_parameter_name = "Base c="


	elif exp_type == "experiment8":
		means = [.1, .2]
		stds = []
		scalings = [30, 30]
		confidence_radii = [ 1 ]*10 ## increase radii
		experiment_name = "experiment8"
		bandit = BernoulliBandit(means, scalings)
		num_arms = len(means)

		def get_base_algorithms(parameters):
			base_algorithms = [UCBalgorithm(num_arms) for _ in range(len(parameters))]
			return base_algorithms

		exp_info = "means - {} \n conf_radii - {}".format(means, confidence_radii)
		arm_set_type = "MAB"
		experiment_tag = "Experiment B"#r'$\mathrm{UCB}$'
		parameters = confidence_radii
		plot_parameter_name = "Base c="



	elif exp_type == "experiment9":
		dimension = 5
		theta_star = np.arange(dimension)/np.linalg.norm(np.arange(dimension))*5
		#confidence_radii = [.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		
		confidence_radii = [0, .16, 2.5, 5, 25	] ## increase radii
		experiment_name = "experiment9"
		bandit = LinearBandit(theta_star, arm_set = "hypercube", std_scaling = 1)
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(dimension, dimension) for _ in range(len(parameters))]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, confidence_radii)
		arm_set_type = "hypercube"
		experiment_tag = "Experiment C"#r'$\mathrm{LinTS}$'
		parameters = confidence_radii
		plot_parameter_name = "Base c="


	elif exp_type == "experiment10":
		dimension = 10
		theta_star = np.arange(dimension)/np.linalg.norm(np.arange(dimension))*5
		#confidence_radii = [.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		
		confidence_radii = [0, .16, 2.5, 5, 25	] ## increase radii
		experiment_name = "experiment10"
		bandit = LinearBandit(theta_star, arm_set = "hypercube", std_scaling = 1)
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(dimension, dimension) for _ in range(len(parameters))]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, confidence_radii)
		arm_set_type = "hypercube"
		experiment_tag = "Experiment D"#r'$\mathrm{LinTS}$'
		parameters = confidence_radii
		plot_parameter_name = "Base c="



	elif exp_type == "experiment11":
		dimension = 100
		theta_star = np.arange(dimension)/np.linalg.norm(np.arange(dimension))*5
		#confidence_radii = [.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		
		confidence_radii = [0, .16, 2.5, 5, 25	] ## increase radii
		experiment_name = "experiment11"
		bandit = LinearBandit(theta_star, arm_set = "hypercube", std_scaling = 1)
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(dimension, dimension) for _ in range(len(parameters))]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, confidence_radii)
		arm_set_type = "hypercube"
		experiment_tag = "Experiment E"#r'$\mathrm{LinTS}$'
		parameters = confidence_radii
		plot_parameter_name = "Base c="





	elif exp_type == "experiment12":
		dimension = 5
		theta_star = np.arange(dimension)/np.linalg.norm(np.arange(dimension))*5
		#confidence_radii = [.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		
		confidence_radii = [0, .16, 2.5, 5, 25	] ## increase radii
		experiment_name = "experiment12"
		bandit = LinearBandit(theta_star, arm_set = "sphere", std_scaling = 1)
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(dimension, dimension) for _ in range(len(parameters))]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, confidence_radii)
		arm_set_type = "sphere"
		experiment_tag = "Experiment F"#r'$\mathrm{LinTS}$'
		parameters = confidence_radii
		plot_parameter_name = "Base c="


	elif exp_type == "experiment13":
		dimension = 100
		theta_star = np.arange(dimension)/np.linalg.norm(np.arange(dimension))*5
		#confidence_radii = [.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		
		confidence_radii = [0, .16, 2.5, 5, 25	] ## increase radii
		experiment_name = "experiment13"
		bandit = LinearBandit(theta_star, arm_set = "sphere", std_scaling = 1)
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(dimension, dimension) for _ in range(len(parameters))]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, confidence_radii)
		arm_set_type = "sphere"
		experiment_tag = "Experiment G"#r'$\mathrm{LinTS}$'
		parameters = confidence_radii
		plot_parameter_name = "Base c="





	elif exp_type == "experiment14":
		dimension = 5
		theta_star = np.arange(dimension)/np.linalg.norm(np.arange(dimension))*5
		confidence_radii = [0, .16, 2.5, 5, 25	]#[.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		experiment_name = "experiment14"
		context_size = 10
		context_distribution = SphereContextDistribution(dimension, context_size)
		bandit = LinearContextualBandit(theta_star, context_distribution, context_size = context_size)
		#IPython.embed()
		#raise ValueError("asdlfkm")
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(dimension, dimension) for _ in range(len(parameters))]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, confidence_radii)
		arm_set_type = "contextual"
		experiment_tag = "Experiment H"# r'$\mathrm{LinTS}$ Contextual'
		parameters = confidence_radii
		plot_parameter_name = "Base c="



	elif exp_type == "experiment15":
		dimension = 100
		theta_star = np.arange(dimension)/np.linalg.norm(np.arange(dimension))*5
		confidence_radii = [0, .16, 2.5, 5, 25	]#[.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		experiment_name = "experiment15"
		context_size = 10
		context_distribution = SphereContextDistribution(dimension, context_size)
		bandit = LinearContextualBandit(theta_star, context_distribution, context_size = context_size)
		#IPython.embed()
		#raise ValueError("asdlfkm")
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(dimension, dimension) for _ in range(len(parameters))]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, confidence_radii)
		arm_set_type = "contextual"
		experiment_tag = "Experiment I"# r'$\mathrm{LinTS}$ Contextual'
		parameters = confidence_radii
		plot_parameter_name = "Base c="


	elif exp_type == "experiment16":
		max_dimension = 100
		dstar = 30
		theta_star = np.arange(max_dimension)/np.linalg.norm(np.arange(dstar))*5
		theta_star[dstar:] = 0
		confidence_radii = [2] ## increase radii
		experiment_name = "experiment16"
		bandit = LinearBandit(theta_star, arm_set = "sphere", std_scaling = 1)
		dimensions = [10, 30, 50, 100]
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(d, max_dimension) for d in parameters]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, max_dimension)
		arm_set_type = "sphere"
		experiment_tag = "Experiment J"#r'$\mathrm{LinTS}$'
		parameters = dimensions
		plot_parameter_name = "Base d="



	elif exp_type == "experiment17":

		max_dimension = 15
		dstar = 5
		theta_star = np.arange(max_dimension)#/np.linalg.norm(np.arange(dstar))
		theta_star[dstar:] = 0
		confidence_radii = [2] ## increase radii
		experiment_name = "experiment17"
		bandit = LinearBandit(theta_star, arm_set = "hypercube", std_scaling = 1)
		dimensions = [2, 5, 10, 15]
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(d, max_dimension) for d in parameters]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, max_dimension)
		arm_set_type = "hypercube"
		experiment_tag = "Experiment K"
		parameters = dimensions
		plot_parameter_name = "dimension"

	elif exp_type == "experiment18":

		max_dimension = 100
		dstar = 5
		theta_star = np.arange(max_dimension)/np.linalg.norm(np.arange(dstar))*5
		theta_star[dstar:] = 0
		confidence_radii = [2] ## increase radii
		experiment_name = "experiment18"
		bandit = LinearBandit(theta_star, arm_set = "hypercube", std_scaling = 1)
		dimensions = [10, 30, 50, 100]
		def get_base_algorithms(parameters):
			base_algorithms = [LUCBalgorithm(d, max_dimension) for d in parameters]
			return base_algorithms

		exp_info = "theta_star - {} \n conf_radii - {}".format(theta_star, max_dimension)
		arm_set_type = "hypercube"
		experiment_tag = "Experiment L"
		parameters = dimensions
		plot_parameter_name = "dimension"



	else:
		raise ValueError("experiment type not recognized")


	experiment_info = dict([])

	experiment_info["experiment_name"] = experiment_name
	experiment_info["bandit"] = bandit
	experiment_info["get_base_algorithms_func"] = get_base_algorithms
	experiment_info["exp_info"] = exp_info
	experiment_info["arm_set_type"] = arm_set_type
	experiment_info["experiment_tag"] = experiment_tag
	experiment_info["parameters"] = parameters
	experiment_info["plot_parameter_name"] = plot_parameter_name

	return experiment_info



