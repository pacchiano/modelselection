import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

import IPython


class LinearBandit:
	def __init__(self, theta_star, arm_set = "sphere", std_scaling = 1):
		self.theta_star = theta_star
		self.std_scaling = std_scaling

		self.arm_set = arm_set
			
		if arm_set == "sphere":
			self.max_mean = np.linalg.norm(self.theta_star)

		elif arm_set == "hypercube":
			self.max_mean = np.sum(np.abs(self.theta_star))/np.sqrt(len(theta_star))

		else:
			raise ValueError("Arm set {} not recognized for linear bandit".format(arm_set))


	def get_reward(self, arm_vector):
		mean_reward = self.get_arm_mean( arm_vector)
		return mean_reward + np.random.normal(self.std_scaling)


	def get_max_mean(self):
		return self.max_mean


	def get_arm_mean(self, arm_vector):
		mean_reward = np.dot(arm_vector, self.theta_star)
		return mean_reward		

	def get_context(self):
		return []



class SphereContextDistribution:
	def __init__(self, dimension, context_size):
		self.dimension = dimension
		self.context_size = context_size
		
	def sample_context(self):
		contexts = []
		for i in range(self.context_size):
			context = np.random.normal(0,1,self.dimension)
			context /= np.linalg.norm(context)
			contexts.append(context)
		return contexts


class LinearContextualBandit:
	def __init__(self, theta_star, context_distribution, 
		context_size = 10, std_scaling = 1, max_reward_estimation_samples = 100000):
		
		self.theta_star = theta_star
		self.std_scaling = std_scaling
		#self.arm_set = arm_set
		self.context_distribution = context_distribution
		self.dimension = len(self.theta_star)

		self.context_size = context_size

		self.max_mean = self.compute_max_mean(max_reward_estimation_samples)


	def compute_max_mean(self, num_samples):
		sum_max = 0
		for i in range(num_samples):
			context = self.context_distribution.sample_context()
			sum_max += max([np.dot(arm, self.theta_star) for arm in context ])
		return sum_max / num_samples


	def get_context(self):
		contexts = self.context_distribution.sample_context()
		return contexts

	def get_reward(self, arm_vector):
		mean_reward = self.get_arm_mean( arm_vector)
		return mean_reward + np.random.normal(self.std_scaling)


	def get_max_mean(self):


		return self.max_mean


	def get_arm_mean(self, arm_vector):
		mean_reward = np.dot(arm_vector, self.theta_star)
		return mean_reward		




class BernoulliBandit:
	def __init__(self, base_means, scalings = []):
		self.base_means = base_means
		
		self.num_arms = len(base_means)
		
		if len(scalings) == 0:
			self.scalings = [1 for _ in range(self.num_arms)]
		else:
			self.scalings = scalings

		self.means = [self.base_means[i]*self.scalings[i] for i in range(self.num_arms)]
		self.max_mean = max(self.means)

	def get_reward(self, arm_index):
		if arm_index >= self.num_arms or arm_index < 0:
			raise ValueError("Invalid arm index {}".format(arm_index))

		random_uniform_sample = random.random()
		if random_uniform_sample <= self.base_means[arm_index]:
			return 1*self.scalings[arm_index]
		else:
			return 0

	def get_max_mean(self):
		return self.max_mean

	def get_arm_mean(self, arm_index):
		return self.means[arm_index]

	def get_context(self):
		return []


class GaussianBandit:
	def __init__(self, means, stds):
		self.means = means
		self.stds = stds
		self.num_arms = len(means)
		self.max_mean = max(self.means)


	def get_reward(self, arm_index):
		if arm_index >= self.num_arms or arm_index < 0:
			raise ValueError("Invalid arm index {}".format(arm_index))


		return np.random.normal(self.means[arm_index], self.stds[arm_index])


	def get_max_mean(self):
		return self.max_mean

	def get_arm_mean(self, arm_index):
		return self.means[arm_index]

	def get_context(self):
		return []
