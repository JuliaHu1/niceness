from Event import *

import numpy as np
import sys
class Actor:

	def __init__(self):
		self.rationality = 0.0
		self.altruism = 0.0
		self.selfishness = 0.0

	def get_action_probability(self, event, action):
		# U = [-event.actor_rewards[i]*self.selfishness+event.actor_beliefs[i]*self.altruism for i in range(len(event.actions[action]))]
		U = [sum((event.actor_beliefs[i]*self.altruism)-(event.actor_rewards[i]*self.selfishness)) for i in range(len(event.actions))]
		U = [utility-np.max(U) for utility in U]
		action_probabilities = [np.exp(utility/self.rationality) for utility in U]
		action_probabilities = [i/sum(action_probabilities) for i in action_probabilities]
		return action_probabilities[action]

	def set_rationality(self, rationality):
		self.rationality = rationality

	def set_altruism(self, altruism):
		self.altruism = altruism

	def set_selfishness(self, selfishness):
		self.selfishness = selfishness

	
