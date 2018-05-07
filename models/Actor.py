import numpy as np

class Actor:

	def __init__(self, rationality=0.0, altruism=0.0, selfishness=0.0):
		self.rationality = rationality
		self.altruism = altruism
		self.selfishness = selfishness

	def get_action_probability(self, state, action):
		U = [sum((self.altruism*state.actor_beliefs[i])-(self.selfishness*state.actor_rewards[i])) for i in range(len(state.actions))]
		U = [utility-np.max(U) for utility in U]
		action_probabilities = [np.exp(utility/self.rationality) for utility in U]
		action_probabilities = [i/sum(action_probabilities) for i in action_probabilities]
		return action_probabilities[action]
