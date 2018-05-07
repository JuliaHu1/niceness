import numpy as np

class Observer:

	def __init__(self, state, actor):
		self.state = state
		self.actor = actor

	def sacrifice(self, action):
		return len(action)

	def utilitarian(self, action, receiver_reward_values):
		action_rewards = np.array([receiver_reward_values[option] for option in action])
		return sum(action_rewards) / sum(self.state.receiver_rewards)

	def ToM(self, action, samples=10000):
		altruism = [np.random.uniform(0, 1) for i in range(samples)]
		selfishness = [np.random.uniform(0, 1) for i in range(samples)]
		probability = np.zeros(samples)
		for i in range(samples):
			self.actor.rationality = 0.01
			self.actor.altruism = altruism[i]
			self.actor.selfishness = selfishness[i]
			probability[i] = self.actor.get_action_probability(self.state, action)
		# Normalize because his is Monte Carlo sampling
		probability = [p/sum(probability) for p in probability]
		t = [np.dot(altruism, probability), np.dot(selfishness, probability)]
		return t[0]/sum(t), t[1]/sum(t)

	def judgment(self):
		pass