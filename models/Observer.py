import numpy as np

class Observer:

	def __init__(self, event, actor):
		self.event = event
		self.actor = actor

	def ToM(self, action, samples=10000):
		altruism = [np.random.uniform(0, 1) for i in range(samples)]
		selfishness = [np.random.uniform(0, 1) for i in range(samples)]
		probability = np.zeros(samples)
		for i in range(samples):
			self.actor.set_rationality(0.01)
			self.actor.set_altruism(altruism[i])
			self.actor.set_selfishness(selfishness[i])
			probability[i] = self.actor.get_action_probability(self.event, action)
		# Normalize because his is Monte Carlo sampling
		probability = [p/sum(probability) for p in probability]
		t = [np.dot(altruism, probability), np.dot(selfishness, probability)]
		return t[0]/sum(t), t[1]/sum(t)
