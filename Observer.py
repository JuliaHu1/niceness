import numpy as np

class Observer:

	def __init__(self, MyEvent, MyAgent):
		self.MyEvent = MyEvent
		self.MyAgent = MyAgent

	def ToM(self, ObservedChoice, samples=10000):
		Selfishness = [np.random.uniform(0,1) for i in range(samples)]
		Altruism = [np.random.uniform(0,1) for i in range(samples)]
		Probability = [0] * samples
		for x in range(samples):
			self.MyAgent.SetParameters(Selfishness[x], Altruism[x])
			Probability[x] = self.MyAgent.GetChoiceProb(self.MyEvent, ObservedChoice)
		# Normalize because his is Monte Carlo sampling
		Probability = [i/sum(Probability) for i in Probability]
		return([np.dot(Selfishness,Probability),np.dot(Altruism,Probability)])
