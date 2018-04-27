from Event import *
import numpy as np

class Agent:

	def __init__(self, selfishness=1, altruism=1, tau=0.01):
		self.selfishness = selfishness
		self.altruism = altruism
		self.rationality = tau

	def GetChoiceProb(self, myevent, choice):
		utilities = [-myevent.agentvalue[i]*self.selfishness+myevent.agentbeliefs[i]*self.altruism for i in range(len(myevent.options))]
		utilities = [i - max(utilities) for i in utilities]
		values = [np.exp(i/self.rationality) for i in utilities]
		values = [i/sum(values) for i in values]
		return(values[choice])

	def SetParameters(self, selfishness, altruism):
		self.selfishness = selfishness
		self.altruism = altruism
