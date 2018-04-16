
# coding: utf-8

# In[106]:


## Minimal code for inverting a utility function
import numpy as np
import warnings; warnings.simplefilter('ignore')


# In[107]:


# A forward model (i.e. a mental model of how someone makes choices) to infer someone's mental states 
#(in this case, their reward function) from their choice. I added a lot of comments and wrote different 
#versions of the same function so it's easier to go through. 

#A niceness model would be more complicated:
#Each choice associated with a list of rewards (one for each event dimension that the observer cares about)
#Model would add up rewards associated with a choice to calculate final utility

    #A gives a to B -> A sacrifices
    #A gives bb to B -> A sacrifices + makes the agent happy

#If an observer cares about sacrifice but also about making the other person happy, then all choices 
#would have a sacrifice reward, but only the choice that the other agent likes would also have a "making 
#the other person happy" reward).


# In[108]:


def ToM(Desires, Tau=0.01):
	"""
	Minimal mental model.
	This model does not take beliefs into account and only assumes that the agent is more likely to choose higher utilities.
	Arguments:
	Desires [list]: an agent's desires formalized as a reward function
	Tau [float]: agent's level of rationality. As Tau is closer to 0, the agent is optimal, as Tau goes to infinity the agent selects randomly.
	By default Tau is set to assume that the agent is rational.
	"""
	# Transform the desires into probabilities of acting:
	# First, create a probability vector that equals the number of desires that the agent has
	Probabilities = [0] * len(Desires)
	# Iterate over each probability, and compute the soft-maxed desire
	for i in range(len(Probabilities)):
		Probabilities[i] = np.exp(Desires[i]/Tau)
	# Softmaxed vector is not a probability distribution because it does not add up to 1, so normalize it.
	NormalizingConstant = sum(Probabilities)
	for i in range(len(Probabilities)):
		Probabilities[i] = Probabilities[i]/NormalizingConstant
	# Now you have a vector of probabilities. Make a choice.
	ChoiceSpace = list(range(len(Desires))) # Just create a list from 0 to n.
	return(np.random.choice(ChoiceSpace,p=Probabilities))


# In[109]:


# 

# #ChoiceSpace = list(range(len(Desires)))

# #ChoiceSpace
# 	Desires = [10,0,2,1]
# 	Probabilities = [0] * len(Desires)
# for i in range(len(Probabilities)):
# 		Probabilities[i] = np.exp(Desires[i]/Tau)
# 	# Softmaxed vector is not a probability distribution because it does not add up to 1, so normalize it.
# 	NormalizingConstant = sum(Probabilities)
# 	for i in range(len(Probabilities)):
# 		Probabilities[i] = Probabilities[i]/NormalizingConstant
# 	return(Probabilities)


# In[110]:


def ToM_short(Desires, Tau=0.01):
	"""
	Equivalent function as above, but compressed using list comprehension.
	This function includes a simple math trick that doesn't affect the results
	but prevents underflow and overflow from happening.
	"""
	C = max(Desires) # This is a simple trick that doesn't affect the result after softmaxing
	# but it prevents numerical underflow (numbers that are too small) and overflow (numbers that are too big)
	Probabilities = [np.exp(i-C)/Tau for i in Desires]
	Probabilities = [i/sum(Probabilities) for i in Probabilities]
	return(np.random.choice(list(range(len(Desires))),p=Probabilities))


# In[111]:


# These are probabilistic functions, but the noise depends on rationality. For instance:
ToM([10,0]) # Agent has two options, rewards are 10 and 0, and (by default) agent is pretty rational. Should always select choose 10 [the choice in position 0 [0,1]] no matter how many times you run it.
[ToM([10,0]) for i in range(10)] # Run this agent 10 times


# In[112]:


ToM([10,0],Tau=500) # Lesion rationality. This agent should start choosing the option in position 1 of [0,1] fairly often.
[ToM([10,0], Tau=500) for i in range(10)] # Run irrational agent 10 times


# In[113]:


# When the utilities get big, simple softmax starts to break:
ToM([99,100]) # Should give the wrong answers: 0 instead of 1.
[ToM([99,100]) for i in range(10)] # Consistently chooses option 0.


# In[114]:


# This is because the exponential of the utility (the desire divided by Tau) is larger than what can be stored in a floating point variable.
# ToM_short fixes that:
ToM_short([99,100])
# Should now choose a mixture between 0 and 1 since they're both pretty much identical.
[ToM_short([99,100]) for i in range(10)] 


# In[115]:


# Strengthening rationality a lot. Agent should now choose option 1 more often.
[ToM_short([99,100], Tau=0.000001) for i in range(10)] 


# In[116]:


#### DESIRE INFERENCE MODEL
# Now we need a model for an observer who sees a choice and tries to infer the underlying reward function.
# To do this, we need an edited model of the agent, where instead of returning an action, it returns the probability of a given action.
def ToM_prob(Desires, Choice, Tau=0.01):
	"""
	Same model as above, but this function does not return a choice based on the desires, instead, it returns the probability of a choice.
	"""
	C = max(Desires)
	Probabilities = [np.exp((i-C)/Tau) for i in Desires]
	Probabilities = [i/sum(Probabilities) for i in Probabilities]
	# Everything is identical so far. But now, instead of sampling an action based on the probabilities,
	# just return the probability of the action of the choice
	return(Probabilities[Choice])


# In[117]:


def InferDesires(Options, Choice, Tau=0.01):
	"""
	Gets the number of options that an agent can choose from, gets the choice that the agent made, and the agent's rationality.
	Function uses Bayesian inference to infer the agent's desires, given their choice.
	"""
	# In a niceness model; Each choice associated w/ a list of rewards (one for each factor observer weights)
	# Model would add up rewards associated with a choice to calculate final utility
	## Inference Example Model
	# Easiest thing to do is just Monte-Carlo sampling.
	Samples = 500
	# Generate Monte Carlo samples:
	MC_Samples = [list(np.random.randint(0,50,Options)) for i in range(Samples)]
	# MC_Samples is now a list of samples. Each sample is a list (so this is a list of lists) with one reward between 0 and 50 for each option.
	# Biases go here. For now, Just assume any reward function is as probable.
	Prior = [1.0/Samples] * Samples
	# Compute the likelihood of each sample:
	# This is just calling ToM_prob, always with the same observed choice (Choice) and the same rationality (Tau) which we got as input
	# And it's calculating how likely different reward functions are.
	Likelihood = [ToM_prob(MC_Samples[i], Choice, Tau) for i in range(Samples)]
	# Now compute posterior
	Posterior = [Prior[i]*Likelihood[i] for i in range(Samples)]
	# Normalize
	Posterior = [i/sum(Posterior) for i in Posterior]
	# Marginalize over the distribution to get the expected reward function:
	InferredRewards = [0] * Options
	# This is just applying the marginalization rule to the monte carlo samples, but let me know if it's confusing.
	for i in range(Options):
		InferredRewards[i] = sum([MC_Samples[x][i]*Posterior[x] for x in range(Samples)])
	return(InferredRewards)


# In[118]:


# Samples = 500
# MC_Samples = [list(np.random.randint(0,50,2)) for i in range(Samples)]
# MC_Samples #we perhaps might bias these with an expected reward function
#Prior = [1.0/Samples] * Samples
#Prior #All options equally likely
# Likelihood = [ToM_prob(MC_Samples[i], 0, 0.01) for i in range(Samples)]
# Likelihood 
#InferredRewards = [0] * 3 # with 3 options, creates an event space of N = 3


# In[119]:


# Look at inferences:
InferDesires(2,0) # two options and agent chooses option 0


# In[120]:


InferDesires(2,0,Tau=1000) # should infer weaker preference because agent is less rational


# In[121]:


InferDesires(5,3, Tau=.001) # Should have a more or less uniform vector, with a higher reward on choice 3 [0,1,2,3,4] (fourth position)


# In[122]:


InferDesires(4,3)


# In[123]:


#############################
#### PROSOCIAL INFERENCE ####
#############################


# In[127]:


## Add features to the social environment ('Event')

class Event:
    def __init__(self, options, agentvalue, agentbeliefs, recipientvalue, choicespace, choice, outcome): # 'self' allows function to call itself
        self.options = options # unique resources a prosocial agent can share 
        self.agentvalue = agentvalue # the value the prosocial agent places on the resources being shared
        self.agentbeliefs = agentbeliefs # agents beliefs (certainty) about what the recipient values
        self.recipientvalue = recipientvalue # the value the recipient places on the resources (i.e., their needs)
        self.choicespace = choicespace # all possible prosocial choices
        self.choice = choice # the prosocial agent's choice
        self.outcome = outcome # prosocial agent's benefits from acting (e.g, praise, good feelings)
        
#additions
    #social norms (action is normative (common) vs supra-normative)
    #reward expectations (expects to benefit or does not)
    #reward outcomes (benefits or does not)
    #reactions (happy or not happy)
    
    
    # ToM should be called within a model of morality


# In[128]:


## Define features of our specific prosocial scenario

nice = Event(['a','b'], # options (unique actions)
                  [1,1], # agentvalue (agent's value on actions)
                  [[1,0],[0,1],[.5,.5]], # agentbeliefs (agent's beliefs (certainty) about recipient's value for each action)
                  [10,0], # recipientvalue (recipient's value on unique actions)
                  [['a','a'], # choicespace (possible actions) 
                   ['b','b'],
                   ['a','b'],
                  ['a','a','a','a'],
                  ['b','b','b','b'],
                  ['a','b','a','b']],
                  2, # choice in choicespace (action)
             1 # outcomes of choice (benefitted? y[1]/n[0])
            ) 


# loop through everything that can vary, nested for loops
# get predictions for each
# look in R
# some dimensions matter, some are interesting


# norms (simulating past observation of people, knowledge, 
#get norms by asking people who much they expect people to give from turk, input that into the 
#model
#- moral judgments 
#what people tell you the norms are, plug in to the model
#or set parameter, and try to fit that parameter to look like  
#try to figoure out the norm, based on niceeness judgment)


# In[129]:


### SACRIFICE MODEL ###

    ## Do some basic division

sacr_outcome = len(nice.choicespace[nice.choice]) /len(max(nice.choicespace,key=len))


print("niceness(Sacrifice model):", sacr_outcome)

## Other possible models (sacrifice relative to norm, or observed actions when repeated observation possible)


### OTHER-UTILITY MODEL ###

    ## Dictionary to convert unique actions into recipient utilities

utility = {
    nice.options[0]: nice.recipientvalue[0], # action 1 -> recipient utility 1
    nice.options[1]: nice.recipientvalue[1] # action 2 -> recipient utility 2
}

    ## Get choice utilities via list comprehension
u = []
for choice in range(len(nice.choicespace)):
    u.append(sum([utility[m] for m in nice.choicespace[choice]]))
    
util_outcome= abs(u[nice.choice]/max(u)) #Choice utility divided by max possibile choice utility


print("niceness(Other-utility model):", util_outcome)


### PURITY MODEL ###

    ## If the agent acted nice without benefitting, act was pure
if len(nice.choicespace[nice.choice]) > 0:
    pure_outcome=((len(nice.choicespace[nice.choice])*1) - nice.outcome)
else:
    pure_outcome=0
print("niceness(Purity model):", pure_outcome)


### HYBRID MODEL ###

#need to decide weights


### ToM Model ###


# In[93]:


if temperature > 70:
    print('Wear shorts.')
else:
    print('Wear long pants.')


# In[ ]:


### ToM MODEL

#define an agent, just an event
    #how much agent values each object [5,0]
    #how much agent value giving any object [1,1]
    #caring about others = 0-1
    
#where giving [a,a] would mean =  x
# they chose  choice 2 out of 3.
# simulate agents and see which agent would be maximizing 
# giving two 2, means don't care about value of a, or they care alot o

# how much they value objects and how much they care about person are unknown

# value of giving all objects should be equal


# In[58]:


#Michael's ToM niceness model

import numpy as np

# Giver doesn't know what Receiver wants
 
def receiver(rationality, receiver_rewards, giver_action):

	U = receiver_rewards + actions[giver_action]

	return U

def selfish_giver(rationality, giver_rewards):
	
	U = giver_rewards
	action_probabilities = softmax(U, rationality)

	return action_probabilities

def selfless_giver(rationality, giver_rewards, receiver_rewards=None):

	if type(receiver_rewards) != type(None):
		U = giver_rewards + (cooperation*receiver_rewards)
	else:
		receiver_rewards = np.random.choice(MAX_VALUE, (MAX_SAMPLES, NUM_ACTIONS))
		giver_actions = np.random.choice()
		for receiver_reward in receiver_rewards:
			for giver_action in giver_actions:
				U = giver
	
	action_probabilities = softmax(U, rationality)

	return action_probabilities


if "__name__" == __main__:
	giver_rewards = np.array([4, 4])

	objects = ["orange", "apple"]
	actions = np.random.choice(objects, p=(0.5, 0.5), size=())

	actions = {"a": 10, "b": -10}
	actions.keys()
	actions.values()
	actions["b"]



# In[104]:


#Michael's ToM model

from utils import * #version issue?

import itertools as it
import matplotlib.pyplot as plt
import numpy as np

def agent_no_ToM(rationality, agent_reward, enforcer_action):
	# Compute the utilities.
	agent_cost = convert_cost(enforcer_action) if GRIDWORLD == True else NATURAL_COST + enforcer_action
	U = agent_reward - agent_cost

	# Compute the action probabilities.
	action_probabilities = softmax(U, rationality)

	return action_probabilities

def agent_ToM(rationality, agent_reward, enforcer_action, cooperation, cache=False, plot=False):
	# Set up the likelihood space.
	space = tuple([MAX_VALUE for action in np.arange(NUM_ACTIONS)])
	likelihood = np.zeros(space)
	
	# Generate possible enforcer rewards.
	if SAMPLING == True:
		enforcer_rewards = np.random.choice(MAX_VALUE, (MAX_SAMPLES, NUM_ACTIONS))
	else:
		enforcer_rewards = np.array(list(it.product(np.arange(MAX_VALUE), repeat=NUM_ACTIONS)))

	# Compute the likelihood.
	if cache == True:
		likelihood = retrieve_enforcer_no_ToM(rationality, enforcer_rewards, enforcer_action, likelihood)
	else:
		for enforcer_reward in enforcer_rewards:
			enforcer_action_probabilities = enforcer(rationality, enforcer_reward, cache=True)
			likelihood[tuple(enforcer_reward)] = enforcer_action_probabilities[tuple(enforcer_action)]

	# Normalize the likelihood to generate the posterior.
	likelihood = likelihood.flatten()
	if sum(likelihood) == 0:
		posterior = likelihood.reshape(space)
	else:
		posterior = (likelihood/sum(likelihood)).reshape(space)

	# Plot the posterior.
	if plot == True:
		plt.figure()
		plt.title("ToM Agent with Rationality = " + str(rationality))
		plt.ylabel("Enforcer Rewards for Action 0")
		plt.xlabel("Enforcer Rewards for Action 1")
		plt.pcolor(posterior)

	# Compute the utilities.
	smart_agent_reward = agent_reward + cooperative_reward(enforcer_rewards, posterior, cooperation)
	smart_agent_cost = convert_cost(enforcer_action) if GRIDWORLD == True else NATURAL_COST + enforcer_action
	U = smart_agent_reward - smart_agent_cost

	# Compute the action probabilities.
	action_probabilities = softmax(U, rationality)

	return action_probabilities

def enforcer(rationality, enforcer_reward, p=0.0, cooperation=None, reward_assumptions=[], cache=False, plot=False):
	# Set up the utility space.
	space = tuple([MAX_VALUE for action in np.arange(NUM_ACTIONS)]) if GRIDWORLD != True else 			tuple([GRIDWORLD_MAX_ACTION for action in np.arange(NUM_ACTIONS)])
	U = np.zeros(space)
	
	# Generate possible agent rewards and enforcer actions, taking into account
	# any potential assumptions the enforcer may have about agent rewards.
	if SAMPLING == True:
		agent_rewards = np.random.choice(MAX_VALUE, (MAX_SAMPLES, NUM_ACTIONS))
		enforcer_actions = np.random.choice(MAX_VALUE, (MAX_SAMPLES, NUM_ACTIONS)) if GRIDWORLD != True else 						   np.random.choice(GRIDWORLD_MAX_ACTION, (GRIDWORLD_MAX_SAMPLES, NUM_ACTIONS))
	else:
		if len(reward_assumptions) == 0:
			agent_rewards = np.array(list(it.product(np.arange(MAX_VALUE), repeat=NUM_ACTIONS)))
		elif np.size(reward_assumptions) == NUM_ACTIONS:
			agent_rewards = np.array([reward_assumptions])
		else:
			agent_rewards = np.array(reward_assumptions)
		enforcer_actions = np.array(list(it.product(np.arange(MAX_VALUE), repeat=NUM_ACTIONS))) if GRIDWORLD != True else 						   np.array(list(it.product(np.arange(GRIDWORLD_MAX_ACTION), repeat=NUM_ACTIONS)))

	# Compute the utilities.
	if cache == True:
		U = retrieve_agent(rationality, enforcer_reward, agent_rewards, enforcer_actions, p, cooperation, U)
	else:
		U_agent_no_ToM = np.zeros(space)
		U_agent_ToM = np.zeros(space)
		temp_agent_no_ToM = np.zeros(space)
		temp_agent_ToM = np.zeros(space)
		for agent_reward in agent_rewards:
			for enforcer_action in enforcer_actions:
				# Reason about a non-ToM agent.
				if p != 1.0:
					agent_action_probabilities = agent_no_ToM(rationality, agent_reward, enforcer_action)
					expected_enforcer_reward = np.dot(enforcer_reward, agent_action_probabilities)
					temp_agent_no_ToM[tuple(enforcer_action)] = expected_enforcer_reward - (COST_RATIO*sum(enforcer_action))

				# Reason about a ToM agent.
				if p != 0.0:
					agent_action_probabilities = agent_ToM(rationality, agent_reward, enforcer_action, cooperation, cache=True)
					expected_enforcer_reward = np.dot(enforcer_reward, agent_action_probabilities)
					temp_agent_ToM[tuple(enforcer_action)] = expected_enforcer_reward - (COST_RATIO*sum(enforcer_action))

			U_agent_no_ToM = U_agent_no_ToM + temp_agent_no_ToM
			U_agent_ToM = U_agent_ToM + temp_agent_ToM
		U_agent_no_ToM = U_agent_no_ToM / len(agent_rewards)
		U_agent_ToM = U_agent_ToM / len(agent_rewards)
		U = ((1.0-p)*U_agent_no_ToM) + (p*U_agent_ToM)

	# Compute the action probabilities.
	action_probabilities = softmax(U.flatten(), rationality).reshape(space)

	# Plot the action probabilities.
	if plot == True:
		plt.figure()
		plt.title("Enforcing Agent with Rationality = " + str(rationality))
		plt.ylabel("Agent Cost (Enforcer Action) for Action 0")
		plt.xlabel("Agent Cost (Enforcer Action) for Action 1")
		plt.pcolor(action_probabilities)

	return action_probabilities

def observer(infer, rationality, **kwargs):
	# Infer the enforcer's reward.
	if infer == "enforcer_reward":
		# Extract variables.
		cooperation = kwargs["cooperation"]
		p = kwargs["p"]
		enforcer_action = kwargs["enforcer_action"]
		plot = kwargs["plot"]

		# Set up the likelihood space.
		space = tuple([MAX_VALUE for action in np.arange(NUM_ACTIONS)])
		likelihood = np.zeros(space)

		# Generate possible enforcer rewards.
		if SAMPLING == True:
			enforcer_rewards = np.random.choice(MAX_VALUE, (MAX_SAMPLES, NUM_ACTIONS))
		else:
			enforcer_rewards = np.array(list(it.product(np.arange(MAX_VALUE), repeat=NUM_ACTIONS)))
		
		# Compute the likelihood.
		for enforcer_reward in enforcer_rewards:
			enforcer_action_probabilities = enforcer(rationality, enforcer_reward, p=p, cooperation=cooperation, cache=True)
			likelihood[tuple(enforcer_reward)] = enforcer_action_probabilities[tuple(enforcer_action)]

		# Normalize the likelihood to generate the posterior.
		likelihood = likelihood.flatten()
		if sum(likelihood) == 0:
			posterior = likelihood.reshape(space)
		else:
			posterior = (likelihood/sum(likelihood)).reshape(space)

		# Plot the posterior.
		if plot == True:
			plt.figure()
			plt.title("Observer with Rationality = " + str(rationality))
			plt.ylabel("Enforcer Rewards for Action 0")
			plt.xlabel("Enforcer Rewards for Action 1")
			plt.pcolor(posterior)

	# Infer the enforcer's beliefs about the cooperativeness of agents.
	elif infer == "cooperation":
		# Extract variables.
		enforcer_reward = kwargs["enforcer_reward"]
		p = kwargs["p"]
		cooperation_set = kwargs["cooperation_set"]
		enforcer_action = kwargs["enforcer_action"]
		plot = kwargs["plot"]

		# Set up the space of possible cooperation parameters and the
		# likelihood space.
		cooperation_set = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 5.0])
		space = np.shape(cooperation_set)
		likelihood = np.zeros(space)

		# Compute the likelihood.
		for c in np.arange(cooperation_set.size):
			enforcer_action_probabilities = enforcer(rationality, enforcer_reward, p=p, cooperation=cooperation_set[c], cache=True)
			likelihood[c] = enforcer_action_probabilities[tuple(enforcer_action)]

		# Normalize the likelihood to generate the posterior.
		if sum(likelihood) == 0:
			posterior = likelihood
		else:
			posterior = likelihood / sum(likelihood)
		
		# Print the posterior.
		if plot == True:
			print(posterior)

	# Infer the degree of ToM that the enforcer was acting for.
	elif infer == "p":
		# Extract variables.
		enforcer_reward = kwargs["enforcer_reward"]
		cooperation = kwargs["cooperation"]
		enforcer_action = kwargs["enforcer_action"]
		plot = kwargs["plot"]

		# Set up the space of possible proportion parameters and the likelihood
		# space.
		p_set = np.linspace(0.0, 1.0, num=11)
		space = np.shape(p_set)
		likelihood = np.zeros(space)

		# Compute the likelihood.
		for p in np.arange(p_set.size):
			enforcer_action_probabilities = enforcer(rationality, enforcer_reward, p=p_set[p], cooperation=cooperation, cache=True)
			likelihood[p] = enforcer_action_probabilities[tuple(enforcer_action)]

		# Normalize the likelihood to generate the posterior.
		if sum(likelihood) == 0:
			posterior = likelihood
		else:
			posterior = likelihood / sum(likelihood)

		# Print the posterior.
		if plot == True:
			print(posterior)

	# Jointly infer what the enforcer's beliefs of the agent rewards and
	# the degree of ToM that the enforcer was acting for.
	elif infer == "agent_reward_and_p":
		# Extract variables.
		enforcer_reward = kwargs["enforcer_reward"]
		cooperation = kwargs["cooperation"]
		enforcer_action = kwargs["enforcer_action"]
		
		# Set up the space of possible proportion parameters and the likelihood
		# space.
		p_set = np.linspace(0.0, 1.0, num=11)
		space = (MAX_VALUE**2, p_set.size)
		likelihood = np.zeros(space)

		# Generate possible enforcer rewards.
		if SAMPLING == True:
			agent_rewards = np.random.choice(MAX_VALUE, (MAX_SAMPLES, NUM_ACTIONS))
		else:
			agent_rewards = np.array(list(it.product(np.arange(MAX_VALUE), repeat=NUM_ACTIONS)))

		# Compute the likelihood.
		for ar in np.arange(len(agent_rewards)):
			for p in np.arange(p_set.size):
				enforcer_action_probabilities = enforcer(rationality, enforcer_reward, p=p_set[p], cooperation=cooperation, 														 reward_assumptions=agent_rewards[ar], cache=True)
				likelihood[ar][p] = enforcer_action_probabilities[tuple(enforcer_action)]

		# Normalize the likelihood to generate the posterior.
		likelihood = likelihood.flatten()
		if sum(likelihood) == 0:
			posterior = likelihood.reshape(space)
		else:
			posterior = (likelihood/sum(likelihood)).reshape(space)

	return posterior


# In[ ]:


def Niceness(Options, Choice, Target_Util):
    sacr_Outcome = []
    oUtil_Outcome = []
    ToM_Outcome = []
    
# sacrifice model [if enabled]: 
    sacr_outcome = len(Choice)/max(Choice,key=len) # elements in choice / total possible elements
    # this model could be improved by sampling from largest amount observed across a set of observations
    # norm model?
            
# other-utility model [if enabled]:
    oUtil_Outcome = count(Target_Util) in Choice / max(Choice,key=len) #
    
# ToM model [if enabled]:
    ToM_Outcome
        InferDesires()
    
    Nice = sum[sacr_Outcome,oUtil_Outcome,ToM_Outcome] / count numeric values in [sacr_Outcome,oUtil_Outcome,ToM_Outcome]
    return Nice
    
    

#This model infers utility agents place on options based on choices
#We want the utility observers place on which features of the choice
#The model should add up rewards associated with a choice to calculate final utility

#Agent chose 2 out of 0,1,2 -> Agent kinda prefers 2
#Agent chose 2 out of 0=[a,a], 1=[b,b], 2=[a,a,a,a] -> Agent kinda prefers 2
#Agent chose 2 out of 0=[.5], 1=[.5], 2=[1.0] -> Agent kinda prefers [sacrificing for others] -> Agent must be kinda nice

#Agent in context A (not knowing target's preference) chose 2 

##Event space:
#(i) A thinks B wants = [a/b, ?]
#(ii) A expects a reward = [0,1]
#(iii) A gives = [a,a],[b,b],[a,b],[a,a,a,a],[b,b,b,b],[a,b,a,b]
#(iv) B wants = [a] or [b]
#(v) A gets a reward = [0,1]


#Generosity model (count(iii)/max(iii))
    #choice         ([a,a],[b,b],[a,b],[a,a,a,a],[b,b,b,b],[a,b,a,b])
    #choice niceness ([.5],[.5],[.5],[1.0],[1.0],[1.0])
    
#Other-Utility model (sum contents of (iii) after multiplying by B's liking)/max(iii)
    #if B likes a:
    #choice         ([aa],[bb],[ab],[aaaa],[bbbb],[abab])
    #choice niceness ([.5],[.0],[.25],[1.0],[0],[.5])
    
#ToM model (iii) defined by (i) where (i) can be a, b, or ?
    #if A (mistakenly) thinks B likes b
    #choice         ([aa],[bb],[ab],[aaaa],[bbbb],[abab])
    #choice niceness ([.0],[.1],[.5],[0],[1],[.5])
    
    #if A is uncertain what B likes 
    #choice         ([aa],[bb],[ab],[aaaa],[bbbb],[abab])
    #choice niceness ([.5],[.5],[1.0],[.5],[.5],[1.0])
    

# In this model "Choice" = sum of utility yielded across each model (equally weighted)

# Questions:

# at what point is bayesian inference coming into play?
# probability of what? what counts as an option for the agent

# OPTIONS
# aa
# bb
# ab
# aa
# bb
# ab
# aa
# bb
# ab
# aaaa
# bbbb
# abab
# aaaa
# bbbb
# abab
# aaaa
# bbbb
# abab

#CHOICE
#aa

