from Actor import *
from Observer import *
from State import *

import itertools as it
import numpy as np
import sys

# Parameterize the action space.
options = ["A", "B"]
max_options = int(sys.argv[1])
actions = list(it.combinations_with_replacement(options, r=max_options))
for a in range(len(actions)):
    actions[a] = [option for option in actions[a] if option != ""]

# Set the value of each option for the actor's rewards and beliefs and the
# receiver's rewards.
actor_reward_values = {"A": 2, "B": 9}
actor_belief_values = {"A": 3, "B": 9}
receiver_reward_values = {"A": 4, "B": 8}

# Generate actor rewards, actor beliefs, and receiver rewards by mapping the
# action space using the corresponding option values.
actor_rewards = []
actor_beliefs = []
receiver_rewards = []
for action in actions:
	actor_rewards.append([actor_reward_values[option] for option in action if action != []])
	actor_beliefs.append([actor_belief_values[option] for option in action if action != []])
	receiver_rewards.append([receiver_reward_values[option] for option in action if action != []])

# Convert to numpy arrays.
actions = np.array(actions)
actor_rewards = np.array(actor_rewards)
actor_beliefs = np.array(actor_beliefs)
receiver_rewards = np.array(receiver_rewards)

# Instantiate a state, actor, and observer.
state = State(actions=actions, actor_rewards=actor_rewards, actor_beliefs=actor_beliefs, receiver_rewards=receiver_rewards)
actor = Actor()
observer = Observer(state, actor)

print("Actions:")
print(actions)
index = 0
action = actions[index]
print("Action:")
print(action)

# Run the sacrifice model.
print(observer.sacrifice(action))

# Run the utilitarian model.
print(observer.utilitarian(action, receiver_reward_values))

# Run the ToM model.
print(observer.ToM(index))
