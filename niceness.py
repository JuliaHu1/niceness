from utils import *

import itertools as it
import numpy as np

# Goals:
# (1) Iterate through different model weights
# (2) Simulate data from predictions

def actor_sacrifice(actor_action):
    # Remove the empty options. Replace this by changing actor_action format.
    actor_action = [option for option in actor_action if option != ""]

    return len(actor_action) / len(actor_actions[0])

# Observer needs to infer actor's beliefs of receiver's utility
def actor_intention():
    # Actor has a utility for each action in the action space
    actor_utilities = {"": 0, "a": 4, "b": 2}
    receiver_utilities = {"": 0, "a": 2, "b": 4}


def receiver(receiver_utility, actor_action):
    # Convert the actor action into utilities and add them to get the overall receiver utility.
    U = sum([receiver_utility[option] for option in actor_action])

    return U

def actor(rationality, actions, receiver_utility):
    # Define a space of potential actions and reciver utilities.
    space = []

    # Sample receiver utilities.
    # receiver_utilities = list(it.product(range(MAX_VALUE), repeat=len(options)))

    # Plug them into receiver model and see which one maximizes
    # for aa in range(len(actor_actions)):
    #     for ru in range(len(receiver_utilities)):
    #         total_receiver_utility[aa][ru] = receiver(ru, aa)

    for aa in range(len(actions)):
        U[aa] = receiver(receiver_utility, actions[aa])

    # Softmax the utilities 
    action_probabilities = softmax(rationality, U)

    return action_probabilities

def observer():
    actor_action
    # sample receiver rewards and plug into actor model



# class Event:
#     def __init__(self, options, agentvalue, agentbeliefs, recipientvalue, choicespace, choice, outcome): # 'self' allows function to call itself
#         self.options = options # unique resources a prosocial agent can share 
#         self.agentvalue = agentvalue # the value the prosocial agent places on the resources being shared
#         self.agentbeliefs = agentbeliefs # agents beliefs (certainty) about what the recipient values
#         self.recipientvalue = recipientvalue # the value the recipient places on the resources (i.e., their needs)
#         self.choicespace = choicespace # all possible prosocial choices
#         self.choice = choice # the prosocial agent's choice
#         self.outcome = outcome # prosocial agent's benefits from acting (e.g, praise, good feelings)

if __name__ == "__main__":
    
    # Generate the action space.
    actions = list(it.combinations_with_replacement(OPTIONS, r=MAX_OPTIONS))
    for a in range(len(actions)):
        actions[a] = [option for option in actions[a] if option != ""]

    receiver_utilities = {"": 0, "a": 7, "b": 2}
    actor_action = actions[4]
    print(actor_action)
    print()
    print(receiver(receiver_utilities, actor_action))