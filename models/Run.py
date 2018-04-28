from utils import *

from Actor import *
from Event import *
from Observer import *

import itertools as it
import sys


# Generate the action space.
actions = list(it.combinations_with_replacement(OPTIONS, r=MAX_OPTIONS))
for a in range(len(actions)):
    actions[a] = [option for option in actions[a] if option != ""]

ar = {"a": 9, "b": 2}
ab = {"a": 9, "b": 3}
rr = {"a": 4, "b": 8}
actor_rewards, actor_beliefs, receiver_rewards = [], [], []
for action in actions:
	actor_rewards.append([ar[option] for option in action if action != []])
	actor_beliefs.append([ab[option] for option in action if action != []])
	receiver_rewards.append([rr[option] for option in action if action != []])

# actor_rewards[0] = 0

print(actor_rewards)
print(actor_beliefs)
print(receiver_rewards)

event = Event(actions=np.array(actions), actor_rewards=np.array(actor_rewards), actor_beliefs=np.array(actor_beliefs), 
			  receiver_rewards=np.array(receiver_rewards))
actor = Actor()
observer = Observer(event, actor)
print("Actions:", actions)
index = 0
action = actions[index]

print("Action:", action)
# print(actions == action)

print(observer.ToM(index))
sys.exit("Done.")

MyEvent = Event(options=[0,1],agentvalue=[2.5,2],agentbeliefs=[9,3],recipientrewards=[4,8])
MyAgent = Agent()
# Create an observer with an event and a mental model of the agent
MyObserver = Observer(MyEvent, MyAgent)
print(MyObserver.ToM(1)) # Infers that agent really cares about own utilities and not about others
# MyObserver.ToM(0) # Uncertain.

# Same as above, but now make agent value option 0 a lot
MyEvent = Event(options=[0,1],agentvalue=[10,2],agentbeliefs=[9,3],recipientrewards=[4,8])
MyAgent = Agent()
MyObserver = Observer(MyEvent, MyAgent)
# MyObserver.ToM(1) # In contrast to above, now less certain that agent is selfish (because option 0 is very valuable)
# MyObserver.ToM(0) # Should infer low selfishness and high altruism
