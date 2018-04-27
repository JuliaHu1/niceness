from Observer import *
from Agent import *
from Event import *

MyEvent = Event(options=[0,1],agentvalue=[2.5,2],agentbeliefs=[9,3],recipientrewards=[4,8])
MyAgent = Agent()
# Create an observer with an event and a mental model of the agent
MyObserver = Observer(MyEvent, MyAgent)
MyObserver.ToM(1) # Infers that agent really cares about own utilities and not about others
MyObserver.ToM(0) # Uncertain.

# Same as above, but now make agent value option 0 a lot
MyEvent = Event(options=[0,1],agentvalue=[10,2],agentbeliefs=[9,3],recipientrewards=[4,8])
MyAgent = Agent()
MyObserver = Observer(MyEvent, MyAgent)
MyObserver.ToM(1) # In contrast to above, now less certain that agent is selfish (because option 0 is very valuable)
MyObserver.ToM(0) # Should infer low selfishness and high altruism
