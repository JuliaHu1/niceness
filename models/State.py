class State:

    def __init__(self, actions, actor_rewards, actor_beliefs, receiver_rewards):
        self.actions = actions
        self.actor_rewards = actor_rewards
        self.actor_beliefs = actor_beliefs
        self.receiver_rewards = receiver_rewards
