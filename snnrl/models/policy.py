"""Simple policies"""

import torch
from torch.distributions import Categorical
from snnrl.models.mlp import MLP


class CategoricalPolicy(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, action_dim, activation=torch.nn.Tanh):
        super(CategoricalPolicy, self).__init__()
        self.logits = MLP(
            input_size, hidden_sizes, output_size=action_dim, activation=activation
        )

    def forward(self, inputs):
        logits = self.logits(inputs)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()

        return pi, logp_pi


class ActorCritic(torch.nn.Module):
    def __init__(self, policy, value_function):
        super(ActorCritic, self).__init__()
        v_out_size = list(value_function.modules())[-1].out_features
        assert v_out_size == 1, "The value function must have a single output value, got: {}".format(list(value_function.modules())[-1].out_features)
        self.policy = policy
        self.value_function = value_function

    def forward(self, inputs):
        pi, logp_pi = self.policy(inputs)
        v = self.value_function(inputs).squeeze()

        return pi, logp_pi, v


if __name__ == "__main__":
    actor_critic = ActorCritic(
        policy=CategoricalPolicy(10, [100, 200], 2),
        value_function=MLP(10, [100, 200], 1)
    )
    pi, logp_pi, v = actor_critic(torch.zeros([10], dtype=torch.float32))
    print(pi, logp_pi, v)
