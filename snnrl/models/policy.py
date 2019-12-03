"""Simple policies"""

import torch
from torch.distributions import Categorical
from snnrl.models.mlp import MLP
from snnrl.models.cnn import CNN

class CategoricalPolicy(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, action_dim, activation=torch.nn.Tanh, model="MLP"):
        super(CategoricalPolicy, self).__init__()
        if model == "MLP":
            self.logits = MLP(
                input_size, hidden_sizes, output_size=action_dim, activation=activation
            )
        elif model == "CNN":
            self.logits = CNN(
                input_size[0], input_size[1], input_size[2], action_dim)
        else:
            raise Exception(f"Invalid model type: {model}")


    def forward(self, inputs, action=None):
        logits = self.logits(inputs)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()

        if action is not None:
            logp = policy.log_prob(action).squeeze()
        else:
            logp = None

        return pi, logp, logp_pi


class ActorCritic(torch.nn.Module):
    def __init__(self, policy, value_function):
        super(ActorCritic, self).__init__()
        v_out_size = list(value_function.modules())[-1].out_features
        assert v_out_size == 1, "The value function must have a single output value, got: {}".format(list(value_function.modules())[-1].out_features)
        self.policy = policy
        self.value_function = value_function

    def forward(self, inputs, action=None):
        pi, logp, logp_pi = self.policy(inputs, action)
        v = self.value_function(inputs)

        return pi, logp, logp_pi, v


if __name__ == "__main__":
    actor_critic = ActorCritic(
        policy=CategoricalPolicy(10, [100, 200], 2),
        value_function=MLP(10, [100, 200], 1)
    )
    pi, logp_pi, v = actor_critic(torch.zeros([10], dtype=torch.float32))
    print(pi, logp_pi, v)
