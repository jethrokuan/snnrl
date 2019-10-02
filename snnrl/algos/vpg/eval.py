import gym
import torch
from snnrl.models.policy import ActorCritic, CategoricalPolicy, MLP

policy = {
    "hidden": [32, 64]
}
vf = {
    "hidden": [32, 64]
}

env = gym.make("CartPole-v0")
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape

actor_critic = ActorCritic(
        policy=CategoricalPolicy(obs_dim[0], policy["hidden"], env.action_space.n),
        value_function=MLP(obs_dim[0], vf["hidden"], 1)
    )

actor_critic.load_state_dict(torch.load("data/5"))
actor_critic.eval()

for i in range(100):
    d = False
    print(i)
    obs = env.reset()
    while not d:
        act, _, _, _ = actor_critic(torch.Tensor(obs.reshape(1, -1)))
        print(act)
        obs, r, d, _ = env.step(act.detach().numpy()[0])
        env.render()
