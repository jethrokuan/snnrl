import gym
import torch
import argparse
from snnrl.models.policy import ActorCritic, CategoricalPolicy, MLP

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", help="Path to checkpoint.", type=str, required=True)
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint)
actor_critic = checkpoint["model"]
env = gym.make(checkpoint["env"])
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape
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
