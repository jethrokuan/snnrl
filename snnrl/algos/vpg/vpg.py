from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from snnrl.models.policy import CategoricalPolicy, MLP, ActorCritic, CNN
from snnrl.algos.vpg.buffer import VPGBuffer
from torch.utils.tensorboard import SummaryWriter

from snnrl.utils import save_checkpoint

import gym
import torch
import torch.nn.functional as F
import argparse
import json
import sys
from statistics import mean, variance

parser = argparse.ArgumentParser(description="Vanilla Policy Gradients")

parser.add_argument(
    "--save_every",
    type=int,
    default=10,
    help="Frequency of saving the trained model.",
    required=True,
)
parser.add_argument(
    "--epochs", type=int, help="Number of epochs to train the model for.", required=True
)
parser.add_argument(
    "--steps_per_epoch", type=int, help="Number of steps per epoch.", required=True
)
parser.add_argument(
    "--max_ep_length", type=int, help="Maximum length of episode.", required=True
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cuda", "cpu"],
    help="Device to run VPG on.",
    required=True,
)
parser.add_argument(
    "--gym_env", type=str, help="Gym environment to load.", required=True
)
parser.add_argument(
    "--policy_hidden",
    type=str,
    help="Policy hidden sizes. e.g. '[32, 64]'",
    required=True,
)
parser.add_argument(
    "--policy_lr", type=float, help="Policy learning rate.", required=True
)
parser.add_argument(
    "--vf_hidden",
    type=str,
    help="Value Function hidden sizes. e.g. '[32, 64]'",
    required=True,
)
parser.add_argument(
    "--vf_lr", type=float, help="Value Function learning rate.", required=True
)

parser.add_argument(
    "--vf_iters",
    type=int,
    help="Number of iterations for training the value function.",
    required=True,
)

parser.add_argument(
    "--save_dir",
    type=str,
    default="data",
    help="Directory to export model data.",
    required=True,
)
parser.add_argument(
    "--gae_gamma", type=float, help="Gamma for GAE-lambda.", required=True
)
parser.add_argument(
    "--gae_lam", type=float, help="Lambda for GAE-lambda.", required=True
)

parser.add_argument(
    "--show_env",
    type=str,
    help="Whether to show the model training window.",
    required=True,
)

args = parser.parse_args(sys.argv[1:])


def load_arr(arg):
    return map(int, arg.split(","))


policy_hidden = load_arr(args.policy_hidden)
vf_hidden = load_arr(args.vf_hidden)

writer = SummaryWriter(".")
device = torch.device(args.device)
env = gym.make(args.gym_env)
if args.show_env == "no":
    env.set_visibility(False)
o = env.reset()
screen = env.get_screen()
_, screen_height, screen_width = o.shape
frame_history = 4

obs_dim = o.shape
act_dim = env.action_space.shape

actor_critic = ActorCritic(
    policy=CategoricalPolicy(
        (screen_height, screen_width, frame_history), policy_hidden, env.action_space.n, model="CNN"
    ),
    value_function=CNN(screen_height, screen_width, frame_history, 1),
)

actor_critic.to(device=device)

buf = VPGBuffer(obs_dim, act_dim, args.steps_per_epoch, args.gae_gamma, args.gae_lam)

train_pi = torch.optim.Adam(actor_critic.policy.parameters(), lr=args.policy_lr)
train_v = torch.optim.Adam(actor_critic.value_function.parameters(), lr=args.vf_lr)


def update(epoch):
    obs, act, adv, ret, logp_old = [torch.Tensor(x).to(device) for x in buf.get()]

    _, logp, _ = actor_critic.policy(obs, act)
    ent = (-logp).mean()

    pi_loss = -(logp).mean()
    pi_loss = (-logp * adv).mean()

    train_pi.zero_grad()
    pi_loss.backward()
    train_pi.step()

    for _ in range(args.vf_iters):
        v = actor_critic.value_function(obs).squeeze()
        v_loss = F.mse_loss(v, ret)

        train_v.zero_grad()
        v_loss.backward()
        train_v.step()

    # Capture changes from update
    _, logp, _, v = actor_critic(obs, act)
    pi_l_new = -(logp * adv).mean()
    v_l_new = F.mse_loss(v.squeeze(), ret)
    kl = (logp_old - logp).mean()
    writer.add_scalar("pi_loss/train", pi_loss, epoch)
    writer.add_scalar("vf_loss/train", v_loss, epoch)
    writer.add_scalar("kl/train", kl, epoch)
    writer.add_scalar("ent/train", ent, epoch)
    writer.add_scalar("delta_pi/train", (pi_l_new - pi_loss), epoch)


o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

# Training loop
for epoch in range(args.epochs):
    actor_critic.eval()
    ep_ret_lst, ep_len_lst = [], []
    for t in range(args.steps_per_epoch):
        a, _, logp_t, v_t = actor_critic(o.unsqueeze(0).to(device))
        buf.store(
            o, a.cpu().detach().numpy(), r, v_t.item(), logp_t.cpu().detach().numpy()
        )
        action = a.cpu().detach().numpy()[0]
        o, r, d, _ = env.step(action)
        ep_ret += r
        ep_len += 1

        terminal = d or (ep_len == args.max_ep_length)
        if terminal or (t == args.steps_per_epoch - 1):
            if not (terminal):
                print("Warning: trajectory cut off by epoch at %d steps." % ep_len)
            # if trajectory didn't reach terminal state, bootstrap value target
            last_val = (
                r
                if d
                else actor_critic.value_function(o.unsqueeze(0).to(device)).item()
            )
            buf.finish_path(last_val)
            ep_len_lst.append(ep_len)
            ep_ret_lst.append(ep_ret)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    writer.add_scalar("ep_ret_mean/train", mean(ep_ret_lst), epoch)
    writer.add_scalar("ep_len_mean/train", mean(ep_len_lst), epoch)
    writer.add_scalar("ep_ret_var/train", variance(ep_ret_lst), epoch)
    writer.add_scalar("ep_len_var/train", variance(ep_len_lst), epoch)
    actor_critic.train()

    update(epoch)

    if epoch % args.save_every == 0 or epoch == args.epochs - 1:
        save_checkpoint({"env": args.gym_env}, actor_critic, args.save_dir, epoch)
