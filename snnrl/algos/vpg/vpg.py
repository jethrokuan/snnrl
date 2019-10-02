from snnrl.models.policy import CategoricalPolicy, MLP, ActorCritic
from snnrl.algos.vpg.buffer import VPGBuffer

import gym
import torch
import torch.nn.functional as F
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("vpg")
ex.observers.append(MongoObserver.create())

@ex.config
def cfg():
    steps_per_epoch = 4000
    num_epochs = 80
    gamma = 0.99
    lam = 1
    max_ep_len = 400
    gym_env = "CartPole-v0"
    policy = {
        "hidden": [32, 64]
    }
    vf = {
        "hidden": [32, 64]
    }
    save_loc = "data"

@ex.automain
def main(steps_per_epoch,
         num_epochs,
         gamma,
         lam,
         max_ep_len,
         gym_env,
         policy,
         vf,
         save_loc,
         _run):
    env = gym.make(gym_env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    actor_critic = ActorCritic(
        policy=CategoricalPolicy(obs_dim[0], policy["hidden"], env.action_space.n),
        value_function=MLP(obs_dim[0], vf["hidden"], 1)
    )

    buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    train_pi = torch.optim.Adam(actor_critic.policy.parameters(), lr=0.1)
    train_v = torch.optim.Adam(actor_critic.value_function.parameters(), lr=0.1)

    def update():
        obs, act, adv, ret, logp_old = [torch.Tensor(x) for x in buf.get()]

        _, logp, _ = actor_critic.policy(obs, act)
        ent = (-logp).mean()

        pi_loss = -(logp).mean()
        pi_loss = (-logp * adv).mean()

        train_pi.zero_grad()
        pi_loss.backward()
        train_pi.step()

        for _ in range(400):
            v = actor_critic.value_function(obs).squeeze()
            v_loss = F.mse_loss(v, ret)

            train_v.zero_grad()
            v_loss.backward()
            train_v.step()

        # Capture changes from update
        _, logp, _, v = actor_critic(obs, act)
        pi_l_new = -(logp * adv).mean()
        v_l_new = F.mse_loss(v, ret)
        kl = (logp_old - logp).mean()
        _run.log_scalar("training.piloss", pi_loss.item())
        _run.log_scalar("training.vf_loss", v_loss.item())
        _run.log_scalar("training.kl", kl.item())
        _run.log_scalar("training.ent", ent.item())
        _run.log_scalar("training.delta_pi", (pi_l_new - pi_loss).item())

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    for epoch in range(num_epochs):
        actor_critic.eval()
        for t in range(steps_per_epoch):
            a, _, logp_t, v_t = actor_critic(torch.Tensor(o.reshape(1, -1)))
            buf.store(o, a.detach().numpy(), r, v_t.item(), logp_t.detach().numpy())
            o, r, d, _ = env.step(a.detach().numpy()[0])
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else actor_critic.value_function(torch.Tensor(o.reshape(1,-1))).item()
                buf.finish_path(last_val)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        actor_critic.train()
        update()

    torch.save(actor_critic.state_dict(), "{}/{}".format(save_loc, _run._id))
