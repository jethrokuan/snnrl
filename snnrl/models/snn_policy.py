from snnrl.encoders.encoders import CartPoleEncoder, ImageEncoder

import slayerSNN as snn
import torch
from torch.distributions import Categorical


class SNNCategoricalPolicy(torch.nn.Module):
    def __init__(self, input_size, params):
        super(SNNCategoricalPolicy, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        # self.encoder = CartPoleEncoder(200)
        self.encoder = ImageEncoder()
        self.fc1 = self.slayer.dense((input_size[0] * input_size[1] * 3), 300)
        self.fc2 = self.slayer.dense(300, 2)  # output of 2 classes

    def _encode(self, o, ts):
        encoded = self.encoder(o * 255, ts)
        # squeeze all into channel dimension
        encoded = encoded.reshape(-1, 1, 1, encoded.shape[-1])
        return encoded

    def forward(self, inputs, action=None):
        encoded_input = torch.stack([self._encode(o, 300) for o in inputs]).to(
            device=torch.device("cuda")
        )
        sl1 = self.slayer.spike(self.slayer.psp(self.fc1(encoded_input)))
        sl2 = self.slayer.spike(self.slayer.psp(self.fc2(sl1)))
        num_spikes = torch.sum(sl2, 4)
        num_spikes = num_spikes.reshape((num_spikes.shape[0], -1))
        policy = Categorical(logits=num_spikes)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()

        if action is not None:
            logp = policy.log_prob(action).squeeze()
        else:
            logp = None

        return pi, logp, logp_pi


if __name__ == "__main__":
    import gym

    env = gym.make("CartPole-v0")
    obs = env.reset()
    device = torch.device("cuda")
    params = {
        "neuron": {
            "type": "SRMALPHA",
            "theta": 10,
            "tauSr": 10.0,
            "tauRef": 1.0,
            "scaleRef": 2,
            "tauRho": 1,
            "scaleRho": 1,
        },
        "simulation": {"Ts": 1.0, "tSample": 300, "nSample": 1},
    }
    net = SNNCategoricalPolicy(params).to(device=device)
    input = torch.Tensor(obs.reshape(1, -1)).to(device=device)
    output = net(input)
    print(output)
