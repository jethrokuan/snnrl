"""Encoders for various kinds of data."""
import torch


def poisson(datum, time, dt=1.0):
    assert (datum >= 0).all(), "Inputs must be non-negative"
    shape, size = datum.shape, datum.numel()
    datum = datum.view(-1)
    time = int(time / dt)

    rate = torch.zeros(size)
    rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time + 1]))
    intervals[:, datum != 0] = (intervals[:, datum != 0] == 0).float()

    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time + 1] = 0

    spikes = torch.zeros(time + 1, size)
    spikes[times, torch.arange(size)] = 1
    spikes = spikes[1:]

    return spikes.view(*shape, time)


class CartPoleEncoder:
    """
    Observation

    Type: Box(4)
    Num Observation 	     Min     Max
    0 	Cart Position        -2.4    2.4
    1 	Cart Velocity        -Inf    Inf
    2 	Pole Angle           ~-41.8° ~41.8°
    3 	Pole Velocity At Tip -Inf   Inf

    We encode each observation into 2 values:
    1. sign (0 for negative, 1 for positive)
    2. magnitude ()
    """

    def __init__(self, scale):
        self.obs_max = [3, -1, 42, -1]
        self.scale = scale

    @staticmethod
    def _squeeze(obs, max_val):
        if max_val == -1:  # Inf
            return 1 - math.exp(-obs)
        return obs / max_val

    def __call__(self, obs, timesteps):
        obs_mag = [1 if o > 0 else 0 for o in obs]
        obs_vals = [
            self._squeeze(abs(o), self.obs_max[i]) * self.scale
            for i, o in enumerate(obs)
        ]
        encoded_obs = torch.Tensor(obs_mag + obs_vals).reshape(-1, 1, 1)
        return poisson(encoded_obs, timesteps)
