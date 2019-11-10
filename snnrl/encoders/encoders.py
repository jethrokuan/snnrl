"""Encoders for various kinds of data."""
import torch
import numpy as np


def single(
    datum: torch.Tensor, time: int, dt: float = 1.0, sparsity: float = 0.5, **kwargs
) -> torch.Tensor:
    # language=rst
    """
    Generates timing based single-spike encoding. Spike occurs earlier if the
    intensity of the input feature is higher. Features whose value is lower than
    threshold is remain silent.
    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of the input and output.
    :param dt: Simulation time step.
    :param sparsity: Sparsity of the input representation. 0 for no spikes and 1 for all
        spikes.
    :return: Tensor of shape ``[time, n_1, ..., n_k]``.
    """
    datum = datum.cpu()
    time = int(time / dt)
    shape = list(datum.shape)
    datum = np.copy(datum)
    quantile = np.quantile(datum, 1 - sparsity)
    s = np.zeros([time, *shape])
    s[0] = np.where(datum > quantile, np.ones(shape), np.zeros(shape))
    return torch.Tensor(np.moveaxis(s, 0, -1))


def poisson(datum, time, dt=1.0):
    device = datum.device
    assert (datum >= 0).all(), "Inputs must be non-negative"
    shape, size = datum.shape, datum.numel()
    datum = datum.view(-1)
    time = int(time / dt)

    rate = torch.zeros(size).to(device)
    rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time + 1])).to(device)
    intervals[:, datum != 0] += (intervals[:, datum != 0] == 0).float()

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


class ImageEncoder:
    """ Flattens and encodes images"""

    def __call__(self, obs, ts):
        # return poisson(obs, ts)
        return single(obs, ts).to(obs.device)
        # return _to_spikes(obs.to("cpu"), length=ts, max_rate=255.0).to(obs.device)
