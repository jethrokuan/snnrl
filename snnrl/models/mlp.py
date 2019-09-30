"""Multilayer Perception"""

import torch


class MLP(torch.nn.Module):
    """A simple MLP."""

    def __init__(self, input_size, hidden_sizes, output_size, activation=torch.nn.ReLU):
        """ Initializer.

        Args:
            layers: an arra
        """
        super(MLP, self).__init__()
        hidden_layers = [
            torch.nn.Linear(n_in, n_out)
            for n_in, n_out in zip([input_size] + hidden_sizes[:-1], hidden_sizes)
        ]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend([layer, activation()])
        last_size = hidden_sizes[-1] if hidden_sizes else input_size
        sequence.append(torch.nn.Linear(last_size, output_size))
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = output_size

    def forward(self, input_features):
        res = self.model(input_features)

        if self._output_size == 1:
            return res.squeeze()
        else:
            return res

    @property
    def output_size(self):
        return self._output_size


if __name__ == "__main__":
    # Simple MLP usage
    mlp = MLP(input_size=10, hidden_sizes=[100, 200], output_size=2)
    print(mlp)
