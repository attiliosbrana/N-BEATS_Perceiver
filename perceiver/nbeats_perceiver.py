from typing import Tuple

import numpy as np
import torch as t
import torch.nn.functional as F
from perceiver import PerceiverEmbeddings, PerceiverConfig, PerceiverEncoder

class Mish(t.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x * (t.tanh(F.softplus(x)))
        return x

def generic(stack_params):
    input_size, output_size, stacks, layers, layer_size = stack_params
    """
    Create N-BEATS generic model.
    """
    return NBeats(t.nn.ModuleList([NBeatsBlock(input_size=input_size,
                                               theta_size=input_size + output_size,
                                               basis_function=GenericBasis(backcast_size=input_size,
                                                                           forecast_size=output_size),
                                               layers=layers,
                                               layer_size=layer_size)
                                   for _ in range(stacks)]))


#Create the same block as above, but with a TransformerEnconderLayer
class NBeatsBlock(t.nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self,
                 input_size,
                 theta_size: int,
                 basis_function: t.nn.Module,
                 layers: int,
                 layer_size: int):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.config = PerceiverConfig()
        self.config.d_model = input_size
        self.config.batch_size = 32 
        self.perceiver = PerceiverEncoder(self.config, self.config.d_model)
        self.embed = PerceiverEmbeddings(self.config)
        self.layers = t.nn.ModuleList([t.nn.Linear(in_features=self.config.d_latents * self.config.num_latents, out_features=layer_size)] +
                                      [t.nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = t.nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function
        self.Mish = Mish()

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        x = x.unsqueeze(1)
        out = self.perceiver(self.embed(self.config.batch_size), inputs = x)
        block_input = out[0].view(out[0].shape[0], -1)
        for layer in self.layers:
            block_input = self.Mish(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)

class NBeats(t.nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self, blocks: t.nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: t.Tensor, input_mask: t.Tensor) -> t.Tensor:
        # pad x and input_mask so that they increase input_size by 5
        x = t.nn.functional.pad(x, (0, 5))
        input_mask = t.nn.functional.pad(input_mask, (0, 5))
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast


class GenericBasis(t.nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]