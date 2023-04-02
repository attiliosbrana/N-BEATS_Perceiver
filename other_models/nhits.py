import numpy as np

import torch as t
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from functools import partial


# Cell
class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert (interpolation_mode in ['linear','nearest']) or ('cubic' in interpolation_mode)
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(self, theta: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:

        backcast = theta[:, :self.backcast_size]
        knots = theta[:, self.backcast_size:]

        if self.interpolation_mode=='nearest':
            knots = knots[:,None,:]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:,0,:]
        elif self.interpolation_mode=='linear':
            knots = knots[:,None,:]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode) #, align_corners=True)
            forecast = forecast[:,0,:]
        elif 'cubic' in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split('-')[-1])
            knots = knots[:,None,None,:]
            forecast = t.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots)/batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(knots[i*batch_size:(i+1)*batch_size], size=self.forecast_size, mode='bicubic') #, align_corners=True)
                forecast[i*batch_size:(i+1)*batch_size] += forecast_i[:,0,0,:]

        return backcast, forecast

# Cell
def init_weights(module, initialization):
    if type(module) == t.nn.Linear:
        if initialization == 'orthogonal':
            t.nn.init.orthogonal_(module.weight)
        elif initialization == 'he_uniform':
            t.nn.init.kaiming_uniform_(module.weight)
        elif initialization == 'he_normal':
            t.nn.init.kaiming_normal_(module.weight)
        elif initialization == 'glorot_uniform':
            t.nn.init.xavier_uniform_(module.weight)
        elif initialization == 'glorot_normal':
            t.nn.init.xavier_normal_(module.weight)
        elif initialization == 'lecun_normal':
            pass #t.nn.init.normal_(module.weight, 0.0, std=1/np.sqrt(module.weight.numel()))
        else:
            assert 1<0, f'Initialization {initialization} not found'

# Cell
ACTIVATIONS = ['ReLU',
               'Softplus',
               'Tanh',
               'SELU',
               'LeakyReLU',
               'PReLU',
               'Sigmoid']

class _NHITSBlock(nn.Module):
    """
    N-HiTS block which takes a basis function as an argument.
    """
    def __init__(self, n_time_in: int, n_time_out: int, n_x: int,
                 n_s: int, n_s_hidden: int, n_theta: int, n_theta_hidden: list,
                 n_pool_kernel_size: int, pooling_mode: str, basis: nn.Module,
                 n_layers: int,  batch_normalization: bool, dropout_prob: float, activation: str):
        """
        """
        super().__init__()

        assert (pooling_mode in ['max','average'])

        n_time_in_pooled = int(np.ceil(n_time_in/n_pool_kernel_size))

        if n_s == 0:
            n_s_hidden = 0
        n_theta_hidden = [n_time_in_pooled + (n_time_in+n_time_out)*n_x + n_s_hidden] + n_theta_hidden

        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.n_s = n_s
        self.n_s_hidden = n_s_hidden
        self.n_x = n_x
        self.n_pool_kernel_size = n_pool_kernel_size
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob

        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        activ = getattr(nn, activation)()

        if pooling_mode == 'max':
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)
        elif pooling_mode == 'average':
            self.pooling_layer = nn.AvgPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(nn.Linear(in_features=n_theta_hidden[i], out_features=n_theta_hidden[i+1]))
            hidden_layers.append(activ)

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=n_theta_hidden[i+1]))

            if self.dropout_prob>0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=n_theta_hidden[-1], out_features=n_theta)]
        layers = hidden_layers + output_layer

        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:

        insample_y = insample_y.unsqueeze(1)
        # Pooling layer to downsample input
        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.squeeze(1)

        batch_size = len(insample_y)

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)

        return backcast, forecast

# Cell
class NHITS(nn.Module):
    """
    N-HiTS Model.
    """
    def __init__(self,
                 n_time_in, 
                 n_time_out, #This is 48
                 n_s = 0,
                 n_x = 0,
                 n_s_hidden = 0,
                 n_x_hidden = 0,
                 stack_types = 3*['identity'],
                 n_layers = 9*[2], #len(n_layers) = len(stack_types)
                 n_theta_hidden = 3 * [[512, 512]], #len(n_theta_hidden) = len(stack_types)
                 n_pool_kernel_size = 3*[8], #len(n_pool_kernel_size) = len(stack_types)
                 n_freq_downsample = [168, 24, 1], #len(n_freq_downsample) = len(stack_types)
                 n_blocks = 3*[1], #len(n_blocks) = len(stack_types)
                 interpolation_mode = 'linear',
                 batch_normalization = False,
                 initialization = 'lecun_normal',
                 dropout_prob_theta = 0.0,
                 pooling_mode = 'max',
                 activation = 'ReLU',
                 shared_weights = False):
        super().__init__()

        self.n_time_out = n_time_out

        blocks = self.create_stack(stack_types=stack_types,
                                   n_blocks=n_blocks,
                                   n_time_in=n_time_in,
                                   n_time_out=n_time_out,
                                   n_x=n_x,
                                   n_x_hidden=n_x_hidden,
                                   n_s=n_s,
                                   n_s_hidden=n_s_hidden,
                                   n_layers=n_layers,
                                   n_theta_hidden=n_theta_hidden,
                                   n_pool_kernel_size=n_pool_kernel_size,
                                   n_freq_downsample=n_freq_downsample,
                                   pooling_mode=pooling_mode,
                                   interpolation_mode=interpolation_mode,
                                   batch_normalization=batch_normalization,
                                   dropout_prob_theta=dropout_prob_theta,
                                   activation=activation,
                                   shared_weights=shared_weights,
                                   initialization=initialization)
        self.blocks = t.nn.ModuleList(blocks)

    def create_stack(self, stack_types, n_blocks,
                     n_time_in, n_time_out,
                     n_x, n_x_hidden, n_s, n_s_hidden,
                     n_layers, n_theta_hidden,
                     n_pool_kernel_size, n_freq_downsample, pooling_mode, interpolation_mode,
                     batch_normalization, dropout_prob_theta,
                     activation, shared_weights, initialization):

        block_list = []
        for i in range(len(stack_types)):
            #print(f'| --  Stack {stack_types[i]} (#{i})')
            for block_id in range(n_blocks[i]):

                # Batch norm only on first block
                if (len(block_list)==0) and (batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                # Shared weights
                if shared_weights and block_id>0:
                    nbeats_block = block_list[-1]
                else:
                    if stack_types[i] == 'identity':
                        n_theta = (n_time_in + max(n_time_out//n_freq_downsample[i], 1) )
                        basis = IdentityBasis(backcast_size=n_time_in,
                                              forecast_size=n_time_out,
                                              interpolation_mode=interpolation_mode)

                    else:
                        assert 1<0, f'Block type not found!'

                    nbeats_block = _NHITSBlock(n_time_in=n_time_in,
                                                   n_time_out=n_time_out,
                                                   n_x=n_x,
                                                   n_s=n_s,
                                                   n_s_hidden=n_s_hidden,
                                                   n_theta=n_theta,
                                                   n_theta_hidden=n_theta_hidden[i],
                                                   n_pool_kernel_size=n_pool_kernel_size[i],
                                                   pooling_mode=pooling_mode,
                                                   basis=basis,
                                                   n_layers=n_layers[i],
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=dropout_prob_theta,
                                                   activation=activation)

                # Select type of evaluation and apply it to all layers of block
                init_function = partial(init_weights, initialization=initialization)
                nbeats_block.layers.apply(init_function)
                #print(f'     | -- {nbeats_block}')
                block_list.append(nbeats_block)
        return block_list

    # def forward(self, S: t.Tensor, Y: t.Tensor, X: t.Tensor,
    #             insample_mask: t.Tensor, outsample_mask: t.Tensor,
    #             return_decomposition: bool=False):

    def forward(self, x, x_mask):
        # # insample
        # insample_y    = Y[:, :-self.n_time_out]
        # insample_x_t  = X[:, :, :-self.n_time_out]
        # insample_mask = insample_mask[:, :-self.n_time_out]

        # # outsample
        # outsample_y   = Y[:, -self.n_time_out:]
        # outsample_x_t = X[:, :, -self.n_time_out:]
        # outsample_mask = outsample_mask[:, -self.n_time_out:]

        x = t.nn.functional.pad(x, (0, 5))
        x_mask = t.nn.functional.pad(x_mask, (0, 5))
        forecast = self.forecast(insample_y=x,
                                    insample_mask=x_mask)
        return forecast 

    def forecast(self, insample_y: t.Tensor, insample_mask: t.Tensor):

        residuals = insample_y.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:] # Level with Naive1
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast

        return forecast
