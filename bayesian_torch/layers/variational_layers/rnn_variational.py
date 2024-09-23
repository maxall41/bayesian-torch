# Copyright (C) 2024 Intel Labs
#
# BSD-3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# LSTM Reparameterization Layer with reparameterization estimator to perform
# variational inference in Bayesian neural networks. Reparameterization layers
# enables Monte Carlo approximation of the distribution over 'kernel' and 'bias'.
#
# Kullback-Leibler divergence between the surrogate posterior and prior is computed
# and returned along with the tensors of outputs after linear opertaion, which is
# required to compute Evidence Lower Bound (ELBO).
#
# @authors: Piero Esposito
#
# ======================================================================================

from .linear_variational import LinearReparameterization
from ..base_variational_layer import BaseVariationalLayer_
import torch.nn as nn

import torch


class LSTMReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 batch_first=False,
                 bias=True):
        """
        Implements LSTM layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init std for the trainable mu parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init: float -> init std for the trainable rho parameter, sampled from N(0, posterior_rho_init),
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias
        self.batch_first = batch_first

        self.ih = LinearReparameterization(
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            in_features=in_features,
            out_features=out_features * 4,
            bias=bias)

        self.hh = LinearReparameterization(
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            in_features=out_features,
            out_features=out_features * 4,
            bias=bias)

    def kl_loss(self):
        kl_i = self.ih.kl_loss()
        kl_h = self.hh.kl_loss()
        return kl_i + kl_h

    def forward(self, X, hidden_states=None, return_kl=True):

        if self.dnn_to_bnn_flag:
            return_kl = False
        if self.batch_first:
            X = X.transpose(0, 1)
        
        batch_size, seq_size, _ = X.size()

        hidden_seq = []
        c_ts = []

        if hidden_states is None:
            h_t, c_t = (torch.zeros(batch_size,
                                    self.out_features).to(X.device),
                        torch.zeros(batch_size,
                                    self.out_features).to(X.device))
        else:
            h_t, c_t = hidden_states

        HS = self.out_features
        kl = 0
        for t in range(seq_size):
            x_t = X[:, t, :]

            ff_i, kl_i = self.ih(x_t)
            ff_h, kl_h = self.hh(h_t)
            gates = ff_i + ff_h

            kl += kl_i + kl_h

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
            c_ts.append(c_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        c_ts = torch.cat(c_ts, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        c_ts = c_ts.transpose(0, 1).contiguous()

        if return_kl:
            return hidden_seq, (hidden_seq, c_ts), kl
        return hidden_seq, (hidden_seq, c_ts)


class BidirectionalLSTMReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 num_layers=1,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True,
                 batch_first=False):
        """
        Implements Multi-layer Bidirectional LSTM layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            num_layers: int -> number of recurrent layers. Default: 1
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init std for the trainable mu parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init: float -> init std for the trainable rho parameter, sampled from N(0, posterior_rho_init),
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
            batch_first: bool -> if True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias = bias
        self.batch_first = batch_first

        self.forward_layers = nn.ModuleList()
        self.backward_layers = nn.ModuleList()

        for layer in range(num_layers):
            layer_in_features = in_features if layer == 0 else out_features * 2

            # Forward LSTM
            self.forward_layers.append(nn.ModuleDict({
                'ih': LinearReparameterization(
                    prior_mean=prior_mean,
                    prior_variance=prior_variance,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                    in_features=layer_in_features,
                    out_features=out_features * 4,
                    bias=bias),
                'hh': LinearReparameterization(
                    prior_mean=prior_mean,
                    prior_variance=prior_variance,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                    in_features=out_features,
                    out_features=out_features * 4,
                    bias=bias)
            }))

            # Backward LSTM
            self.backward_layers.append(nn.ModuleDict({
                'ih': LinearReparameterization(
                    prior_mean=prior_mean,
                    prior_variance=prior_variance,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                    in_features=layer_in_features,
                    out_features=out_features * 4,
                    bias=bias),
                'hh': LinearReparameterization(
                    prior_mean=prior_mean,
                    prior_variance=prior_variance,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                    in_features=out_features,
                    out_features=out_features * 4,
                    bias=bias)
            }))

    def kl_loss(self):
        kl = 0
        for layer in range(self.num_layers):
            kl += (self.forward_layers[layer]['ih'].kl_loss() +
                   self.forward_layers[layer]['hh'].kl_loss() +
                   self.backward_layers[layer]['ih'].kl_loss() +
                   self.backward_layers[layer]['hh'].kl_loss())
        return kl

    def _process_direction(self, X, layer, reverse=False):
        batch_size, seq_size, _ = X.size()
        hidden_seq = []
        h_t = c_t = torch.zeros(batch_size, self.out_features).to(X.device)
        HS = self.out_features
        kl = 0

        ih, hh = (layer['ih'], layer['hh'])
        sequence = range(seq_size) if not reverse else range(seq_size - 1, -1, -1)

        for t in sequence:
            x_t = X[:, t, :]

            ff_i, kl_i = ih(x_t)
            ff_h, kl_h = hh(h_t)
            gates = ff_i + ff_h

            kl += kl_i + kl_h

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        if reverse:
            hidden_seq = hidden_seq.flip(0)
        
        return hidden_seq, kl

    def forward(self, X, hidden_states=None, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        # If batch_first, transpose the input
        if self.batch_first:
            X = X.transpose(0, 1)

        kl_total = 0
        for layer in range(self.num_layers):
            forward_hidden, forward_kl = self._process_direction(X, self.forward_layers[layer])
            backward_hidden, backward_kl = self._process_direction(X, self.backward_layers[layer], reverse=True)

            # Concatenate forward and backward sequences
            hidden_seq = torch.cat([forward_hidden, backward_hidden], dim=2)
            kl_total += forward_kl + backward_kl

            # Use this layer's output as input to the next layer
            X = hidden_seq

        # If batch_first, transpose the output back
        if self.batch_first:
            hidden_seq = hidden_seq.transpose(0, 1)
        else:
            hidden_seq = hidden_seq.contiguous()

        if return_kl:
            return hidden_seq, None, kl_total
        return hidden_seq, None