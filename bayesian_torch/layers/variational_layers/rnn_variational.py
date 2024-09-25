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


import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from ..base_variational_layer import BaseVariationalLayer_

class BayesianLSTM(BaseVariationalLayer_):
    def __init__(self, in_features, out_features, num_layers=1, bias=True, batch_first=False, 
                 dropout=0., bidirectional=False, prior_mean=0, prior_variance=1, 
                 posterior_mu_init=0, posterior_rho_init=-3.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        
        # Create a standard PyTorch LSTM
        self.lstm = nn.LSTM(in_features, out_features, num_layers, bias=bias, 
                            batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        
        # Initialize posterior parameters
        self.weight_mu = Parameter(torch.Tensor(self.lstm.all_weights[0][0].shape))
        self.weight_rho = Parameter(torch.Tensor(self.lstm.all_weights[0][0].shape))
        if bias:
            self.bias_mu = Parameter(torch.Tensor(self.lstm.all_weights[0][1].shape))
            self.bias_rho = Parameter(torch.Tensor(self.lstm.all_weights[0][1].shape))
        
        # Initialize posterior
        self.weight_mu.data.normal_(posterior_mu_init, 0.1)
        self.weight_rho.data.normal_(posterior_rho_init, 0.1)
        if bias:
            self.bias_mu.data.normal_(posterior_mu_init, 0.1)
            self.bias_rho.data.normal_(posterior_rho_init, 0.1)
        
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        if bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
    
    def forward(self, x, hidden_states=None, return_kl=True):
        # Sample weights
        weight_epsilon = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + self.weight_sigma * weight_epsilon
        
        if self.bias:
            bias_epsilon = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + self.bias_sigma * bias_epsilon
        else:
            bias = None
        
        # Replace LSTM weights with sampled weights
        with torch.no_grad():
            for i in range(len(self.lstm.all_weights)):
                self.lstm.all_weights[i][0].copy_(weight)
                if self.bias:
                    self.lstm.all_weights[i][1].copy_(bias)
        
        # Forward pass
        output, hidden = self.lstm(x, hidden_states)
        
        if return_kl:
            kl = self.kl_loss()
            return output, hidden, kl
        return output, hidden
    
    def kl_loss(self):
        kl = self.kl_div(self.weight_mu, self.weight_sigma, self.prior_mean, self.prior_variance)
        if self.bias:
            kl += self.kl_div(self.bias_mu, self.bias_sigma, self.prior_mean, self.prior_variance)
        return kl
    
    @staticmethod
    def kl_div(mu_q, sig_q, mu_p, sig_p):
        kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + 
                    ((mu_p - mu_q) / sig_p).pow(2)).sum()
        return kl