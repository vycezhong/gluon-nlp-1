# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
from mxnet.gluon import nn, Block, HybridBlock
from gcn import GCN
from transformer import TransformerEncoder
from rnn import RNN
from mlp import MLP

__all__ = ['DeepRoutingNetwork']


class DeepRoutingNetwork(Block):
    def __init__(self, enc_model, graph_size, embed_size, hidden_size,
                 concat, num_gcn_layers, num_encoder_layers,
                 max_length, num_heads=8, dropout=0.0, use_baseline=False,
                 prefix='deep_routing_model_', params=None):
        super(DeepRoutingNetwork, self).__init__(prefix=prefix, params=params)
        self.graph_size = graph_size
        self.embed_size = embed_size
        self.concat = concat
        self.use_baseline = use_baseline
        with self.name_scope():
            if not use_baseline:
                self.gcn = GCN(graph_size, embed_size, num_gcn_layers)
            else:
                self.embeddings = nn.Embedding(input_dim=graph_size, output_dim=embed_size)
            if enc_model == 'transformer':
                if concat:
                    enc_embed_size = embed_size * 2
                    hidden_size *= 2
                else:
                    enc_embed_size = embed_size
                self.encoder = TransformerEncoder(num_layers=num_encoder_layers,
                                                  num_heads=num_heads,
                                                  max_length=max_length,
                                                  units=enc_embed_size,
                                                  hidden_size=hidden_size,
                                                  dropout=dropout,
                                                  prefix=prefix + 'enc_', params=params)
            else:
                self.encoder = RNN(enc_model, hidden_size, num_encoder_layers, dropout)
            if concat:
                self.dense = nn.Dense(embed_size, use_bias=False, flatten=False)
            self.proj = MLP()

    def lookup_embeddings(self, seq, neighbors, destinations, embeddings):
        if not self.use_baseline:
            seq_embeddings = mx.nd.Embedding(seq, embeddings, self.graph_size, self.embed_size)
            neighbor_embeddings = mx.nd.Embedding(neighbors, embeddings, self.graph_size, self.embed_size)
            destination_embeddings = mx.nd.Embedding(destinations, embeddings, self.graph_size, self.embed_size)
        else:
            seq_embeddings = self.embeddings(seq)
            neighbor_embeddings = self.embeddings(neighbors)
            destination_embeddings = self.embeddings(destinations)
        return seq_embeddings, neighbor_embeddings, destination_embeddings

    def compute_embeddings(self, embeddings, adjacency_matrix):
        return self.gcn(embeddings, adjacency_matrix)

    def encode_seq(self, inputs, neighbors, destinations, embeddings, valid_length=None):
        """Decode given the input sequence.

        Parameters
        ----------
        inputs : NDArray
        neighbors : NDArray
        destinations : NDArray
        embeddings : NDArray
        valid_length : NDArray or None, default None

        Returns
        -------
        output : NDArray
            The output of the encoder. Shape is (batch_size, length, dim)
        states: list
            The new states of the decoder
        additional_outputs : list
            Additional outputs of the decoder, e.g, the attention weights
        """
        inputs, neighbors, destinations = self.lookup_embeddings(inputs, neighbors, destinations,
                                                                 embeddings)
        if self.concat:
            destinations = mx.nd.broadcast_axes(mx.nd.expand_dims(destinations, axis=1), axis=1, size=inputs.shape[1])
            inputs = mx.nd.concat(inputs, destinations, dim=2)
        else:
            inputs = inputs + mx.nd.expand_dims(destinations, axis=1)
        outputs, states, additional_outputs =\
            self.encoder.encode_seq(inputs=inputs,
                                    valid_length=valid_length)
        if self.concat:
            outputs = self.dense(outputs)
        batch_size = outputs.shape[0]
        outputs = self.proj(outputs, neighbors)
        outputs = outputs.reshape((-4, batch_size, -1, 0))
        return outputs, states, additional_outputs

    def encode_step(self, step_input, neighbors, destinations, embeddings, states):
        """One step encoding of the model.

        Parameters
        ----------
        step_input : NDArray
            Shape (batch_size,)
        neighbors : NDArray
            Shape (batch_size, degree)
        destinations : NDArray
            Shape (batch_size,)
        embeddings : NDArray
        states : list of NDArrays

        Returns
        -------
        step_output : NDArray
            Shape (batch_size, C_out)
        states : list
        step_additional_outputs : list
            Additional outputs of the step, e.g, the attention weights
        """
        step_input, neighbors, destinations = self.lookup_embeddings(step_input, neighbors, destinations,
                                                                     embeddings)
        if self.concat:
            step_input = mx.nd.concat(step_input, destinations, dim=1)
        else:
            step_input = step_input + destinations
        step_output, states, step_additional_outputs =\
            self.encoder(step_input, states)
        if self.concat:
            step_output = self.dense(step_output)
        step_output = self.proj(mx.nd.expand_dims(step_output, axis=1),
                                mx.nd.expand_dims(neighbors, axis=1))
        return step_output, states, step_additional_outputs

    def forward(self, seq, neighbors, destinations, valid_length, embeddings=None):
        outputs, _, additional_outputs = self.encode_seq(seq,
                                                         neighbors,
                                                         destinations,
                                                         embeddings,
                                                         valid_length)
        return outputs, additional_outputs

