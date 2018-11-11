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

import functools
import mxnet as mx
from mxnet.gluon import nn, rnn, Block, HybridBlock

__all__ = ['RNN']


def _get_rnn_layer(mode, num_layers, hidden_size, dropout):
    """create rnn layer given specs"""
    if mode == 'rnn_relu':
        rnn_block = functools.partial(rnn.RNN, activation='relu')
    elif mode == 'rnn_tanh':
        rnn_block = functools.partial(rnn.RNN, activation='tanh')
    elif mode == 'lstm':
        rnn_block = rnn.LSTM
    elif mode == 'gru':
        rnn_block = rnn.GRU

    block = rnn_block(hidden_size, num_layers, dropout=dropout)

    return block


class RNN(Block):
    """Standard RNN model.

    Parameters
    ----------
    mode : str
        The type of RNN to use. Options are 'lstm', 'gru', 'rnn_tanh', 'rnn_relu'.
    hidden_size : int
        Number of hidden units for RNN.
    num_layers : int
        Number of RNN layers.
    dropout : float
        Dropout rate to use for encoder output.
    """
    def __init__(self, mode, hidden_size, num_layers,
                 dropout=0.0, **kwargs):
        super(RNN, self).__init__(**kwargs)
        self._mode = mode
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        with self.name_scope():
            self.encoder = self._get_encoder()

    def _get_encoder(self):
        return _get_rnn_layer(self._mode, self._num_layers,
                              self._hidden_size, self._dropout)

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

    def state_info(self, *args, **kwargs):
        return self.encoder.state_info(*args, **kwargs)

    def encode_seq(self, inputs, valid_length=None):
        inputs = inputs.transpose((1, 0, 2))
        output, states, additional_outputs = self.forward(inputs)
        output = output.transpose((1, 0, 2))
        if valid_length is not None:
            output = mx.nd.SequenceMask(output,
                                        sequence_length=valid_length,
                                        use_sequence_length=True,
                                        axis=1)
        return output, states, additional_outputs

    def __call__(self, step_input, states):
        step_input = mx.nd.expand_dims(step_input, axis=0)
        output, states, additional_outputs = super(RNN, self).__call__(step_input, states)
        output = mx.nd.squeeze(output, axis=0)
        return output, states, additional_outputs

    def forward(self, inputs, begin_state=None): # pylint: disable=arguments-differ
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`.

        Parameters
        -----------
        inputs : NDArray
            input tensor with shape `(batch_size, sequence_length, input_size)`
            when `layout` is "TNC".
        begin_state : list
            initial recurrent state tensor with length equals to num_layers.
            the initial state with shape `(batch_size, num_layers, num_hidden)`

        Returns
        --------
        encoded: NDArray
            output tensor with shape `(batch_size, sequence_length, input_size)`
            when `layout` is "TNC".
        out_states: list
            output recurrent state tensor with length equals to num_layers-1.
            the state with shape `(num_layers, batch_size, num_hidden)`
        """
        if not begin_state:
            begin_state = self.begin_state(batch_size=inputs.shape[1])
        encoded, state = self.encoder(inputs, begin_state)
        if self._dropout:
            encoded = mx.nd.Dropout(encoded, p=self._dropout, axes=(0,))
        return encoded, state, []