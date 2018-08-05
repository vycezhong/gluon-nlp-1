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
"""Encoder and decoder usded in sequence-to-sequence learning."""
__all__ = ['ParallelGNMTEncoder', 'ParallelGNMTDecoder', 'get_parallel_gnmt_encoder_decoder']

import math
import mxnet as mx
from mxnet.base import _as_list
from mxnet.gluon import nn, rnn
from mxnet.gluon.block import HybridBlock
try:
    from encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder, _get_attention_cell, \
        _get_cell_type, _nested_sequence_last, _expand_size, _reshape_size
except ImportError:
    from .encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder, _get_attention_cell, \
        _get_cell_type, _nested_sequence_last, _expand_size, _reshape_size


class ParallelGNMTEncoder(Seq2SeqEncoder):
    r"""Structure of the RNN Encoder similar to that used in
     "[Arxiv2016] Google's Neural Machine Translation System:
                 Bridgeing the Gap between Human and Machine Translation"

    The encoder first stacks several bidirectional RNN layers and then stacks multiple
    uni-directional RNN layers with residual connections.

    Parameters
    ----------
    cell_type : str or function
        Can be "lstm", "gru" or constructor functions that can be directly called,
         like rnn.LSTMCell
    num_layers : int
        Total number of layers
    num_bottom_layers : int
    num_bi_layers : int
        Total number of bidirectional layers
    num_states : int
    hidden_size : int
        Number of hidden units
    dropout : float
        The dropout rate
    use_residual : bool
        Whether to use residual connection. Residual connection will be added in the
        uni-directional RNN layers
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, cell_type='lstm', num_layers=2, num_bi_layers=1, hidden_size=128,
                 num_bottom_layers=1, num_states=5,
                 dropout=0.0, use_residual=True,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 prefix=None, params=None):
        super(ParallelGNMTEncoder, self).__init__(prefix=prefix, params=params)
        self._cell_type = _get_cell_type(cell_type)
        assert num_bi_layers <= num_bottom_layers,\
            'Number of bidirectional layers must be smaller than the total number of bottom layers, ' \
            'num_bi_layers={}, num_bottom_layers={}'.format(num_bi_layers, num_bottom_layers)
        assert num_layers - num_bottom_layers >= 1, 'num_layers={} - num_bottom_layers={} ' \
                                                    'should be larger than or equal to 1'.format(num_layers,
                                                                                                 num_bottom_layers)
        assert hidden_size % num_states == 0, 'In ParallelGNMTEncoder, The hidden_size should be divided exactly' \
                                              ' by the number of states. Received hidden_size={}, num_states={}'\
            .format(hidden_size, num_states)
        self._num_bi_layers = num_bi_layers
        self._num_layers = num_layers
        self._num_bottom_layers = num_bottom_layers
        self._num_states = num_states
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._use_residual = use_residual
        self._scale = math.sqrt(num_states)
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.rnn_cells = nn.HybridSequential()
            hidden_units = hidden_size
            for i in range(num_layers):
                if i == self._num_bottom_layers:
                    hidden_units = round(float(hidden_size) / self._scale)
                if i < num_bi_layers:
                    self.rnn_cells.add(rnn.BidirectionalCell(
                        l_cell=self._cell_type(hidden_size=hidden_units,
                                               i2h_weight_initializer=i2h_weight_initializer,
                                               h2h_weight_initializer=h2h_weight_initializer,
                                               i2h_bias_initializer=i2h_bias_initializer,
                                               h2h_bias_initializer=h2h_bias_initializer,
                                               prefix='rnn%d_l_' % i),
                        r_cell=self._cell_type(hidden_size=hidden_units,
                                               i2h_weight_initializer=i2h_weight_initializer,
                                               h2h_weight_initializer=h2h_weight_initializer,
                                               i2h_bias_initializer=i2h_bias_initializer,
                                               h2h_bias_initializer=h2h_bias_initializer,
                                               prefix='rnn%d_r_' % i)))
                else:
                    self.rnn_cells.add(
                        self._cell_type(hidden_size=hidden_units,
                                        i2h_weight_initializer=i2h_weight_initializer,
                                        h2h_weight_initializer=h2h_weight_initializer,
                                        i2h_bias_initializer=i2h_bias_initializer,
                                        h2h_bias_initializer=h2h_bias_initializer,
                                        prefix='rnn%d_' % i))
            self.transition_cell = nn.Dense(units=hidden_units * self._num_states,
                                            flatten=False, activation='relu')
            self.residual_proj = nn.Dense(units=hidden_units, flatten=False, use_bias=False)
            self.layer_norm = nn.LayerNorm()

    def __call__(self, inputs, states=None, valid_length=None):
        """Encoder the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or None
            Initial states. The list of initial states
        valid_length : NDArray or None
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        encoder_outputs: list
            Outputs of the encoder. Contains:

            - outputs of the last RNN layer
            - new_states of all the RNN layers
        """
        return self.forward(inputs, states, valid_length)

    def forward(self, inputs, states=None, valid_length=None):  #pylint: disable=arguments-differ
        _, length, _ = inputs.shape
        new_states = []
        outputs = inputs
        for i, cell in enumerate(self.rnn_cells):
            begin_state = None if states is None else states[i]
            if i == self._num_bottom_layers:
                transition_out = self.transition_cell(inputs)
                transition_out = self.dropout_layer(transition_out)
                inputs = self.residual_proj(inputs)
                inputs = mx.nd.expand_dims(inputs, 2)
                inputs = self.layer_norm(transition_out.reshape(shape=(0, 0, -4, self._num_states, -1)) + inputs)
                inputs = inputs.transpose(axes=(0, 2, 1, 3)).reshape(shape=(-3, 0, 0))
                if valid_length is not None:
                    valid_length = mx.nd.broadcast_to(mx.nd.expand_dims(valid_length, -1),
                                                      (0, self._num_states))
                    valid_length = valid_length.reshape((-1))
            if begin_state is None:
                begin_state = cell.begin_state(func=mx.nd.zeros, batch_size=inputs.shape[0], ctx=inputs.context)
            outputs, layer_states = cell.unroll(
                length=length, inputs=inputs, begin_state=begin_state, merge_outputs=True,
                valid_length=valid_length, layout='NTC')
            if i < self._num_bi_layers:
                # For bidirectional RNN, we use the states of the backward RNN
                new_states.append(layer_states[len(self.rnn_cells[i].state_info()) // 2:])
            else:
                new_states.append(layer_states)
            # Apply Dropout
            outputs = self.dropout_layer(outputs)
            if self._use_residual:
                if i > self._num_bi_layers:
                    outputs = outputs + inputs
            inputs = outputs
        new_states[self._num_bottom_layers:] = _reshape_size(new_states[self._num_bottom_layers:],
                                                             self._num_states, action='convert_to_tensor')
        if valid_length is not None:
            outputs = mx.nd.SequenceMask(outputs, sequence_length=valid_length,
                                         use_sequence_length=True, axis=1)
        outputs = outputs.reshape(shape=(-4, -1, self._num_states, -2))
        return [outputs, new_states], []


class ParallelGNMTDecoder(HybridBlock, Seq2SeqDecoder):
    """Structure of the RNN Encoder similar to that used in the
    Google Neural Machine Translation paper.

    We use gnmt_v2 strategy in tensorflow/nmt

    Parameters
    ----------
    cell_type : str or type
    attention_cell : AttentionCell or str
        Arguments of the attention cell.
        Can be 'scaled_luong', 'normed_mlp', 'dot'
    num_layers : int
    num_bottom_layers : int
    num_states : int
    hidden_size : int
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, cell_type='lstm', attention_cell='multi_memory',
                 num_layers=2, hidden_size=128, num_bottom_layers=1,
                 scaled=True, num_states=5, dropout=0.0,
                 use_residual=True, output_attention=False,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 prefix=None, params=None):
        super(ParallelGNMTDecoder, self).__init__(prefix=prefix, params=params)
        assert num_layers - num_bottom_layers >= 1, 'num_layers={} - num_bottom_layers={} ' \
                                                    'should be larger than or equal to 1'.format(num_layers,
                                                                                                 num_bottom_layers)
        assert hidden_size % num_states == 0, 'In ParallelGNMTDecoder, the hidden_size should be divided ' \
                                              'exactly by the number of decoder. Received hidden_size={}, ' \
                                              'num_states={}'.format(hidden_size, num_states)
        self._cell_type = _get_cell_type(cell_type)
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._num_bottom_layers = num_bottom_layers
        self._num_states = num_states
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        self._scale = math.sqrt(num_states)
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.rnn_cells = nn.HybridSequential()
            hidden_units = hidden_size
            for i in range(num_layers):
                if i == self._num_bottom_layers:
                    hidden_units = round(float(hidden_size) / self._scale)
                self.rnn_cells.add(
                    self._cell_type(hidden_size=hidden_units,
                                    i2h_weight_initializer=i2h_weight_initializer,
                                    h2h_weight_initializer=h2h_weight_initializer,
                                    i2h_bias_initializer=i2h_bias_initializer,
                                    h2h_bias_initializer=h2h_bias_initializer,
                                    prefix='rnn%d_' % i))
            self.attention_cell = _get_attention_cell(attention_cell, units=hidden_units * self._num_states,
                                                      num_memories=num_states, scaled=scaled, use_bias=True)
            self.transition_cell = nn.Dense(units=hidden_units * self._num_states,
                                            flatten=True, activation='relu')
            # Construct mixture probability proj
            self.mix_proj = nn.Dense(units=num_states, flatten=False, prefix='mix_proj_')
            self.residual_proj = nn.Dense(units=hidden_units, flatten=False, use_bias=False)
            self.layer_norm = nn.LayerNorm()
            self._state_units = hidden_units

    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length=None):
        """Initialize the state from the encoder outputs.

        Parameters
        ----------
        encoder_outputs : list
        encoder_valid_length : NDArray or None

        Returns
        -------
        decoder_states : list
            The decoder states, includes:

            - rnn_states : NDArray
            - attention_vec : NDArray
            - mem_value : NDArray
            - mem_masks : NDArray, optional
        """
        mem_value, rnn_states = encoder_outputs
        batch_size, num_memories, _, mem_size = mem_value.shape
        attention_vec = mx.nd.zeros(shape=(batch_size, num_memories * mem_size),
                                    ctx=mem_value.context)
        decoder_states = [rnn_states, attention_vec, mem_value]
        mem_length = mem_value.shape[2]
        if encoder_valid_length is not None:
            mem_masks = mx.nd.broadcast_lesser(
                mx.nd.arange(mem_length, ctx=encoder_valid_length.context).reshape((1, -1)),
                encoder_valid_length.reshape((-1, 1)))
            decoder_states.append(mem_masks)
        return decoder_states

    def decode_seq(self, inputs, states, valid_length=None):
        length = inputs.shape[1]
        output = []
        additional_outputs = []
        inputs = _as_list(mx.nd.split(inputs, num_outputs=length, axis=1, squeeze_axis=True))
        rnn_states_b = []
        rnn_states_t = []
        attention_output_l = []
        attention_weight_l = []
        mix_prob_l = []
        fixed_states = states[2:]
        for i in range(length):
            ele_output, states, ele_additional_outputs = self.forward(inputs[i], states)
            rnn_states_b.append(states[0][:self._num_bottom_layers])
            rnn_states_t.append(states[0][self._num_bottom_layers:])
            attention_output_l.append(states[1])
            output.append(ele_output)
            mix_prob_l.append(ele_additional_outputs[0])
            attention_weight_l.extend(ele_additional_outputs[1:])
        output = mx.nd.stack(*output, axis=1)
        mix_prob = mx.nd.stack(*mix_prob_l, axis=1)
        if valid_length is not None:
            augmented_valid_length = mx.nd.broadcast_to(mx.nd.expand_dims(valid_length, -1), (0, self._num_states))
            augmented_valid_length = augmented_valid_length.reshape((-1))
            states = [_nested_sequence_last(rnn_states_b, valid_length) +
                      _nested_sequence_last(rnn_states_t, valid_length),
                      _nested_sequence_last(attention_output_l, valid_length)] + fixed_states
            output = mx.nd.SequenceMask(output,
                                        sequence_length=augmented_valid_length,
                                        use_sequence_length=True,
                                        axis=1)
        additional_outputs.append(mix_prob)
        if self._output_attention:
            additional_outputs.append(mx.nd.concat(*attention_weight_l, dim=-2))
        return output, states, additional_outputs

    def __call__(self, step_input, states):
        """One-step-ahead decoding of the Parallel GNMT decoder.

        Parameters
        ----------
        step_input : NDArray or Symbol
        states : NDArray or Symbol

        Returns
        -------
        step_output : NDArray or Symbol
            The output of the decoder. Shape is (batch_size, C_out)
        new_states: list
            Includes

            - rnn_states : list of NDArray or Symbol
            - attention_vec : NDArray or Symbol, Shape (batch_size, C_memory)
            - mem_value : NDArray
            - mem_masks : NDArray, optional

        step_additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, 1, mem_length) or
            (batch_size, num_heads, 1, mem_length)
        """
        return self.forward(step_input, states)

    def forward(self, step_input, states):  #pylint: disable=arguments-differ
        step_output, new_states, step_additional_outputs =\
            super(ParallelGNMTDecoder, self).forward(step_input, states)
        # In hybrid_forward, only the rnn_states and attention_vec are calculated.
        # We directly append the mem_value and mem_masks in the forward() function.
        # We apply this trick because the memory value/mask can be directly appended to the next
        # timestamp and there is no need to create additional NDArrays. If we use HybridBlock,
        # new NDArrays will be created even for identity mapping.
        # See https://github.com/apache/incubator-mxnet/issues/10167
        new_states += states[2:]
        return step_output, new_states, step_additional_outputs

    def hybrid_forward(self, F, step_input, states):  #pylint: disable=arguments-differ
        """

        Parameters
        ----------
        step_input : NDArray or Symbol
        states : NDArray or Symbol

        Returns
        -------
        step_output : NDArray or Symbol
            The output of the decoder. Shape is (batch_size * num_states, C_out)
        new_states: list
            Includes

            - rnn_states : list of NDArray or Symbol
            - attention_vec : NDArray or Symbol, Shape (batch_size, C_memory)

        step_additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, 1, mem_length) or
            (batch_size, num_heads, 1, mem_length)

        """
        has_mem_mask = (len(states) == 4)
        if has_mem_mask:
            rnn_states, attention_output, mem_value, mem_masks = states
            mem_masks = F.expand_dims(mem_masks, axis=1)
        else:
            rnn_states, attention_output, mem_value = states
            mem_masks = None
        new_rnn_states = []
        # Process the first layer
        rnn_out, layer_state =\
            self.rnn_cells[0](F.concat(step_input, attention_output, dim=-1), rnn_states[0])

        new_rnn_states.append(layer_state)
        attention_vec, attention_weights =\
            self.attention_cell(F.expand_dims(rnn_out, axis=1),  # Shape(B, 1, C)
                                mem_value,
                                mem_value,
                                mem_masks)
        attention_vec = attention_vec.reshape(shape=(0, -1))
        rnn_states[self._num_bottom_layers:] = _reshape_size(rnn_states[self._num_bottom_layers:],
                                                             self._num_states, action='convert_to_batch')
        step_additional_outputs = []
        # Process the rest bottom layers
        for i in range(1, self._num_layers):
            if i == self._num_bottom_layers:
                # Compute mixture probabilities
                mix_prob = self.mix_proj(F.concat(rnn_out, attention_vec, dim=-1))
                step_additional_outputs.append(mix_prob)
                # Apply transition cell to obtain K states
                curr_input = rnn_out
                rnn_out = self.transition_cell(curr_input)
                rnn_out = self.dropout_layer(rnn_out)
                rnn_out = rnn_out.reshape(shape=(0, -4, self._num_states, -1))
                curr_input = self.residual_proj(curr_input)
                curr_input = F.expand_dims(curr_input, 1).broadcast_axes(axis=1, size=self._num_states)
                rnn_out = self.layer_norm(rnn_out + curr_input)
                # Shape(batch_size * num_states, round(hidden_size / scale))
                rnn_out = rnn_out.reshape(shape=(-3, 0))
                # Shape(batch_size * num_states, round(hidden_size / scale))
                attention_vec = attention_vec.reshape(shape=(-1, self._state_units))

            curr_input = rnn_out
            rnn_cell = self.rnn_cells[i]
            # Concatenate the attention vector calculated by the bottom layer and the output of the
            # previous layer
            rnn_out, layer_state = rnn_cell(F.concat(curr_input, attention_vec, dim=-1),
                                            rnn_states[i])
            rnn_out = self.dropout_layer(rnn_out)
            if self._use_residual:
                rnn_out = rnn_out + curr_input
            # Append new RNN state
            new_rnn_states.append(layer_state)

        attention_vec = attention_vec.reshape(shape=(-1, self._state_units * self._num_states))
        new_rnn_states[self._num_bottom_layers:] = _reshape_size(new_rnn_states[self._num_bottom_layers:],
                                                                 self._num_states, action='convert_to_tensor')
        new_states = [new_rnn_states, attention_vec]
        if self._output_attention:
            step_additional_outputs.append(attention_weights)
        return rnn_out, new_states, step_additional_outputs


def get_parallel_gnmt_encoder_decoder(cell_type='lstm', attention_cell='multi_memory', num_layers=2,
                                      num_bottom_layers=1, num_states=4, num_bi_layers=1, scaled=True,
                                      hidden_size=128, dropout=0.0, use_residual=True,
                                      i2h_weight_initializer=None, h2h_weight_initializer=None,
                                      i2h_bias_initializer=mx.init.LSTMBias(forget_bias=1.0),
                                      h2h_bias_initializer='zeros',
                                      prefix='parallel_gnmt_', params=None):
    """Build a pair of Parallel GNMT encoder/decoder

    Parameters
    ----------
    cell_type : str or type
    attention_cell : str or AttentionCell
    num_layers : int
    num_bottom_layers : int
    num_states : int
    num_bi_layers : int
    hidden_size : int
    dropout : float
    use_residual : bool
    use_residual_proj : bool
    i2h_weight_initializer : mx.init.Initializer or None
    h2h_weight_initializer : mx.init.Initializer or None
    i2h_bias_initializer : mx.init.Initializer or None
    h2h_bias_initializer : mx.init.Initializer or None
    prefix :
    params :

    Returns:
    -------
    encoder : GNMTEncoder
    decoder : ParallelGNMTDecoder
    """
    encoder = ParallelGNMTEncoder(cell_type=cell_type, num_layers=num_layers, num_bi_layers=num_bi_layers,
                                  num_bottom_layers=num_bottom_layers, num_states=num_states,
                                  hidden_size=hidden_size, dropout=dropout,
                                  use_residual=use_residual,
                                  i2h_weight_initializer=i2h_weight_initializer,
                                  h2h_weight_initializer=h2h_weight_initializer,
                                  i2h_bias_initializer=i2h_bias_initializer,
                                  h2h_bias_initializer=h2h_bias_initializer,
                                  prefix=prefix + 'enc_', params=params)
    decoder = ParallelGNMTDecoder(cell_type=cell_type, attention_cell=attention_cell, num_layers=num_layers,
                                  num_bottom_layers=num_bottom_layers, num_states=num_states,
                                  hidden_size=hidden_size, dropout=dropout,
                                  scaled=scaled,
                                  use_residual=use_residual,
                                  i2h_weight_initializer=i2h_weight_initializer,
                                  h2h_weight_initializer=h2h_weight_initializer,
                                  i2h_bias_initializer=i2h_bias_initializer,
                                  h2h_bias_initializer=h2h_bias_initializer,
                                  prefix=prefix + 'dec_', params=params)
    return encoder, decoder
