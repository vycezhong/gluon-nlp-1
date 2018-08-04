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
__all__ = ['ParallelTransformerEncoder', 'ParallelTransformerDecoder', 'get_parallel_transformer_encoder_decoder']

import math
import mxnet as mx
from mxnet.gluon import nn, rnn
from mxnet.gluon.block import HybridBlock
try:
    from encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder, _get_attention_cell, \
        _get_cell_type, _nested_sequence_last, _expand_size, _reshape_size
    from transformer import TransformerEncoderCell, PositionwiseFFN, _position_encoding_init
except ImportError:
    from .encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder, _get_attention_cell, \
        _get_cell_type, _nested_sequence_last, _expand_size, _reshape_size
    from .transformer import TransformerEncoderCell, PositionwiseFFN, _position_encoding_init


class TransformerDecoderCell(HybridBlock):
    def __init__(self, units=128, hidden_size=512, num_heads=4,
                 attention_cell_in='multi_head',
                 attention_cell_inter='multi_memory', scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TransformerDecoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        self._scaled = scaled
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.attention_cell_in = _get_attention_cell(attention_cell_in, units=units,
                                                         num_heads=num_heads, num_memories=num_heads,
                                                         scaled=scaled, dropout=dropout)
            self.attention_cell_inter = _get_attention_cell(attention_cell_inter, units=units,
                                                            num_heads=num_heads, num_memories=num_heads,
                                                            scaled=scaled, dropout=dropout)
            self.proj_in = nn.Dense(units=units, flatten=False,
                                    use_bias=False,
                                    weight_initializer=weight_initializer,
                                    bias_initializer=bias_initializer,
                                    prefix='proj_in_')
            self.proj_inter = nn.Dense(units=units, flatten=False,
                                       use_bias=False,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       prefix='proj_inter_')
            self.ffn = PositionwiseFFN(hidden_size=hidden_size,
                                       units=units,
                                       use_residual=use_residual,
                                       dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)

            self.layer_norm_in = nn.LayerNorm()
            self.layer_norm_inter = nn.LayerNorm()

    def hybrid_forward(self, F, inputs, mem_value, mask=None, mem_mask=None):
        #  pylint: disable=arguments-differ
        """Transformer Decoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        mem_value : Symbol or NDArrays
            Memory value, i.e. output of the encoder. Shape (batch_size, mem_length, C_in)
        mask : Symbol or NDArray or None
            Mask for inputs. Shape (batch_size, length, length)
        mem_mask : Symbol or NDArray or None
            Mask for mem_value. Shape (batch_size, length, mem_length)

        Returns
        -------
        decoder_cell_outputs: list
            Outputs of the decoder cell. Contains:

            - outputs of the transformer decoder cell. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer decoder cell
        """
        outputs, attention_in_outputs = \
            self.attention_cell_in(inputs, inputs, inputs, mask)
        outputs = self.proj_in(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm_in(outputs)
        inputs = outputs
        outputs, attention_inter_outputs = \
            self.attention_cell_inter(inputs, mem_value, mem_value, mem_mask)
        outputs = self.proj_inter(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm_inter(outputs)
        outputs = self.ffn(outputs)
        additional_outputs = []
        if self._output_attention:
            additional_outputs.append(attention_in_outputs)
            additional_outputs.append(attention_inter_outputs)
        return outputs, additional_outputs


class ParallelTransformerEncoder(HybridBlock, Seq2SeqEncoder):
    def __init__(self, attention_cell='multi_head', num_layers=2,
                 units=128, hidden_size=512, max_length=50,
                 num_bottom_layers=1, num_states=4, scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(ParallelTransformerEncoder, self).__init__(prefix=prefix, params=params)
        assert num_layers - num_bottom_layers >= 1, 'num_layers={} - num_bottom_layers={} ' \
                                                    'should be larger than or equal to 1'.format(num_layers,
                                                                                                 num_bottom_layers)
        assert units % num_states == 0, 'In ParallelTransformerEncoder, The units should be divided exactly' \
                                        ' by the number of states. Received units={}, num_states={}' \
            .format(units, num_states)
        self._num_layers = num_layers
        self._num_bottom_layers = num_bottom_layers
        self._num_states = num_states
        self._max_length = max_length
        self._units = units
        self._hidden_size = hidden_size
        self._output_attention = output_attention
        self._dropout = dropout
        self._use_residual = use_residual
        self._scaled = scaled
        self._scale = math.sqrt(num_states)
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.pre_layer_norm = nn.LayerNorm()
            self.layer_norm = nn.LayerNorm()
            self.position_weight = self.params.get_constant('const',
                                                            _position_encoding_init(max_length,
                                                                                    units))
            self.transformer_cells = nn.HybridSequential()
            curr_units = units
            curr_hidden_size = hidden_size
            for i in range(num_layers):
                if i == self._num_bottom_layers:
                    attention_cell = 'scaled_luong'
                    curr_units = round(units / self._scale)
                    curr_hidden_size = round(hidden_size / self._scale)
                self.transformer_cells.add(TransformerEncoderCell(units=curr_units,
                                                                  hidden_size=curr_hidden_size,
                                                                  num_heads=num_states,
                                                                  dropout=dropout,
                                                                  attention_cell=attention_cell,
                                                                  weight_initializer=weight_initializer,
                                                                  bias_initializer=bias_initializer,
                                                                  use_residual=use_residual,
                                                                  scaled=scaled,
                                                                  output_attention=output_attention,
                                                                  prefix='transformer%d_' % i))
            self.transition_cell = nn.Dense(units=curr_units * self._num_states,
                                            flatten=False, activation='relu', prefix='transition_')
            self.residual_proj = nn.Dense(units=curr_units, flatten=False, prefix='residual_proj_', use_bias=False)

    def __call__(self, inputs, states=None, valid_length=None):
        """Encoder the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or None
            Initial states. The list of initial states and masks
        valid_length : NDArray or None
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        encoder_outputs: list
            Outputs of the encoder. Contains:

            - outputs of the parallel transformer encoder. Shape (batch_size, num_state, length, C_out)
            - additional_outputs of all the parallel transformer encoder
        """
        return self.forward(inputs, states, valid_length)

    def forward(self, inputs, states=None, valid_length=None, steps=None):  # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray, Shape(batch_size, length, C_in)
        states : list of NDArray
        valid_length : NDArray
        steps : NDArray
            Stores value [0, 1, ..., length].
            It is used for lookup in positional encoding matrix

        Returns
        -------
        outputs : NDArray
            The output of the encoder. Shape is (batch_size, length, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, length) or
            (batch_size, num_heads, length, length)

        """
        length = inputs.shape[1]
        if valid_length is not None:
            mask = mx.nd.broadcast_lesser(
                mx.nd.arange(length, ctx=valid_length.context).reshape((1, -1)),
                valid_length.reshape((-1, 1)))
            mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=1), axis=1, size=length)
            if states is None:
                states = [mask]
            else:
                states.append(mask)
        inputs = inputs * math.sqrt(inputs.shape[-1])
        steps = mx.nd.arange(length, ctx=inputs.context)
        if states is None:
            states = [steps]
        else:
            states.append(steps)
        if valid_length is not None:
            step_output, additional_outputs = \
                super(ParallelTransformerEncoder, self).forward(inputs, states, valid_length)
        else:
            step_output, additional_outputs = \
                super(ParallelTransformerEncoder, self).forward(inputs, states)
        return step_output, additional_outputs

    def hybrid_forward(self, F, inputs, states=None, valid_length=None, position_weight=None):  # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray or Symbol, Shape(batch_size, length, C_in)
        states : list of NDArray or Symbol
        valid_length : NDArray or Symbol
        position_weight : NDArray or Symbol

        Returns
        -------
        outputs : NDArray or Symbol
            The output of the encoder. Shape is (batch_size, length, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, length) or
            (batch_size, num_heads, length, length)

        """
        if states is not None:
            steps = states[-1]
            inputs = F.broadcast_add(inputs, F.expand_dims(F.Embedding(steps, position_weight,
                                                                       self._max_length,
                                                                       self._units), axis=0))
        inputs = self.dropout_layer(inputs)
        inputs = self.pre_layer_norm(inputs)
        outputs = inputs
        if valid_length is not None:
            mask = states[-2]
        else:
            mask = None
        additional_outputs = []
        for i, cell in enumerate(self.transformer_cells):
            if i == self._num_bottom_layers:
                transition_out = self.transition_cell(inputs)
                transition_out = self.dropout_layer(transition_out)
                if self._use_residual:
                    inputs = self.residual_proj(inputs)
                    inputs = F.expand_dims(inputs, 2)
                    inputs = F.broadcast_add(transition_out.reshape(shape=(0, 0, -4, self._num_states, -1)), inputs)
                else:
                    inputs = transition_out.reshape(shape=(0, 0, -4, self._num_states, -1))
                inputs = self.layer_norm(inputs)
                inputs = inputs.transpose(axes=(0, 2, 1, 3)).reshape(shape=(-3, -2))
                if valid_length is not None:
                    valid_length = F.broadcast_to(F.expand_dims(valid_length, -1), (0, self._num_states))
                    valid_length = valid_length.reshape(shape=(-1,))
                    mask = F.broadcast_to(F.expand_dims(mask, 1), (0, self._num_states, 0, 0))
                    mask = mask.reshape(shape=(-3, -2))
            outputs, attention_weights = cell(inputs, mask)
            inputs = outputs
            if self._output_attention:
                additional_outputs.append(attention_weights)
        if valid_length is not None:
            outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                     use_sequence_length=True, axis=1)
        outputs = outputs.reshape(shape=(-4, -1, self._num_states, -2))
        return outputs, additional_outputs


class ParallelTransformerDecoder(HybridBlock, Seq2SeqDecoder):
    def __init__(self, attention_cell_in='multi_head',
                 attention_cell_inter='multi_memory', num_layers=2,
                 units=128, hidden_size=512,
                 max_length=50, num_bottom_layers=1, num_states=4, scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(ParallelTransformerDecoder, self).__init__(prefix=prefix, params=params)
        assert num_layers - num_bottom_layers >= 1, 'num_layers={} - num_bottom_layers={} ' \
                                                    'should be larger than or equal to 1'.format(num_layers,
                                                                                                 num_bottom_layers)
        assert units % num_states == 0, 'In ParallelTransformerDecoder, the units should be divided ' \
                                        'exactly by the number of states. Received units={}, ' \
                                        'num_states={}'.format(units, num_states)
        self._num_layers = num_layers
        self._units = units
        self._hidden_size = hidden_size
        self._num_bottom_layers = num_bottom_layers
        self._num_states = num_states
        self._max_length = max_length
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        self._scaled = scaled
        self._scale = math.sqrt(num_states)
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.pre_layer_norm = nn.LayerNorm()
            self.layer_norm = nn.LayerNorm()
            self.position_weight = self.params.get_constant('const',
                                                            _position_encoding_init(max_length,
                                                                                    units))
            self.transformer_cells = nn.HybridSequential()
            curr_units = units
            curr_hidden_size = hidden_size
            for i in range(num_layers):
                if i == self._num_bottom_layers:
                    curr_units = round(units / self._scale)
                    curr_hidden_size = round(hidden_size / self._scale)
                    attention_cell_in = 'scaled_luong'
                    attention_cell_inter = 'scaled_luong'
                self.transformer_cells.add(TransformerDecoderCell(units=curr_units,
                                                                  hidden_size=curr_hidden_size,
                                                                  num_heads=num_states,
                                                                  attention_cell_in=attention_cell_in,
                                                                  attention_cell_inter=attention_cell_inter,
                                                                  weight_initializer=weight_initializer,
                                                                  bias_initializer=bias_initializer,
                                                                  dropout=dropout,
                                                                  scaled=scaled,
                                                                  use_residual=use_residual,
                                                                  output_attention=output_attention,
                                                                  prefix='transformer%d_' % i))
            self.transition_cell = nn.Dense(units=curr_units * self._num_states,
                                            flatten=False, activation='relu', prefix='transition_')
            self.residual_proj = nn.Dense(units=curr_units, flatten=False,
                                          prefix='residual_proj_', use_bias=False)
            self.mix_proj = nn.Dense(units=num_states, flatten=False, prefix='mix_proj_')

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

            - mem_value : NDArray
            - mem_masks : NDArray, optional
        """
        mem_value = encoder_outputs
        decoder_states = [mem_value]
        mem_length = mem_value.shape[2]
        if encoder_valid_length is not None:
            mem_masks = mx.nd.broadcast_lesser(
                mx.nd.arange(mem_length, ctx=encoder_valid_length.context).reshape((1, -1)),
                encoder_valid_length.reshape((-1, 1)))
            decoder_states.append(mem_masks)
        self._encoder_valid_length = encoder_valid_length
        return decoder_states

    def decode_seq(self, inputs, states, valid_length=None):
        """Decode the decoder inputs. This function is only used for training.

         Parameters
         ----------
         inputs : NDArray, Shape (batch_size, length, C_in)
         states : list of NDArrays or None
             Initial states. The list of decoder states
         valid_length : NDArray or None
             Valid lengths of each sequence. This is usually used when part of sequence has
             been padded. Shape (batch_size,)

         Returns
         -------
         output : NDArray, Shape (batch_size, length, C_out)
         states : list
             The decoder states, includes:

             - mem_value : NDArray
             - mem_masks : NDArray, optional
         additional_outputs : list of list
             Either be an empty list or contains the attention weights in this step.
             The attention weights will have shape (batch_size, length, mem_length) or
             (batch_size, num_heads, length, mem_length)
         """
        batch_size = inputs.shape[0]
        length = inputs.shape[1]
        length_array = mx.nd.arange(length, ctx=inputs.context)
        mask = mx.nd.broadcast_lesser_equal(
            length_array.reshape((1, -1)),
            length_array.reshape((-1, 1)))
        if valid_length is not None:
            batch_mask = mx.nd.broadcast_lesser(
                mx.nd.arange(length, ctx=valid_length.context).reshape((1, -1)),
                valid_length.reshape((-1, 1)))
            mask = mx.nd.broadcast_mul(mx.nd.expand_dims(batch_mask, -1),
                                       mx.nd.expand_dims(mask, 0))
        else:
            mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=0), axis=0, size=batch_size)
        states = [None] + states
        output, states, additional_outputs = self.forward(inputs, states, mask)
        states = states[1:]
        if valid_length is not None:
            augmented_valid_length = mx.nd.broadcast_to(mx.nd.expand_dims(valid_length, -1), (0, self._num_states))
            augmented_valid_length = augmented_valid_length.reshape((-1))
            output = mx.nd.SequenceMask(output,
                                        sequence_length=augmented_valid_length,
                                        use_sequence_length=True,
                                        axis=1)
        return output, states, additional_outputs

    def __call__(self, step_input, states, mask=None):
        """One-step-ahead decoding of the Parallel GNMT decoder.

        Parameters
        ----------
        step_input : NDArray or Symbol
        states : NDArray or Symbol
        mask : NDArray or Symbol

        Returns
        -------
        step_output : NDArray or Symbol
            The output of the decoder. Shape is (batch_size * num_states, query_length, C_out)
        new_states: list
            Includes
             - last_embeds : NDArray or None
                None in training. It is only given during testing
            - mem_value : NDArray
            - mem_masks : NDArray, optional

        step_additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, 1, mem_length) or
            (batch_size, num_heads, 1, mem_length)
        """
        return self.forward(step_input, states, mask)

    def forward(self, step_input, states, mask=None):  #pylint: disable=arguments-differ
        input_shape = step_input.shape
        mem_mask = None
        if len(input_shape) == 2:
            if self._encoder_valid_length is not None:
                has_last_embeds = len(states) == 3
            else:
                has_last_embeds = len(states) == 2
            if has_last_embeds:
                last_embeds = states[0]
                step_input = mx.nd.concat(last_embeds,
                                          mx.nd.expand_dims(step_input, axis=1),
                                          dim=1)
                states = states[1:]
            else:
                step_input = mx.nd.expand_dims(step_input, axis=1)
        elif states[0] is None:
            states = states[1:]
        has_mem_mask = (len(states) == 2)
        if has_mem_mask:
            _, mem_mask = states
            augmented_mem_mask = mx.nd.expand_dims(mem_mask, axis=1)\
                .broadcast_axes(axis=1, size=step_input.shape[1])
            states[-1] = augmented_mem_mask
        if mask is None:
            length_array = mx.nd.arange(step_input.shape[1], ctx=step_input.context)
            mask = mx.nd.broadcast_lesser_equal(
                length_array.reshape((1, -1)),
                length_array.reshape((-1, 1)))
            mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=0),
                                        axis=0, size=step_input.shape[0])
        steps = mx.nd.arange(step_input.shape[1], ctx=step_input.context)
        states.append(steps)
        step_output, step_additional_outputs = \
            super(ParallelTransformerDecoder, self).forward(step_input * math.sqrt(step_input.shape[-1]),
                                                            states, mask)
        states = states[:-1]
        if has_mem_mask:
            states[-1] = mem_mask
        new_states = [step_input] + states
        if len(input_shape) == 2:
            step_output = step_output[:, -1, :]
            step_additional_outputs[0] = step_additional_outputs[0][:, -1, :]
        return step_output, new_states, step_additional_outputs

    def hybrid_forward(self, F, step_input, states, mask=None, position_weight=None):  #pylint: disable=arguments-differ
        """

        Parameters
        ----------
        step_input : NDArray or Symbol, Shape(batch_size, query_length, C_in)
        states : NDArray or Symbol
        mask : NDArray or Symbol

        Returns
        -------
        step_output : NDArray or Symbol
            The output of the decoder. Shape is (batch_size * num_states, query_length, C_out)
        step_additional_outputs : list
            Either be an empty list or contains the attention weights and mix probability in this step.
            The attention weights will have shape (batch_size, 1, mem_length) or
            (batch_size, num_memories, 1, mem_length)

        """
        has_mem_mask = (len(states) == 3)
        if has_mem_mask:
            mem_value, mem_mask, steps = states
        else:
            mem_value, steps = states
            mem_mask = None
        step_input = F.broadcast_add(step_input,
                                     F.expand_dims(F.Embedding(steps,
                                                               position_weight,
                                                               self._max_length,
                                                               self._units),
                                                   axis=0))
        step_input = self.dropout_layer(step_input)
        step_input = self.pre_layer_norm(step_input)
        inputs = step_input
        outputs = inputs
        step_additional_outputs = []
        attention_weights_l = []
        for i, cell in enumerate(self.transformer_cells):
            if i == self._num_bottom_layers:
                # Compute mixture probabilities
                mix_prob = self.mix_proj(inputs)
                step_additional_outputs.append(mix_prob)
                # Apply transition cell to obtain K states
                transition_out = self.transition_cell(inputs)
                transition_out = self.dropout_layer(transition_out)
                if self._use_residual:
                    inputs = self.residual_proj(inputs)
                    inputs = F.expand_dims(inputs, 2)
                    inputs = F.broadcast_add(transition_out.reshape(shape=(0, 0, -4, self._num_states, -1)), inputs)
                else:
                    inputs = transition_out.reshape(shape=(0, 0, -4, self._num_states, -1))
                inputs = self.layer_norm(inputs)
                inputs = inputs.transpose(axes=(0, 2, 1, 3)).reshape(shape=(-3, 0, 0))
                mem_value = mem_value.reshape(shape=(-3, 0, 0))
                if mask is not None:
                    mask = F.expand_dims(mask, 1).broadcast_axes(axis=1, size=self._num_states)\
                        .reshape(shape=(-3, 0, 0))
                if mem_mask is not None:
                    mem_mask = F.expand_dims(mem_mask, 1).broadcast_axes(axis=1, size=self._num_states)\
                        .reshape(shape=(-3, 0, 0))
            outputs, attention_weights = cell(inputs, mem_value, mask, mem_mask)
            if self._output_attention:
                attention_weights_l.append(attention_weights)
            inputs = outputs
        if self._output_attention:
            step_additional_outputs.extend(attention_weights_l)
        return outputs, step_additional_outputs


def get_parallel_transformer_encoder_decoder(num_layers=2, num_bottom_layers=4, num_states=8, scaled=True,
                                             units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                                             max_src_length=50, max_tgt_length=50,
                                             weight_initializer=None, bias_initializer='zeros',
                                             prefix='parallel_transformer_', params=None):
    """Build a pair of Parallel GNMT encoder/decoder

    Parameters
    ----------
    num_layers : int
    num_bottom_layers : int
    num_states : int
    units : int
    hidden_size : int
    dropout : float
    use_residual : bool
    use_residual_proj : bool
    weight_initializer : mx.init.Initializer or None
    bias_initializer : mx.init.Initializer or None
    prefix :
    params :

    Returns
    -------
    encoder : GNMTEncoder
    decoder : ParallelGNMTDecoder
    """
    encoder = ParallelTransformerEncoder(num_layers=num_layers,
                                         num_bottom_layers=num_bottom_layers, num_states=num_states,
                                         units=units,
                                         hidden_size=hidden_size,
                                         dropout=dropout,
                                         scaled=scaled,
                                         max_length=max_src_length,
                                         use_residual=use_residual,
                                         weight_initializer=weight_initializer,
                                         bias_initializer=bias_initializer,
                                         prefix=prefix + 'enc_', params=params)
    decoder = ParallelTransformerDecoder(num_layers=num_layers,
                                         num_bottom_layers=num_bottom_layers, num_states=num_states,
                                         units=units,
                                         hidden_size=hidden_size,
                                         dropout=dropout,
                                         max_length=max_tgt_length,
                                         scaled=scaled,
                                         use_residual=use_residual,
                                         weight_initializer=weight_initializer,
                                         bias_initializer=bias_initializer,
                                         prefix=prefix + 'dec_', params=params)
    return encoder, decoder
