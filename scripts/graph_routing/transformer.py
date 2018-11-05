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

import math
import numpy as np
import mxnet as mx
from mxnet.gluon import nn, Block, HybridBlock
from gluonnlp.model import AttentionCell, MLPAttentionCell, DotProductAttentionCell, \
    MultiHeadAttentionCell

__all__ = ['TransformerEncoderCell', 'TransformerEncoder', 'PositionwiseFFN']


def _get_attention_cell(attention_cell, units=None,
                        scaled=True, num_heads=None,
                        use_bias=False, dropout=0.0):
    """

    Parameters
    ----------
    attention_cell : AttentionCell or str
    units : int or None

    Returns
    -------
    attention_cell : AttentionCell
    """
    if isinstance(attention_cell, str):
        if attention_cell == 'scaled_luong':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=True)
        elif attention_cell == 'scaled_dot':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'dot':
            return DotProductAttentionCell(units=units, scaled=False, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'cosine':
            return DotProductAttentionCell(units=units, scaled=False, use_bias=use_bias,
                                           dropout=dropout, normalized=True)
        elif attention_cell == 'mlp':
            return MLPAttentionCell(units=units, normalized=False)
        elif attention_cell == 'normed_mlp':
            return MLPAttentionCell(units=units, normalized=True)
        elif attention_cell == 'multi_head':
            base_cell = DotProductAttentionCell(scaled=scaled, dropout=dropout)
            return MultiHeadAttentionCell(base_cell=base_cell, query_units=units, use_bias=use_bias,
                                          key_units=units, value_units=units, num_heads=num_heads)
        else:
            raise NotImplementedError
    else:
        assert isinstance(attention_cell, AttentionCell),\
            'attention_cell must be either string or AttentionCell. Received attention_cell={}'\
                .format(attention_cell)
        return attention_cell


def _position_encoding_init(max_length, dim):
    """ Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    # Apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return position_enc


class PositionwiseFFN(HybridBlock):
    """Structure of the Positionwise Feed-Forward Neural Network.

    Parameters
    ----------
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    dropout : float
    use_residual : bool
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    activation : str, default 'relu'
        Activation function
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                 weight_initializer=None, bias_initializer='zeros', activation='relu',
                 prefix=None, params=None):
        super(PositionwiseFFN, self).__init__(prefix=prefix, params=params)
        self._hidden_size = hidden_size
        self._units = units
        self._use_residual = use_residual
        with self.name_scope():
            self.ffn_1 = nn.Dense(units=hidden_size, flatten=False,
                                  activation=activation,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_1_')
            self.ffn_2 = nn.Dense(units=units, flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_2_')
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Position-wise encoding of the inputs.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)

        Returns
        -------
        outputs : Symbol or NDArray
            Shape (batch_size, length, C_out)
        """
        outputs = self.ffn_1(inputs)
        outputs = self.ffn_2(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        return outputs


class TransformerEncoderCell(HybridBlock):
    """Structure of the Transformer Encoder Cell.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, attention_cell='multi_head', units=128,
                 hidden_size=512, num_heads=4, scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TransformerEncoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.attention_cell = _get_attention_cell(attention_cell,
                                                      units=units,
                                                      num_heads=num_heads,
                                                      scaled=scaled,
                                                      dropout=dropout)
            self.proj = nn.Dense(units=units, flatten=False, use_bias=False,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix='proj_')
            self.ffn = PositionwiseFFN(hidden_size=hidden_size, units=units,
                                       use_residual=use_residual, dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs, mask=None):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Transformer Encoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        mask : Symbol or NDArray or None
            Mask for inputs. Shape (batch_size, length, length)

        Returns
        -------
        encoder_cell_outputs: list
            Outputs of the encoder cell. Contains:

            - outputs of the transformer encoder cell. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer encoder cell
        """
        outputs, attention_weights =\
            self.attention_cell(inputs, inputs, inputs, mask)
        outputs = self.proj(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        additional_outputs = []
        if self._output_attention:
            additional_outputs.append(attention_weights)
        return outputs, additional_outputs


class TransformerEncoder(HybridBlock):
    """Structure of the Transformer Encoder.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    num_layers : int
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence. This is used for constructing position encoding
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, attention_cell='multi_head', num_layers=2,
                 units=128, hidden_size=2048, max_length=50,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TransformerEncoder, self).__init__(prefix=prefix, params=params)
        assert units % num_heads == 0, 'In TransformerEncoder, the units should be divided ' \
                                       'exactly by the number of heads. Received units={}, ' \
                                       'num_heads={}'.format(units, num_heads)
        self._num_layers = num_layers
        self._units = units
        self._hidden_size = hidden_size
        self._num_states = num_heads
        self._max_length = max_length
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        self._scaled = scaled
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()
            self.position_weight = self.params.get_constant('const',
                                                            _position_encoding_init(max_length,
                                                                                    units))
            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                self.transformer_cells.add(
                    TransformerEncoderCell(
                        units=units,
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        attention_cell=attention_cell,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer,
                        dropout=dropout,
                        scaled=scaled,
                        use_residual=use_residual,
                        output_attention=output_attention,
                        prefix='transformer%d_' % i))

    def encode_seq(self, inputs, valid_length=None):
        """Encode inputs. This function is only used for training.

        Parameters
        ----------
        inputs : NDArray, Shape (batch_size, length, C_in)
        valid_length : NDArray or None
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        output : NDArray, Shape (batch_size, length, C_out)
        states : list
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
        states = [None]
        output, states, additional_outputs = self.forward(inputs, states, mask)
        states = states[1:]
        if valid_length is not None:
            output = mx.nd.SequenceMask(output,
                                        sequence_length=valid_length,
                                        use_sequence_length=True,
                                        axis=1)
        return output, states, additional_outputs

    def __call__(self, step_input, states): #pylint: disable=arguments-differ
        """One-step-ahead decoding of the Transformer decoder.

        Parameters
        ----------
        step_input : NDArray
        states : list of NDArray

        Returns
        -------
        step_output : NDArray
            The output of the decoder.
            In the train mode, Shape is (batch_size, length, C_out)
            In the test mode, Shape is (batch_size, C_out)
        new_states: list
            Includes
            - last_embeds : NDArray or None
                It is only given during testing

        step_additional_outputs : list of list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)
        """
        return super(TransformerEncoder, self).__call__(step_input, states)

    def forward(self, step_input, states, mask=None):  #pylint: disable=arguments-differ, missing-docstring
        input_shape = step_input.shape
        # If it is in testing, transform input tensor to a tensor with shape NTC
        # Otherwise remove the None in states.
        if len(input_shape) == 2:
            has_last_embeds = len(states) == 1
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
            super(TransformerEncoder, self).forward(step_input * math.sqrt(step_input.shape[-1]),  #pylint: disable=too-many-function-args
                                                    states, mask)
        states = states[:-1]
        new_states = [step_input] + states
        # If it is in testing, only output the last one
        if len(input_shape) == 2:
            step_output = step_output[:, -1, :]
        return step_output, new_states, step_additional_outputs

    def hybrid_forward(self, F, step_input, states, mask=None, position_weight=None):  #pylint: disable=arguments-differ
        """

        Parameters
        ----------
        step_input : NDArray or Symbol, Shape (batch_size, length, C_in)
        states : list of NDArray or Symbol
        mask : NDArray or Symbol
        position_weight : NDArray or Symbol

        Returns
        -------
        step_output : NDArray or Symbol
            The output of the decoder. Shape is (batch_size, length, C_out)
        step_additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)

        """
        steps = states[0]
        # Positional Encoding
        step_input = F.broadcast_add(step_input,
                                     F.expand_dims(F.Embedding(steps,
                                                               position_weight,
                                                               self._max_length,
                                                               self._units),
                                                   axis=0))
        step_input = self.dropout_layer(step_input)
        step_input = self.layer_norm(step_input)
        inputs = step_input
        outputs = inputs
        step_additional_outputs = []
        attention_weights_l = []
        for cell in self.transformer_cells:
            outputs, attention_weights = cell(inputs, mask)
            if self._output_attention:
                attention_weights_l.append(attention_weights)
            inputs = outputs
        if self._output_attention:
            step_additional_outputs.extend(attention_weights_l)
        return outputs, step_additional_outputs

