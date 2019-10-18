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
# pylint: disable=too-many-lines
"""Encoder and decoder used in sequence-to-sequence learning."""

__all__ = ['TitanEncoder', 'PositionwiseFFN', 'TitanEncoderCell']

import math
import numpy as np
import mxnet as mx
from mxnet import cpu, gluon
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from mxnet.gluon.model_zoo import model_store
from gluonnlp.utils.parallel import Parallelizable
from .seq2seq_encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder
from .block import GELU
from .utils import _load_vocab, _load_pretrained_params


###############################################################################
#                               BASE ENCODER  BLOCKS                          #
###############################################################################

# pylint: disable=unused-argument
class _BandMatrix(mx.operator.CustomOp):
    def __init__(self, axis=-1):
        super(_BandMatrix, self).__init__(True)
        self._axis = axis

    def forward(self, is_train, req, in_data, out_data, aux):
        inputs = in_data[0]
        outputs = ((1 - self._epsilon) * inputs) + (self._epsilon / float(inputs.shape[self._axis]))
        self.assign(out_data[0], req[0], outputs)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], (1 - self._epsilon) * out_grad[0])


@mx.operator.register('_BandMatrix')
class _BandMatrixProp(mx.operator.CustomOpProp):
    def __init__(self, axis=-1):
        super(_BandMatrixProp, self).__init__(True)
        self._axis = int(axis)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = data_shape
        return (data_shape,), (output_shape,), ()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return out_grad

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return _BandMatrix(self._axis)
# pylint: enable=unused-argument

def _position_encoding_init(max_length, dim):
    """Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    # Apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return position_enc

def _get_layer_norm(use_bert, units, layer_norm_eps=None):
    from .bert import BERTLayerNorm
    layer_norm = BERTLayerNorm if use_bert else nn.LayerNorm
    if layer_norm_eps:
        return layer_norm(in_channels=units, epsilon=layer_norm_eps)
    else:
        return layer_norm(in_channels=units)


class ConvAttention(HybridBlock):
    def __init__(self, channels, kernel_size, feature_dim, use_bias=False,
                 weight_initializer=None, bias_initializer='zeros',
                 gamma_initializer='ones',
                 activation='relu', prefix=None, params=None):
        super(ConvAttention, self).__init__(prefix=prefix, params=params)
        self._channels = channels
        self._kernel_size = kernel_size
        self._feature_dim = feature_dim
        self._use_bias = use_bias
        units = {'kernel': kernel_size, 'feature': feature_dim}
        for name, unit in units.items():
            if name == 'feature' and unit % self._channels != 0:
                raise ValueError(
                    'In ConvAttentionn, the {name}_dim should be divided exactly'
                    ' by the number of channels. Received {name}_dim={unit}, channels={c}'.format(
                        name=name, unit=unit, c=channels))
            if name == 'kernel' and (unit - 1) % 2 != 0:
                raise ValueError(
                    'In ConvAttentionn, the {name}_size - 1 should be divided exactly'
                    ' by 2. Received {name}_size={unit}'.format(
                        name=name, unit=unit))
            with self.name_scope():
                setattr(
                    self, 'proj_{}'.format(name),
                    nn.Dense(units=unit, use_bias=self._use_bias, flatten=False,
                             weight_initializer=weight_initializer,
                             bias_initializer=bias_initializer, prefix='{}_'.format(name)))
        if activation is not None:
            self.act = nn.Activation(activation, prefix=activation + '_')
        else:
            self.act = None
        self.gamma = self.params.get('gamma', grad_req='write',
                                     shape=1, init=gamma_initializer,
                                     allow_deferred_init=True)

    def _project(self, F, name, x):
        # Shape (batch_size, length, units)
        x = getattr(self, 'proj_{}'.format(name))(x)
        # Shape (batch_size * channels, length, ele_units)
        x = F.transpose(x.reshape(shape=(0, 0, self._channels, -1)),
                        axes=(0, 2, 1, 3))\
             .reshape(shape=(-1, 0, 0), reverse=True)
        return x

    def hybrid_forward(self, F, kernel, feature, gamma, mask=None, bias=None):
        """
        Parameters
        ----------
        kernel : Symbol or NDArray
            Query vector. Shape (batch_size, kernel_length, kernel_dim)
        feature : Symbol or NDArray
            Key of the memory. Shape (batch_size, memory_length, feature_dim)
        mask : Symbol or NDArray or None, default None
            Mask of the memory slots. Shape (batch_size, kernel_length, memory_length)
            Only contains 0 or 1 where 0 means that the memory slot will not be used.
            If set to None. No mask will be used.

        Returns
        -------
        context_vec : Symbol or NDArray
            Shape (batch_size, query_length, context_vec_dim)
        """
        kernel = self._project(F, 'kernel', kernel)
        feature = self._project(F, 'feature', feature)
        kernel = F.expand_dims(kernel, axis=2)
        feature = F.expand_dims(feature, axis=1)
        act = F.broadcast_mul(kernel, feature)
        if bias is None:
            act = act + bias
        if self.act is not None:
            act = self.act(act)
        if mask is not None:
            mask = F.broadcast_axis(F.expand_dims(mask, axis=1),
                                    axis=1, size=self._channels) \
                .reshape(shape=(-1, 0, 0), reverse=True)
            act = act * F.expand_dims(mask, axis=-1)
        act = F.sum(act, axis=2)
        act = F.transpose(act.reshape(shape=(-1, self._channels, 0, 0),
                                      reverse=True),
                          axes=(0, 2, 1, 3)).reshape(shape=(0, 0, -1))
        time_wise_norm = F.sqrt(F.sum(F.square(act), axis=-1, keepdims=True))
        act = F.broadcast_div(act, time_wise_norm)
        act = act * gamma
        return act


class BasePositionwiseFFN(HybridBlock):
    """Base Structure of the Positionwise Feed-Forward Neural Network.

    Parameters
    ----------
    units : int
        Number of units for the output
    hidden_size : int
        Number of units in the hidden layer of position-wise feed-forward networks
    dropout : float
    use_residual : bool
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    activation : str, default 'relu'
        Activation function
    use_bert_layer_norm : bool, default False.
        Whether to use the BERT-stype layer norm implemented in Tensorflow, where
        epsilon is added inside the square root. Set to True for pre-trained BERT model.
    ffn1_dropout : bool, default False
        If True, apply dropout both after the first and second Positionwise
        Feed-Forward Neural Network layers. If False, only apply dropout after
        the second.
    prefix : str, default None
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    layer_norm_eps : float, default None
        Epsilon for layer_norm

    Inputs:
        - **inputs** : input sequence of shape (batch_size, length, C_in).

    Outputs:
        - **outputs** : output encoding of shape (batch_size, length, C_out).
    """

    def __init__(self, units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                 weight_initializer=None, bias_initializer='zeros', activation='relu',
                 use_bert_layer_norm=False, ffn1_dropout=False, prefix=None, params=None,
                 layer_norm_eps=None):
        super(BasePositionwiseFFN, self).__init__(prefix=prefix, params=params)
        self._hidden_size = hidden_size
        self._units = units
        self._use_residual = use_residual
        self._dropout = dropout
        self._ffn1_dropout = ffn1_dropout
        with self.name_scope():
            self.ffn_1 = nn.Dense(units=hidden_size, flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_1_')
            self.activation = self._get_activation(activation) if activation else None
            self.ffn_2 = nn.Dense(units=units, flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_2_')
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.layer_norm = _get_layer_norm(use_bert_layer_norm, units,
                                              layer_norm_eps=layer_norm_eps)

    def _get_activation(self, act):
        """Get activation block based on the name. """
        if isinstance(act, str):
            if act.lower() == 'gelu':
                return GELU()
            else:
                return gluon.nn.Activation(act)
        assert isinstance(act, gluon.Block)
        return act

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
        if self.activation:
            outputs = self.activation(outputs)
        if self._dropout and self._ffn1_dropout:
            outputs = self.dropout_layer(outputs)
        outputs = self.ffn_2(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        return outputs


class BaseTitanEncoderCell(HybridBlock):
    """Base Structure of the Titan Encoder Cell.
    """
    def __init__(self, units=128,
                 hidden_size=512, channels=4,
                 dropout=0.0, use_residual=True,
                 weight_initializer=None, bias_initializer='zeros',
                 proj_use_bias=False,
                 use_bert_layer_norm=False, use_bert_ffn=False,
                 prefix=None, params=None,
                 activation='relu', layer_norm_eps=None):
        super(BaseTitanEncoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._channels = channels
        self._dropout = dropout
        self._use_residual = use_residual
        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.attention_cell = ConvAttention(channels, units,
                                                units, activation=activation)
            self.proj = nn.Dense(units=units, flatten=False,
                                 use_bias=proj_use_bias,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix='proj_')
            self.ffn = self._get_positionwise_ffn(use_bert_ffn, units, hidden_size, dropout,
                                                  use_residual, weight_initializer,
                                                  bias_initializer, activation=activation,
                                                  layer_norm_eps=layer_norm_eps)
            self.layer_norm = _get_layer_norm(use_bert_layer_norm, units,
                                              layer_norm_eps=layer_norm_eps)

    def _get_positionwise_ffn(self, use_bert, units, hidden_size, dropout, use_residual,
                              weight_initializer, bias_initializer, activation='relu',
                              layer_norm_eps=None):
        from .bert import BERTPositionwiseFFN
        positionwise_ffn = BERTPositionwiseFFN if use_bert else PositionwiseFFN
        return positionwise_ffn(units=units, hidden_size=hidden_size, dropout=dropout,
                                use_residual=use_residual, weight_initializer=weight_initializer,
                                bias_initializer=bias_initializer, activation=activation,
                                layer_norm_eps=layer_norm_eps)

    def hybrid_forward(self, F, inputs, mask=None):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Titan Encoder Attention Cell.

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

            - outputs of the titan encoder cell. Shape (batch_size, length, C_out)
        """
        outputs, attention_weights =\
            self.attention_cell(inputs, inputs, mask)
        outputs = self.proj(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        return outputs


class BaseTitanEncoder(HybridBlock, Seq2SeqEncoder):
    """Base Structure of the Titan Encoder.
    """
    def __init__(self, num_layers=2,
                 units=512, hidden_size=2048, max_length=50,
                 channels=4, dropout=0.0,
                 use_residual=True, output_all_encodings=False,
                 weight_initializer=None, bias_initializer='zeros',
                 positional_weight='sinusoidal', use_bert_encoder=False,
                 use_layer_norm_before_dropout=False, scale_embed=True,
                 prefix=None, params=None, activation='relu', layer_norm_eps=None):
        super(BaseTitanEncoder, self).__init__(prefix=prefix, params=params)
        assert units % channels == 0,\
            'In TitanEncoder, The units should be divided exactly ' \
            'by the number of channels. Received units={}, channels={}' \
            .format(units, channels)
        self._num_layers = num_layers
        self._max_length = max_length
        self._channels = channels
        self._units = units
        self._hidden_size = hidden_size
        self._output_all_encodings = output_all_encodings
        self._dropout = dropout
        self._use_residual = use_residual
        self._use_layer_norm_before_dropout = use_layer_norm_before_dropout
        self._scale_embed = scale_embed
        self._dtype = 'float32'

        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.layer_norm = _get_layer_norm(use_bert_encoder, units,
                                              layer_norm_eps=layer_norm_eps)
            self.position_weight = self._get_positional(positional_weight, max_length, units,
                                                        weight_initializer)
            self.titan_cells = nn.HybridSequential()
            for i in range(num_layers):
                cell = self._get_encoder_cell(use_bert_encoder, units, hidden_size, channels,
                                              weight_initializer, bias_initializer,
                                              dropout, use_residual, i,
                                              activation=activation, layer_norm_eps=layer_norm_eps)
                self.titan_cells.add(cell)

    def _get_positional(self, weight_type, max_length, units, initializer):
        if weight_type == 'sinusoidal':
            encoding = _position_encoding_init(max_length, units)
            position_weight = self.params.get_constant('const', encoding)
        elif weight_type == 'learned':
            position_weight = self.params.get('position_weight', shape=(max_length, units),
                                              init=initializer)
        else:
            raise ValueError('Unexpected value for argument position_weight: %s'%(position_weight))
        return position_weight

    def _get_encoder_cell(self, use_bert, units, hidden_size, channels,
                          weight_initializer, bias_initializer, dropout, use_residual,
                          i, activation='relu', layer_norm_eps=None):
        from .bert import BERTEncoderCell
        cell = BERTEncoderCell if use_bert else BaseTitanEncoderCell
        return cell(units=units, hidden_size=hidden_size,
                    channels=channels,
                    weight_initializer=weight_initializer,
                    bias_initializer=bias_initializer,
                    dropout=dropout, use_residual=use_residual,
                    prefix='titan%d_'%i,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps)

    def cast(self, dtype):
        """Cast the data type of the parameters"""
        self._dtype = dtype
        super(BaseTitanEncoder, self).cast(dtype)

    def __call__(self, inputs, states=None, valid_length=None):
        #pylint: disable=arguments-differ, dangerous-default-value
        """Encode the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray or Symbol
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or Symbols
            Initial states. The list of initial states and masks
        valid_length : NDArray or Symbol
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        encoder_outputs: list
            Outputs of the encoder. Contains:

            - outputs of the titan encoder. Shape (batch_size, length, C_out)
            - additional_outputs of all the titan encoder
        """
        # XXX Temporary hack for hybridization as hybridblock does not support None inputs
        valid_length = [] if valid_length is None else valid_length
        states = [] if states is None else states
        return super(BaseTitanEncoder, self).__call__(inputs, states, valid_length)

    def _arange_like(self, F, inputs, axis):
        """Helper function to generate indices of a range"""
        if F == mx.ndarray:
            seq_len = inputs.shape[axis]
            arange = F.arange(seq_len, dtype=inputs.dtype, ctx=inputs.context)
        else:
            input_axis = inputs.slice(begin=(0, 0, 0), end=(1, None, 1)).reshape((-1))
            zeros = F.zeros_like(input_axis)
            arange = F.arange(start=0, repeat=1, step=1,
                              infer_range=True, dtype=self._dtype)
            arange = F.elemwise_add(arange, zeros)
        return arange


    def hybrid_forward(self, F, inputs, states=None, valid_length=None, position_weight=None):
        # pylint: disable=arguments-differ
        """Encode the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray or Symbol
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or Symbols
            Initial states. The list of initial states and masks
        valid_length : NDArray or Symbol
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)
        position_weight : NDArray or Symbol
            The weight of positional encoding. Shape (max_len, C_in).

        Returns
        -------
        encoder_outputs: list
            Outputs of the encoder. Contains:

            - outputs of the titan encoder. Shape (batch_size, length, C_out)
            - additional_outputs of all the titan encoder

        Returns
        -------
        outputs : NDArray or Symbol, or List[NDArray] or List[Symbol]
            If output_all_encodings flag is False, then the output of the last encoder.
            If output_all_encodings flag is True, then the list of all outputs of all encoders.
            In both cases, shape of the tensor(s) is/are (batch_size, length, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, length) or
            (batch_size, num_heads, length, length)

        """
        # XXX Temporary hack for hybridization as hybridblock does not support None inputs
        if isinstance(valid_length, list) and len(valid_length) == 0:
            valid_length = None
        if isinstance(states, list) and len(states) == 0:
            states = None

        steps = self._arange_like(F, inputs, axis=1)
        if valid_length is not None:
            mask = F.broadcast_lesser(F.reshape(steps, shape=(1, -1)),
                                      F.reshape(valid_length, shape=(-1, 1)))
            mask = F.broadcast_mul(F.expand_dims(mask, axis=1),
                                   F.expand_dims(mask, axis=-1))
            if states is None:
                states = [mask]
            else:
                states.append(mask)

        if self._scale_embed:
            # XXX: input.shape[-1] and self._units are expected to be the same
            inputs = inputs * math.sqrt(self._units)

        if states is None:
            states = [steps]
        else:
            states.append(steps)

        if states is not None:
            steps = states[-1]
            # positional encoding
            positional_embed = F.Embedding(steps, position_weight, self._max_length, self._units)
            inputs = F.broadcast_add(inputs, F.expand_dims(positional_embed, axis=0))
        if self._dropout:
            if self._use_layer_norm_before_dropout:
                inputs = self.layer_norm(inputs)
                inputs = self.dropout_layer(inputs)
            else:
                inputs = self.dropout_layer(inputs)
                inputs = self.layer_norm(inputs)
        else:
            inputs = self.layer_norm(inputs)
        outputs = inputs
        if valid_length is not None:
            mask = states[-2]
        else:
            mask = None

        all_encodings_outputs = []
        additional_outputs = []
        for cell in self.titan_cells:
            outputs = cell(inputs, mask)
            inputs = outputs
            if self._output_all_encodings:
                if valid_length is not None:
                    outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                             use_sequence_length=True, axis=1)
                all_encodings_outputs.append(outputs)

        if valid_length is not None:
            outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                     use_sequence_length=True, axis=1)

        if self._output_all_encodings:
            return all_encodings_outputs, additional_outputs
        else:
            return outputs, additional_outputs

###############################################################################
#                                ENCODER                                      #
###############################################################################

class PositionwiseFFN(BasePositionwiseFFN):
    """Structure of the Positionwise Feed-Forward Neural Network for
    Titan.

    Computes the positionwise encoding of the inputs.

    Parameters
    ----------
    units : int
        Number of units for the output
    hidden_size : int
        Number of units in the hidden layer of position-wise feed-forward networks
    dropout : float
        Dropout probability for the output
    use_residual : bool
        Add residual connection between the input and the output
    ffn1_dropout : bool, default False
        If True, apply dropout both after the first and second Positionwise
        Feed-Forward Neural Network layers. If False, only apply dropout after
        the second.
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None
        Prefix for name of `Block`s (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.
    activation : str, default 'relu'
        Activation methods in PositionwiseFFN
    layer_norm_eps : float, default None
        Epsilon for layer_norm

    Inputs:
        - **inputs** : input sequence of shape (batch_size, length, C_in).

    Outputs:
        - **outputs** : output encoding of shape (batch_size, length, C_out).
    """

    def __init__(self, units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                 ffn1_dropout=False, weight_initializer=None, bias_initializer='zeros', prefix=None,
                 params=None, activation='relu', layer_norm_eps=None):
        super(PositionwiseFFN, self).__init__(
            units=units,
            hidden_size=hidden_size,
            dropout=dropout,
            use_residual=use_residual,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            prefix=prefix,
            params=params,
            # extra configurations for titan
            activation=activation,
            use_bert_layer_norm=False,
            layer_norm_eps=layer_norm_eps,
            ffn1_dropout=ffn1_dropout)


class TitanEncoderCell(BaseTitanEncoderCell):
    """Structure of the Titan Encoder Cell.
    Inputs:
        - **inputs** : input sequence. Shape (batch_size, length, C_in)
        - **mask** : mask for inputs. Shape (batch_size, length, length)

    Outputs:
        - **outputs**: output tensor of the titan encoder cell.
            Shape (batch_size, length, C_out)
    """
    def __init__(self, units=128,
                 hidden_size=512, channels=4,
                 dropout=0.0, use_residual=True,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None, activation='relu', layer_norm_eps=None):
        super(TitanEncoderCell, self).__init__(units=units, hidden_size=hidden_size,
                                               channels=channels,
                                               dropout=dropout, use_residual=use_residual,
                                               weight_initializer=weight_initializer,
                                               bias_initializer=bias_initializer,
                                               prefix=prefix, params=params,
                                               # extra configurations for titan
                                               proj_use_bias=False,
                                               use_bert_layer_norm=False,
                                               use_bert_ffn=False,
                                               activation=activation,
                                               layer_norm_eps=layer_norm_eps)

class TitanEncoder(BaseTitanEncoder):
    """Structure of the Titan Encoder.
    Inputs:
        - **inputs** : input sequence of shape (batch_size, length, C_in)
        - **states** : list of tensors for initial states and masks.
        - **valid_length** : valid lengths of each sequence. Usually used when part of sequence
            has been padded. Shape is (batch_size, )

    Outputs:
        - **outputs** : the output of the encoder. Shape is (batch_size, length, C_out)
        - **additional_outputs** : list of tensors.
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)
    """
    def __init__(self, num_layers=2,
                 units=512, hidden_size=2048, max_length=50,
                 channels=4, dropout=0.0,
                 use_residual=True,
                 activation='relu',
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TitanEncoder, self).__init__(
            num_layers=num_layers, units=units,
            hidden_size=hidden_size, max_length=max_length,
            channels=channels,
            dropout=dropout, use_residual=use_residual,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            prefix=prefix, params=params,
            # extra configurations for titan
            positional_weight='sinusoidal',
            use_bert_encoder=False,
            use_layer_norm_before_dropout=False,
            scale_embed=True,
            activation=activation)

###############################################################################
#                                DECODER                                      #
###############################################################################

class TitanDecoderCell(HybridBlock):
    """Structure of the Titan Decoder Cell.
    """
    def __init__(self, units=128,
                 hidden_size=512, channels=4,
                 dropout=0.0, use_residual=True,
                 weight_initializer=None, bias_initializer='zeros',
                 activation=None, prefix=None, params=None):
        super(TitanDecoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._channels = channels
        self._dropout = dropout
        self._use_residual = use_residual
        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.attention_cell_in = ConvAttention(channels, units,
                                                   units, activation=activation)
            self.attention_cell_inter = ConvAttention(channels, units,
                                                      units, activation=activation)
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

    def hybrid_forward(self, F, inputs, mem_value, mask=None, mem_mask=None):  #pylint: disable=unused-argument
        #  pylint: disable=arguments-differ
        """Titan Decoder Attention Cell.

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

            - outputs of the titan decoder cell. Shape (batch_size, length, C_out)
            - additional_outputs of all the titan decoder cell
        """
        outputs, attention_in_outputs =\
            self.attention_cell_in(inputs, inputs, mask)
        outputs = self.proj_in(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm_in(outputs)
        inputs = outputs
        outputs, attention_inter_outputs = \
            self.attention_cell_inter(inputs, mem_value, mem_mask)
        outputs = self.proj_inter(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm_inter(outputs)
        outputs = self.ffn(outputs)
        return outputs


class TitanDecoder(HybridBlock, Seq2SeqDecoder):
    """Structure of the Titan Decoder.
    """
    def __init__(self, num_layers=2,
                 units=128, hidden_size=2048, max_length=50,
                 channels=4, dropout=0.0,
                 use_residual=True,
                 activation='relu',
                 weight_initializer=None, bias_initializer='zeros',
                 scale_embed=True, prefix=None, params=None):
        super(TitanDecoder, self).__init__(prefix=prefix, params=params)
        assert units % channels == 0, 'In TitanDecoder, the units should be divided ' \
                                      'exactly by the number of channels. Received units={}, ' \
                                      'channels={}'.format(units, channels)
        self._num_layers = num_layers
        self._units = units
        self._hidden_size = hidden_size
        self._channels = channels
        self._max_length = max_length
        self._dropout = dropout
        self._use_residual = use_residual
        self._scale_embed = scale_embed
        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.layer_norm = nn.LayerNorm()
            encoding = _position_encoding_init(max_length, units)
            self.position_weight = self.params.get_constant('const', encoding)
            self.titan_cells = nn.HybridSequential()
            for i in range(num_layers):
                self.titan_cells.add(
                    TitanDecoderCell(
                        units=units,
                        hidden_size=hidden_size,
                        channels=channels,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer,
                        dropout=dropout,
                        use_residual=use_residual,
                        activation=activation,
                        prefix='titan%d_' % i))

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
        mem_length = mem_value.shape[1]
        if encoder_valid_length is not None:
            dtype = encoder_valid_length.dtype
            ctx = encoder_valid_length.context
            mem_masks = mx.nd.broadcast_lesser(
                mx.nd.arange(mem_length, ctx=ctx, dtype=dtype).reshape((1, -1)),
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
        length_array = mx.nd.arange(length, ctx=inputs.context, dtype=inputs.dtype)
        mask = mx.nd.broadcast_lesser_equal(
            length_array.reshape((1, -1)),
            length_array.reshape((-1, 1)))
        if valid_length is not None:
            arange = mx.nd.arange(length, ctx=valid_length.context, dtype=valid_length.dtype)
            batch_mask = mx.nd.broadcast_lesser(
                arange.reshape((1, -1)),
                valid_length.reshape((-1, 1)))
            mask = mx.nd.broadcast_mul(mx.nd.expand_dims(batch_mask, -1),
                                       mx.nd.expand_dims(mask, 0))
        else:
            mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=0), axis=0, size=batch_size)
        states = [None] + states
        output, states, additional_outputs = self.forward(inputs, states, mask)
        states = states[1:]
        if valid_length is not None:
            output = mx.nd.SequenceMask(output,
                                        sequence_length=valid_length,
                                        use_sequence_length=True,
                                        axis=1)
        return output, states, additional_outputs

    def __call__(self, step_input, states): #pylint: disable=arguments-differ
        """One-step-ahead decoding of the Titan decoder.

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
            - mem_value : NDArray
            - mem_masks : NDArray, optional

        step_additional_outputs : list of list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)
        """
        return super(TitanDecoder, self).__call__(step_input, states)

    def forward(self, step_input, states, mask=None):  #pylint: disable=arguments-differ, missing-docstring
        input_shape = step_input.shape
        mem_mask = None
        # If it is in testing, transform input tensor to a tensor with shape NTC
        # Otherwise remove the None in states.
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
            length_array = mx.nd.arange(step_input.shape[1], ctx=step_input.context,
                                        dtype=step_input.dtype)
            mask = mx.nd.broadcast_lesser_equal(
                length_array.reshape((1, -1)),
                length_array.reshape((-1, 1)))
            mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=0),
                                        axis=0, size=step_input.shape[0])
        steps = mx.nd.arange(step_input.shape[1], ctx=step_input.context)
        states.append(steps)
        if self._scale_embed:
            scaled_step_input = step_input * math.sqrt(step_input.shape[-1])
        # pylint: disable=too-many-function-args
        step_output, step_additional_outputs = \
            super(TitanDecoder, self).forward(scaled_step_input, states, mask)
        states = states[:-1]
        if has_mem_mask:
            states[-1] = mem_mask
        new_states = [step_input] + states
        # If it is in testing, only output the last one
        if len(input_shape) == 2:
            step_output = step_output[:, -1, :]
        return step_output, new_states, step_additional_outputs

    def hybrid_forward(self, F, step_input, states, mask=None, position_weight=None):
        #pylint: disable=arguments-differ
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
        has_mem_mask = (len(states) == 3)
        if has_mem_mask:
            mem_value, mem_mask, steps = states
        else:
            mem_value, steps = states
            mem_mask = None
        # Positional Encoding
        step_input = F.broadcast_add(step_input,
                                     F.expand_dims(F.Embedding(steps,
                                                               position_weight,
                                                               self._max_length,
                                                               self._units),
                                                   axis=0))
        if self._dropout:
            step_input = self.dropout_layer(step_input)
        step_input = self.layer_norm(step_input)
        inputs = step_input
        outputs = inputs
        step_additional_outputs = []
        for cell in self.titan_cells:
            outputs = cell(inputs, mem_value, mask, mem_mask)
            inputs = outputs
        return outputs, step_additional_outputs



###############################################################################
#                                  MODEL API                                  #
###############################################################################

def get_titan_encoder_decoder(num_layers=2,
                              channels=8,
                              units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                              max_src_length=50, max_tgt_length=50,
                              weight_initializer=None, bias_initializer='zeros',
                              prefix='titan_', params=None):
    """Build a pair of Parallel Titan encoder/decoder
    Returns
    -------
    encoder : TitanEncoder
    decoder :TitanDecoder
    """
    encoder = TitanEncoder(num_layers=num_layers,
                           channels=channels,
                           max_length=max_src_length,
                           units=units,
                           hidden_size=hidden_size,
                           dropout=dropout,
                           use_residual=use_residual,
                           weight_initializer=weight_initializer,
                           bias_initializer=bias_initializer,
                           prefix=prefix + 'enc_', params=params)
    decoder = TitanDecoder(num_layers=num_layers,
                           channels=channels,
                           max_length=max_tgt_length,
                           units=units,
                           hidden_size=hidden_size,
                           dropout=dropout,
                           use_residual=use_residual,
                           weight_initializer=weight_initializer,
                           bias_initializer=bias_initializer,
                           prefix=prefix + 'dec_', params=params)
    return encoder, decoder


def _get_titan_model(model_cls, model_name, dataset_name, src_vocab, tgt_vocab,
                     encoder, decoder, share_embed, embed_size, tie_weights,
                     embed_initializer, pretrained, ctx, root, **kwargs):
    src_vocab = _load_vocab(dataset_name + '_src', src_vocab, root)
    tgt_vocab = _load_vocab(dataset_name + '_tgt', tgt_vocab, root)
    kwargs['encoder'] = encoder
    kwargs['decoder'] = decoder
    kwargs['src_vocab'] = src_vocab
    kwargs['tgt_vocab'] = tgt_vocab
    kwargs['share_embed'] = share_embed
    kwargs['embed_size'] = embed_size
    kwargs['tie_weights'] = tie_weights
    kwargs['embed_initializer'] = embed_initializer
    # XXX the existing model is trained with prefix 'titan_'
    net = model_cls(prefix='titan_', **kwargs)
    if pretrained:
        _load_pretrained_params(net, model_name, dataset_name, root, ctx)
    return net, src_vocab, tgt_vocab


class ParallelTitan(Parallelizable):
    """Data parallel titan.

    Parameters
    ----------
    model : Block
        The titan model.
    label_smoothing: Block
        The block to perform label smoothing.
    loss_function : Block
        The loss function to optimizer.
    rescale_loss : float
        The scale to which the loss is rescaled to avoid gradient explosion.
    """
    def __init__(self, model, label_smoothing, loss_function, rescale_loss):
        self._model = model
        self._label_smoothing = label_smoothing
        self._loss = loss_function
        self._rescale_loss = rescale_loss

    def forward_backward(self, x):
        """Perform forward and backward computation for a batch of src seq and dst seq"""
        (src_seq, tgt_seq, src_valid_length, tgt_valid_length), batch_size = x
        with mx.autograd.record():
            out, _ = self._model(src_seq, tgt_seq[:, :-1],
                                 src_valid_length, tgt_valid_length - 1)
            smoothed_label = self._label_smoothing(tgt_seq[:, 1:])
            ls = self._loss(out, smoothed_label, tgt_valid_length - 1).sum()
            ls = (ls * (tgt_seq.shape[1] - 1)) / batch_size / self._rescale_loss
        ls.backward()
        return ls
