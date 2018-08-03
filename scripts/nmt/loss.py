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
"""Loss functions."""

import numpy as np
import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.loss import Loss, SoftmaxCELoss, _reshape_like, _apply_weighting


class SoftmaxCEMaskedLoss(SoftmaxCELoss):
    """Wrapper of the SoftmaxCELoss that supports valid_length as the input

    """
    def hybrid_forward(self, F, pred, label, valid_length): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        F
        pred : Symbol or NDArray
            Shape (batch_size, length, V)
        label : Symbol or NDArray
            Shape (batch_size, length)
        valid_length : Symbol or NDArray
            Shape (batch_size, )
        Returns
        -------
        loss : Symbol or NDArray
            Shape (batch_size,)
        """
        if self._sparse_label:
            sample_weight = F.cast(F.expand_dims(F.ones_like(label), axis=-1), dtype=np.float32)
        else:
            sample_weight = F.ones_like(label)
        sample_weight = F.SequenceMask(sample_weight,
                                       sequence_length=valid_length,
                                       use_sequence_length=True,
                                       axis=1)
        return super(SoftmaxCEMaskedLoss, self).hybrid_forward(F, pred, label, sample_weight)


class LogSumExpOp(mx.operator.CustomOp):
    """Implementation of log sum exp for numerical stability
    """
    def __init__(self, axis):
        self.axis = axis

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        max_x = mx.nd.max_axis(x, axis=self.axis, keepdims=True)
        sum_x = mx.nd.sum(mx.nd.exp(x - max_x), axis=self.axis, keepdims=True)
        y = mx.nd.log(sum_x) + max_x
        y = y.reshape(out_data[0].shape)
        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = out_grad[0]
        x = in_data[0]
        max_x = mx.nd.max_axis(x, axis=self.axis, keepdims=True)
        y = y.reshape(max_x.shape)
        x = mx.nd.exp(x - max_x)
        prob = x / mx.nd.sum(x, axis=self.axis, keepdims=True)
        self.assign(in_grad[0], req[0], prob * y)


@mx.operator.register("log_sum_exp")
class LogSumExpProp(mx.operator.CustomOpProp):
    def __init__(self, axis, keepdims=False):
        super(LogSumExpProp, self).__init__(need_top_grad=True)
        self.axis = int(axis)
        self.keepdims = keepdims in ('True',)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        oshape = []
        for i, x in enumerate(data_shape):
            if i == self.axis:
                if self.keepdims:
                    oshape.append(1)
            else:
                oshape.append(x)
        return [data_shape], [tuple(oshape)], []

    def create_operator(self, ctx, shapes, dtypes):
        return LogSumExpOp(self.axis)


def log_sum_exp(in_sym, axis, keepdims=False, name="log_sum_exp"):
    return mx.symbol.Custom(in_sym, name=name,
                            op_type="log_sum_exp",
                            axis=axis, keepdims=keepdims)


class MixSoftmaxCEMaskedLoss(Loss):
    """Wrapper of the Mixture SoftmaxCELoss that supports valid_length as the input
    """

    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, num_mix=1, **kwargs):
        super(MixSoftmaxCEMaskedLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        self._num_mix = num_mix

    def hybrid_forward(self, F, pred, mix, label, valid_length):
        """
        Parameters
        ----------
        F
        pred : Symbol or NDArray
            Shape (batch_size * num_states, length, V)
        mix : Symbol or NDArray
            Shape (batch_size, length, num_states)
        label : Symbol or NDArray
            Shape (batch_size, length)
        valid_length : Symbol or NDArray
            Shape (batch_size, )
        Returns
        -------
        loss : Symbol or NDArray
            Shape (batch_size,)
        """
        if not self._from_logits:
            mix = F.transpose(F.log_softmax(mix, self._axis), axes=(0, 2, 1)).reshape(shape=(-3, 0))
            pred = F.log_softmax(pred, self._axis)
            pred = F.broadcast_add(pred, F.expand_dims(mix, -1))
            pred = pred.reshape(shape=(-4, -1, self._num_mix, 0, 0))
            #pred = F.Custom(pred, op_type='log_sum_exp', axis=1)
            max_pred = F.max_axis(pred, 1, keepdims=True)
            sum_pred = F.sum(F.exp(F.broadcast_minus(pred, max_pred)), axis=1)
            pred = F.log(sum_pred) + F.squeeze(max_pred, axis=1)
        if self._sparse_label:
            sample_weight = F.cast(F.expand_dims(F.ones_like(label), axis=-1), dtype=np.float32)
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            sample_weight = F.ones_like(label)
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred * label, axis=self._axis, keepdims=True)
        sample_weight = F.SequenceMask(sample_weight,
                                       sequence_length=valid_length,
                                       use_sequence_length=True,
                                       axis=1)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


# pylint: disable=unused-argument
class _SmoothingWithDim(mx.operator.CustomOp):
    def __init__(self, epsilon=0.1, axis=-1):
        super(_SmoothingWithDim, self).__init__(True)
        self._epsilon = epsilon
        self._axis = axis

    def forward(self, is_train, req, in_data, out_data, aux):
        inputs = in_data[0]
        outputs = ((1 - self._epsilon) * inputs) + (self._epsilon / float(inputs.shape[self._axis]))
        self.assign(out_data[0], req[0], outputs)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], (1 - self._epsilon) * out_grad[0])


@mx.operator.register('_smoothing_with_dim')
class _SmoothingWithDimProp(mx.operator.CustomOpProp):
    def __init__(self, epsilon=0.1, axis=-1):
        super(_SmoothingWithDimProp, self).__init__(True)
        self._epsilon = float(epsilon)
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
        return _SmoothingWithDim(self._epsilon, self._axis)
# pylint: enable=unused-argument


class LabelSmoothing(HybridBlock):
    """Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Parameters
    ----------
    axis : int, default -1
        The axis to smooth.
    epsilon : float, default 0.1
        The epsilon parameter in label smoothing
    sparse_label : bool, default True
        Whether input is an integer array instead of one hot array.
    units : int or None
        Vocabulary size. If units is not given, it will be inferred from the input.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, axis=-1, epsilon=0.1, units=None,
                 sparse_label=True, prefix=None, params=None):
        super(LabelSmoothing, self).__init__(prefix=prefix, params=params)
        self._axis = axis
        self._epsilon = epsilon
        self._sparse_label = sparse_label
        self._units = units

    def hybrid_forward(self, F, inputs, units=None): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        F
        inputs : Symbol or NDArray
            Shape (batch_size, length) or (batch_size, length, V)
        units : int or None
        Returns
        -------
        smoothed_label : Symbol or NDArray
            Shape (batch_size, length, V)
        """
        if self._sparse_label:
            assert units is not None or self._units is not None, \
                'units needs to be given in function call or ' \
                'instance initialization when sparse_label is False'
            if units is None:
                units = self._units
            inputs = F.one_hot(inputs, depth=units)
        if units is None and self._units is None:
            return F.Custom(inputs, epsilon=self._epsilon, axis=self._axis,
                            op_type='_smoothing_with_dim')
        else:
            if units is None:
                units = self._units
            return ((1 - self._epsilon) * inputs) + (self._epsilon / units)
