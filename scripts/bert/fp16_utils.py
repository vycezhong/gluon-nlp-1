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

"""Trainer for mixed precision training."""
import os
import warnings
import collections
import mxnet as mx
from mxnet import nd

from mxnet.optimizer import Optimizer, register
from mxnet.engine import bulk
from mxnet.ndarray import zeros, ones_like, NDArray
from mxnet.ndarray import square, power, sqrt, maximum, minimum, clip, where

import math


from mxnet.ndarray import square, power, sqrt, maximum, minimum, clip, where, norm, full

def _projection(weight, var, alpha=0.1, iters=10, eps=1e-6):
    var /= NDArray.mean(var)
    scale = min(1, alpha / norm(weight))
    if scale == 1:
        return weight
    alpha = square(full(shape=(1,), val=alpha, ctx=weight.context))
    weight[:] *= sqrt(var)
    var = 1 / var
    z = weight * scale
    beta = 1 / NDArray.max(var)
    lam = beta
    c = weight.copy()
    for _ in range(iters):
        Az = var * z
        c[:] = z
        c -= lam * Az
        gamma = lam * norm(Az)
        gamma += sqrt(beta) * (sqrt(alpha) - sqrt(NDArray.sum(z * Az)))
        shift_weight = weight - c
        y = shift_weight * min(1, gamma / norm(shift_weight))
        y += c
        if sqrt(NDArray.sum(y * var * y)) <= alpha:
            v = y
        else:
            d = z - y
            denom = NDArray.sum(d * var * d) + eps
            dAz = NDArray.sum(d * Az)
            nom = dAz + sqrt(maximum(square(dAz) - denom * (NDArray.sum(z * Az) - alpha), 0))
            tau = nom / denom
            v = z + tau * (y - z)
        d = v - weight
        denom = NDArray.sum(d * var * d) + eps
        Av = var * v
        dAv = NDArray.sum(d * Av)
        nom = dAv + sqrt(maximum(square(dAv) - denom * (NDArray.sum(v * Av) - alpha), 0))
        tau = nom / denom
        z[:] = v + tau * (weight - v)
    z[:] *= sqrt(var)
    return z

@register
class LAMB2(Optimizer):
    """The LAMB optimizer proposed in
    `Reducing BERT Pre-Training Time from 3 Days to 76 Minutes <https://arxiv.org/abs/1904.00962>`_.

    If bias_correction is set to False, updates are applied by::

        grad = clip(grad * rescale_grad, clip_gradient)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        r1 = min(max(w.norm(), lower_bound), upper_bound)
        g = m / (sqrt(v_hat) + epsilon) + wd * w
        r2 = g.norm()
        r = 1. if r1 == 0. or r2 == 0. else r1 / r2
        lr = r * lr
        w = w - lr * g

    Otherwise, updates are applied by::

        grad = clip(grad * rescale_grad, clip_gradient)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - power(beta1, t))
        v_hat = m / (1 - power(beta2, t))
        r1 = w.norm()
        g = m_hat / (sqrt(v_hat + epsilon)) + wd * w
        r2 = g.norm()
        r = 1. if r1 == 0. or r2 == 0. else r1 / r2
        lr = r * lr
        w = w - lr * g

    Parameters
    ----------
    beta1 : float, optional, default is 0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional, default is 0.999
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional, default is 1e-6
        Small value to avoid division by 0.
    lower_bound : float, optional, default is 1e-3
        Lower limit of norm of weight
    upper_bound : float, optional, default is 10.0
        Upper limit of norm of weight
    bias_correction : bool, optional, default is False
        Whether to use bias correction, in the latest version of the lamb,
        the bias correction was removed and some simple changes were made.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6,
                 lower_bound=1e-3, upper_bound=10.0, bias_correction=False, verbose=False,
                 num_heads=None,  **kwargs):
        super(LAMB2, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bias_correction = bias_correction
        self.num_heads = num_heads
        self.att_list = ['query', 'key', 'value']
        import os
        import logging
        if os.environ.get('EPS_AFTER_SQRT', False):
            self._eps_after_sqrt = True
            logging.info('self._eps_after_sqrt = ' + str(self._eps_after_sqrt))
        else:
            self._eps_after_sqrt = False
        self._bulk = int(os.environ.get('LAMB_BULK', 0))
        logging.info(" bulk = " + str(self._bulk))
        self._verbose = verbose
        if int(os.environ.get('USE_BOUND', False)):
            logging.info("using upper lower bound")
            self._use_bound = True
        else:
            self._use_bound = False
        if int(os.environ.get('USE_PROJ', False)):
            logging.info("use projection")
            self._use_proj = True
        else:
            self._use_proj = False
        if int(os.environ.get('FORCE_WD', False)):
            logging.info("force wd")
            self._force_wd = True
        else:
            self._force_wd = False
        if int(os.environ.get('ADJUST_BOUND', False)):
            logging.info("adjusting bound ")
            self._adjust_bound = True
        else:
            self._adjust_bound = False
        if int(os.environ.get('SPLIT_HEAD', False)):
            logging.info("splitting head ")
            self._split_head = True
            assert self.num_heads is not None, 'When SPLIT_HEAD=True, num_heads needs to be given.'
        else:
            self._split_head = False
        logging.info('attrs = {}'.format(str(self.__dict__)))


    def create_state(self, index, weight):
        stype = weight.stype
        name = self.idx2name[index]
        att = False
        for att_name in self.att_list:
             if att_name in name:
                 att = True
                 break
        if att and self._split_head:
            if len(weight.shape) == 2:
                shape = (self.num_heads, weight.shape[0] // self.num_heads, weight.shape[1])
            else:
                shape = (self.num_heads, weight.shape[0] // self.num_heads)
        else:
            shape = weight.shape
        return (zeros(shape, weight.context, dtype=weight.dtype,
                      stype=stype),  # mean
                zeros(shape, weight.context, dtype=weight.dtype,
                      stype=stype))  # variance

    def update(self, index, weight, grad, state):
        if self._verbose:
            import logging
            logging.info('rescale gradient factor = {}'.format(str(self.rescale_grad)))
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        t = self._index_update_count[index]
        name = self.idx2name[index]
        att = False
        for att_name in self.att_list:
             if att_name in name:
                 att = True
                 break

        with bulk(self._bulk):
            # preprocess grad
            #import logging
            grad *= self.rescale_grad
            #logging.info(name, grad.norm(ord=2).asscalar())
            if not att or not self._split_head:
                grad /= grad.astype('float32').norm(ord=2)
            else: 
                if len(weight.shape) == 2:
                    grad = grad.reshape((-4, self.num_heads, -1, 0))
                else:
                    grad = grad.reshape((-4, self.num_heads, -1))
                #grad_norm = sqrt(mx.nd.sum(square(grad), axis=0, exclude=True, keepdims=True))
                grad_norm = mx.nd.sum(mx.nd.abs(grad), axis=0, exclude=True, keepdims=True)
                grad /= grad_norm
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)

            mean, var = state
            mean *= self.beta1
            mean += (1. - self.beta1) * grad
            var *= self.beta2
            var += (1. - self.beta2) * square(grad)

            if not att or not self._split_head:
                r1 = weight.norm(ord=2)
            else:
                if len(weight.shape) == 2:
                    weight = weight.reshape((-4, self.num_heads, -1, 0))
                else:
                    weight = weight.reshape((-4, self.num_heads, -1))
                r1 = sqrt(mx.nd.sum(square(weight), axis=0, exclude=True, keepdims=True))
                
            if not self.bias_correction:
                r1 = minimum(maximum(r1, self.lower_bound), self.upper_bound)
                sqrt_var = sqrt(var)
                sqrt_var += self.epsilon
                g = mean / sqrt_var
                g += wd * weight
            else:
                # apply bias correction
                if self._use_bound:
                    upper_bound = self.upper_bound
                    if self._use_proj or self._adjust_bound:
                        if 'weight' in name:
                            upper_bound = min(upper_bound, 0.01 * math.sqrt(weight.size))
                        if 'classifier' in name or 'cls' in name:
                            upper_bound = min(upper_bound, 0.04 * math.sqrt(weight.size))
                    r1 = minimum(maximum(r1, self.lower_bound), upper_bound)
                mean_hat = mean / (1. - power(self.beta1, t))
                var_hat = var / (1. - power(self.beta2, t))
                if self._eps_after_sqrt:
                    sqrt(var_hat, out=var_hat)
                    var_hat += self.epsilon
                else:
                    var_hat += self.epsilon
                    sqrt(var_hat, out=var_hat)
                mean_hat /= var_hat
                if not self._use_proj or self._force_wd:
                    mean_hat += wd * weight
                g = mean_hat

            if not att or not self._split_head:
                r2 = g.norm(ord=2)
            else:
                r2 = sqrt(mx.nd.sum(square(g), axis=0, exclude=True, keepdims=True))

            # calculate lamb_trust_ratio
            ratio = r1 / r2
            # becomes NaN if ratio == NaN or 0, otherwise 0
            nan_or_zero = 1 - ratio / ratio
            r = where(nan_or_zero, ones_like(ratio), ratio)
            #lr *= r

            # update weight
            g *= lr * r
            weight[:] -= g

            #denom = 1 + wd * lr
            #weight[:] /= denom
            if self._use_proj:
                alpha = 0
                if 'weight' in name:
                    alpha = 0.01 * math.sqrt(weight.size)
                if 'classifier' in name or 'cls' in name:
                    alpha = 0.04 * math.sqrt(weight.size)
                if alpha:
                    import logging
                    # logging.info("before project: name = {}, norm = {}".format(name, weight.norm().asscalar()))
                    weight[:] = _projection(weight, var_hat, alpha=alpha)
                    # logging.info("after project: name = {}, norm = {}".format(name, weight.norm().asscalar()))

def grad_global_norm(parameters, max_norm):
    """Calculate the 2-norm of gradients of parameters, and how much they should be scaled down
    such that their 2-norm does not exceed `max_norm`.

    If gradients exist for more than one context for a parameter, user needs to explicitly call
    ``trainer.allreduce_grads`` so that the gradients are summed first before calculating
    the 2-norm.

    .. note::

        This function is only for use when `update_on_kvstore` is set to False in trainer.

    Example::

        trainer = Trainer(net.collect_params(), update_on_kvstore=False, ...)
        for x, y in mx.gluon.utils.split_and_load(X, [mx.gpu(0), mx.gpu(1)]):
            with mx.autograd.record():
                y = net(x)
                loss = loss_fn(y, label)
            loss.backward()
        trainer.allreduce_grads()
        norm, ratio = grad_global_norm(net.collect_params().values(), max_norm)
        trainer.update(batch_size * ratio)
        ...

    Parameters
    ----------
    parameters : list of Parameters

    Returns
    -------
    NDArray
      Total norm. Shape is (1,)
    NDArray
      Ratio for rescaling gradients based on max_norm s.t. grad = grad / ratio.
      If total norm is NaN, ratio will be NaN, too. Shape is (1,)
    NDArray
      Whether the total norm is finite. Shape is (1,)
    """
    # collect gradient arrays
    arrays = []
    idx = 0
    for p in parameters:
        if p.grad_req != 'null':
            p_grads = p.list_grad()
            arrays.append(p_grads[idx % len(p_grads)])
            idx += 1
    assert len(arrays) > 0, 'No parameter found available for gradient norm.'

    # compute gradient norms
    def _norm(array):
        # TODO(haibin) norm operator does not support fp16 safe reduction.
        # Issue is tracked at: https://github.com/apache/incubator-mxnet/issues/14126
        x = array.reshape((-1,)).astype('float32', copy=False)
        return nd.dot(x, x)

    norm_arrays = [_norm(arr) for arr in arrays]

    # group norm arrays by ctx
    def group_by_ctx(arr_list):
        groups = collections.defaultdict(list)
        for arr in arr_list:
            ctx = arr.context
            groups[ctx].append(arr)
        return groups
    norm_groups = group_by_ctx(norm_arrays)

    # reduce
    ctx, dtype = arrays[0].context, 'float32'
    norms = [nd.add_n(*g).as_in_context(ctx) for g in norm_groups.values()]
    total_norm = nd.add_n(*norms).sqrt()
    scale = total_norm / max_norm
    # is_finite = 0 if NaN or Inf, 1 otherwise.
    is_finite = nd.contrib.isfinite(scale)
    # if scale is finite, nd.maximum selects the max between scale and 1. That is,
    # 1 is returned if total_norm does not exceed max_norm.
    # if scale = NaN or Inf, the result of nd.minimum is undefined. Therefore, we use
    # choices.take to return NaN or Inf.
    scale_or_one = nd.maximum(nd.ones((1,), dtype=dtype, ctx=ctx), scale)
    choices = nd.concat(scale, scale_or_one, dim=0)
    chosen_scale = choices.take(is_finite)
    #import logging
    #logging.info("total norm = {}, max_norm = {}, scale = {} ".format(total_norm.asscalar(), max_norm.asscalar(), chosen_scale.asscalar()))
    return total_norm, chosen_scale, is_finite


class FP16Trainer:
    """ Trainer for mixed precision training.

    Parameters
    ----------
    trainer: gluon.Trainer
      the original gluon Trainer object for fp32 training.
    dynamic_loss_scale: bool. Default is True
      whether to use dynamic loss scaling. This is recommended for optimizing model
      parameters using FP16.
    loss_scaler_params : dict
        Key-word arguments to be passed to loss scaler constructor. For example,
        `{"init_scale" : 2.**15, "scale_window" : 2000, "tolerance" : 0.05}`
        for `DynamicLossScaler`.
        See each `LossScaler` for a list of supported arguments'
    """
    def __init__(self, trainer, dynamic_loss_scale=True, loss_scaler_params=None):
        if trainer._kvstore_params['update_on_kvstore'] is not False and trainer._kvstore:
            err = 'Only gluon.Trainer created with update_on_kvstore=False is supported.'
            raise NotImplementedError(err)
        self.fp32_trainer = trainer
        loss_scaler_params = loss_scaler_params if loss_scaler_params else {}
        self._scaler = DynamicLossScaler(**loss_scaler_params) if dynamic_loss_scale \
                       else StaticLossScaler(**loss_scaler_params)
        # if the optimizer supports NaN check, we can always defer the NaN check to the optimizer
        # TODO(haibin) this should be added via registry
        self._support_nan_check = trainer._optimizer.__class__.__name__ == 'BERTAdam'

    def backward(self, loss, verbose=False):
        """backward propagation with loss"""
        with mx.autograd.record():
            if isinstance(loss, (tuple, list)):
                ls = [l * self._scaler.loss_scale for l in loss]
            else:
                ls = loss * self._scaler.loss_scale
        if verbose:
            import logging
            #import byteps.mxnet as bps
            #logging.info('{} loss scale = {}'.format(bps.rank(), self._scaler.loss_scale))
            logging.info('loss scale = {}'.format(self._scaler.loss_scale))
        mx.autograd.backward(ls)

    def step(self, batch_size, max_norm=None, num_ctxs=None, verbose=False):
        """Makes one step of parameter update. Should be called after
        `fp16_optimizer.backward()`, and outside of `record()` scope.

        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        max_norm : NDArray, optional, default is None
            max value for global 2-norm of gradients.
        """
        if num_ctxs and num_ctxs > 1:
            self.fp32_trainer.allreduce_grads()
        step_size = batch_size * self._scaler.loss_scale
        if max_norm is not None:
            _, ratio, is_finite = grad_global_norm(self.fp32_trainer._params,
                                                   max_norm * self._scaler.loss_scale)
            #step_size = ratio * step_size
            if self._support_nan_check:
                self.fp32_trainer.update(step_size)
                overflow = is_finite.asscalar() < 1
            else:
                overflow = is_finite.asscalar() < 1
                if verbose:
                    import logging
                    import byteps.mxnet as bps
                    logging.info('{} overflow = {}, ratio = {}'.format(bps.rank(), overflow, ratio.asscalar()))
                if not overflow:
                    self.fp32_trainer.update(step_size)
        else:
            # TODO(haibin) optimize the performance when max_norm is not present
            # sequentially adding isnan/isinf results may be slow
            if self._support_nan_check:
                self.fp32_trainer.update(step_size)
                overflow = self._scaler.has_overflow(self.fp32_trainer._params)
            else:
                overflow = self._scaler.has_overflow(self.fp32_trainer._params)
                if not overflow:
                    self.fp32_trainer.update(step_size)
        # update scale based on overflow information
        self._scaler.update_scale(overflow)

class LossScaler:
    """Abstract loss scaler"""
    def has_overflow(self, params):
        """ detect inf and nan """
        is_not_finite = 0
        for param in params:
            if param.grad_req != 'null':
                grad = param.list_grad()[0]
                is_not_finite += mx.nd.contrib.isnan(grad).sum()
                is_not_finite += mx.nd.contrib.isinf(grad).sum()
        # NDArray is implicitly converted to bool
        if is_not_finite == 0:
            return False
        else:
            return True

    def update_scale(self, overflow):
        raise NotImplementedError()

class StaticLossScaler(LossScaler):
    """Static loss scaler"""
    def __init__(self, init_scale=1):
        self.loss_scale = init_scale

    def update_scale(self, overflow):
        """update loss scale"""

class DynamicLossScaler(LossScaler):
    """Class that manages dynamic loss scaling.

    There are two problems regarding gradient scale when fp16 is used for training.
    One is overflow: the fp16 gradient is too large that it causes NaN.
    To combat such an issue, we need to scale down the gradient when such an event
    is detected. The other is underflow: the gradient is too small such that the
    precision suffers. This is hard to detect though. What dynamic loss scaler does
    it that, it starts the scale at a relatively large value (e.g. 2**15).
    Everytime when a NaN is detected in the gradient, the scale is reduced (by default)
    by 2x. On the other hand, if a NaN is not detected for a long time
    (e.g. 2000 steps), then the scale is increased (by default) by 2x."""
    def __init__(self, init_scale=2.**15, scale_factor=2., scale_window=2000,
                 tolerance=0.01):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self._num_steps = 0
        self._last_overflow_iter = -1
        self._last_rescale_iter = -1
        self._overflows_since_rescale = 0

    def update_scale(self, overflow):
        """dynamically update loss scale"""
        import logging
        iter_since_rescale = self._num_steps - self._last_rescale_iter
        if overflow:
            logging.info('DynamicLossScaler: overflow detected.')
            self._last_overflow_iter = self._num_steps
            self._overflows_since_rescale += 1
            percentage = self._overflows_since_rescale / float(iter_since_rescale)
            # we tolerate a certrain amount of NaNs before actually scaling it down
            if percentage >= self.tolerance:
                self.loss_scale /= self.scale_factor
                self._last_rescale_iter = self._num_steps
                self._overflows_since_rescale = 0
                if self.loss_scale < 1 or True:
                    logging.info('DynamicLossScaler: overflow detected. set loss_scale = %s'%
                                  self.loss_scale)
        elif (self._num_steps - self._last_overflow_iter) % self.scale_window == 0:
            logging.info('DynamicLossScaler: underflow detected. set loss_scale = %s'%
                          self.loss_scale)
            self.loss_scale *= self.scale_factor
            self._last_rescale_iter = self._num_steps
        self._num_steps += 1
