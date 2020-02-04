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

"""Weight updating functions."""
import os
import warnings
import numpy
from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import zeros, NDArray, full
from mxnet.ndarray import lamb_update_phase1, lamb_update_phase2, mp_lamb_update_phase1, mp_lamb_update_phase2
from mxnet.ndarray.contrib import multi_lamb_update, multi_mp_lamb_update

__all__ = ['NLAMB']

@register
class NLAMB(Optimizer):
    """LAMB Optimizer.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6,
                 lower_bound=None, upper_bound=None, bias_correction=True, **kwargs):
        super(NLAMB, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bias_correction = bias_correction
        self.aggregate_num = max(1, min(45, int(os.getenv('MXNET_OPTIMIZER_AGGREGATION_SIZE', "45"))))

    def create_state(self, index, weight):
        stype = weight.stype
        dtype = weight.dtype
        return (zeros(weight.shape, weight.context, dtype=dtype, stype=stype),
                zeros(weight.shape, weight.context, dtype=dtype, stype=stype))

    def _update_impl(self, index, weight, grad, state, multi_precision=False):
        kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                  'bias_correction': self.bias_correction,
                  'rescale_grad': 1.}
        if isinstance(index, (tuple, list)):
            for g in grad:
                g *= self.rescale_grad
                g /= g.norm()
        else:
            grad *= self.rescale_grad
            grad /= grad.norm()

        if self.aggregate_num <= 1 or not isinstance(index, (tuple, list)):
            if isinstance(index, (tuple, list)):
                assert(len(index) == self.aggregate_num)
                index, weight, grad, state = index[0], weight[0], grad[0], state[0]
            assert(isinstance(weight, NDArray))
            assert(isinstance(grad, NDArray))
            self._update_count(index)
            lr = self._get_lr(index)
            wd = self._get_wd(index)
            t = self._index_update_count[index]
            weight_ptr = weight
            grad_ptr = grad
            if multi_precision:
                mean, var = state[1]
                weight32 = state[0]
            else:
                mean, var = state
            kwargs['t'] = t
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient

            if multi_precision:
                g = mp_lamb_update_phase1(weight_ptr, grad_ptr, mean, var, weight32, wd=wd, **kwargs)
                kwargs = {}
                if self.lower_bound:
                    kwargs['lower_bound'] = self.lower_bound
                if self.upper_bound:
                    kwargs['upper_bound'] = self.upper_bound
                r_1 = weight32.norm()
                r_2 = g.norm()
                mp_lamb_update_phase2(weight_ptr, g, r_1, r_2, weight32, lr=lr, out=weight_ptr, **kwargs)
            else:
                g = lamb_update_phase1(weight_ptr, grad_ptr, mean, var, wd=wd, **kwargs)
                kwargs = {}
                if self.lower_bound:
                    kwargs['lower_bound'] = self.lower_bound
                if self.upper_bound:
                    kwargs['upper_bound'] = self.upper_bound
                r_1 = weight_ptr.norm()
                r_2 = g.norm()
                lamb_update_phase2(weight_ptr, g, r_1, r_2, lr=lr, out=weight_ptr, **kwargs)
        else:
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient
            if self.lower_bound:
                kwargs['lower_bound'] = self.lower_bound
            if self.upper_bound:
                kwargs['upper_bound'] = self.upper_bound

            step_count, lrs, wds = [], [], []
            for i, w_i, g_i in zip(index, weight, grad):
                assert(isinstance(w_i, NDArray))
                assert(isinstance(g_i, NDArray))
                self._update_count(i)
                step_count.append(self._index_update_count[i])
                lrs.append(self._get_lr(i))
                wds.append(self._get_wd(i))

            updated_tensors = 0
            while updated_tensors < len(weight):
                sidx = updated_tensors
                eidx = min(updated_tensors + self.aggregate_num, len(weight))
                if not multi_precision:
                    mean, var = list(zip(*state[sidx:eidx]))
                    multi_lamb_update(weight[sidx:eidx],
                                      grad[sidx:eidx],
                                      mean, var,
                                      out=weight[sidx:eidx],
                                      step_count=step_count[sidx:eidx],
                                      lrs=lrs[sidx:eidx],
                                      wds=wds[sidx:eidx],
                                      **kwargs)
                else:
                    mean_var = list(zip(*state[sidx:eidx]))[1]
                    temp = list(zip(*mean_var))
                    mean = temp[0]
                    var = temp[1]
                    multi_mp_lamb_update(weight[sidx:eidx],
                                         grad[sidx:eidx],
                                         mean, var,
                                         list(zip(*state[sidx:eidx]))[0],
                                         out=weight[sidx:eidx],
                                         step_count=step_count[sidx:eidx],
                                         lrs=lrs[sidx:eidx],
                                         wds=wds[sidx:eidx],
                                         **kwargs)
                updated_tensors += self.aggregate_num

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        if not isinstance(index, (tuple, list)):
            use_multi_precision = self.multi_precision and weight.dtype == numpy.float16
        else:
            use_multi_precision = self.multi_precision and weight[0].dtype == numpy.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)
