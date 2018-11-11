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
from mxnet.gluon.loss import SoftmaxCELoss

__all__ = ['SoftmaxCEMaskedLoss']


class SoftmaxCEMaskedLoss(SoftmaxCELoss):
    """Wrapper of the SoftmaxCELoss that supports valid_length and valid_targets as the input

    """
    def hybrid_forward(self, F, pred, label, valid_length, valid_target): # pylint: disable=arguments-differ
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
        valid_target : Symbol or NDArray
            Shpe (batch_size, length)
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
        pred = F.SequenceMask(pred.reshape((-3, 0)), sequence_length=valid_target.reshape((-1)),
                              use_sequence_length=True, axis=1, value=-1e18).reshape_like(pred)
        return super(SoftmaxCEMaskedLoss, self).hybrid_forward(F, pred, label, sample_weight)