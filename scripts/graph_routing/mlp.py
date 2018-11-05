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

from mxnet.gluon import nn, Block, HybridBlock

__all__ = ['MLP']


class MLP(HybridBlock):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, y):
        x = F.expand_dims(x.reshape((-3, 0)), axis=1)
        y = y.reshape((-3, 0, 0))
        return F.squeeze(F.batch_dot(x, y, transpose_b=True), axis=1)
