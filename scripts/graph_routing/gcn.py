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

__all__ = ['GCNLayer', 'GCN']

class GCNLayer(HybridBlock):
    def __init__(self, embed_size, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.dense1 = nn.Dense(embed_size, use_bias=False, flatten=False)
        self.dense2 = nn.Dense(embed_size, use_bias=False, flatten=False)

    def hybrid_forward(self, F, x, adjacency_matrix):
        x1 = self.dense1(x)
        x2 = self.dense2(x)
        x2 = F.sparse.dot(adjacency_matrix, x2)
        return F.Activation(x1 + x2, act_type='relu', name='activation')


class GCN(HybridBlock):
    def __init__(self, graph_size, embed_size, num_layers, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.graph_size = graph_size
        self.embed_size = embed_size
        self.layers = nn.HybridSequential('GCN')
        for _ in range(num_layers):
            self.layers.add(GCNLayer(embed_size))

    def hybrid_forward(self, F, x, adjacency_matrix):
        #x0 = F.ones((self.graph_size, self.embed_size))
        x0 = F.full((self.graph_size, self.embed_size), val=1./self.embed_size)
        x = F.concat(x0, x, dim=1)
        for layer in self.layers:
            x = layer(x, adjacency_matrix)
        return x
