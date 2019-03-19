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

import mxnet as mx
from sampler import RouteSearchSampler

__all__ = ['RouteSearcher']


class RouteSearcher(object):
    """Beam Search Translator

    Parameters
    ----------
    model : Routing model
    """
    def __init__(self, model, graph):
        self._model = model
        self._graph = graph
        self._sampler = RouteSearchSampler(
            decoder=self._decode_logprob,
            graph=graph)
        self._embeddings = None

    def _decode_logprob(self, step_input, neighbors, destinations, states):
        out, states, _ = self._model.encode_step(step_input, neighbors, destinations, self._embeddings, states)
        return mx.nd.log_softmax(out), states

    def search(self, sources, destinations):
        """Get the translation result given the input sentence.

        Parameters
        ----------
        sources : NDArray
            Shape  (batch_size, )
        destinations : NDArray
            Shape  (batch_size, )

        Returns
        -------
        samples : list[list[int]]
            Samples draw by route sampler.
        scores : list[int]
            Scores of the samples.
        """
        samples, scores = self._sampler(sources, destinations)
        return samples, scores

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, embeddings):
        self._embeddings = embeddings

