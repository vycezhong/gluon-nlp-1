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
"""Implements the beam search sampler."""
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['RouteSearchSampler']

from multiprocessing.dummy import Pool as ThreadPool
import threading
import mxnet as mx
import heapq


class RouteSearchSampler(object):
    r"""Draw samples from the decoder by route search.

    Parameters
    ----------
    decoder : callable
        Function of the one-step-ahead decoder, should have the form::

            outputs, new_states = decoder(step_input, states)

        The outputs, input should follow these rules:

        - step_input has shape (batch_size,),
        - outputs has shape (batch_size, V),
        - states and new_states have the same structure and the leading
          dimension of the inner NDArrays is the batch dimension.
    graph : Graph
        It is used to index the neighbors of the node
    """
    def __init__(self, decoder, graph):
        self._decoder = decoder
        self._graph = graph
        self._pool = ThreadPool(32)
        self._lock = threading.Lock()

    def _dijkstra(self, s, d):
        q, seen = [(0, s.asscalar(), [])], set()
        states = []
        step_input = s
        d_scalar = d.asscalar()
        ctx = s.context
        while q:
            (cost, v1, path) = heapq.heappop(q)
            if v1 not in seen:
                seen.add(v1)
                path = path + [v1]
                if v1 == d_scalar:
                    return tuple([path, cost])
                step_input[0] = v1
                neighbors = self._graph.get_neighbors(v1)
                nd_neighbors = mx.nd.array([neighbors], ctx=ctx)
                with self._lock:
                    log_probs, states = self._decoder(step_input, nd_neighbors, d, states)
                log_probs = log_probs.asnumpy()[0]
                for i in range(len(neighbors)):
                    v2 = neighbors[i]
                    c = - log_probs[i]
                    if v2 not in seen:
                        heapq.heappush(q, (cost + c, v2, path))
        return tuple([[], float("inf")])

    def _greedy(self, s, d):
        cost = 0
        path = []
        step_input = s
        v = s.asscalar()
        ds = d.asscalar()
        ctx = s.context
        while v != ds:
            step_input[0] = v
            neighbors = self._graph.get_neighbors(v)
            nd_neighbors = mx.nd.array([neighbors], ctx=ctx)
            with self._lock:
                log_probs, states = self._decoder(step_input, nd_neighbors, d, states)

    def __call__(self, sources, destinations):
        sources = sources.astype('int32', copy=False)
        destinations = destinations.astype('int32', copy=False)
        samples, scores = list(zip(*self._pool.starmap(self._greedy, zip(sources, destinations))))
        return samples, scores

    def stop_threads(self):
        self._pool.close()
        self._pool.join()

    def __del__(self):
        self.stop_threads()

