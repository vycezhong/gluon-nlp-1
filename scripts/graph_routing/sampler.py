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
"""Implements the route search sampler."""
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['RouteSearchSampler']

from multiprocessing.dummy import Pool as ThreadPool
import threading
import numpy as np
import mxnet as mx
from mxnet.gluon import HybridBlock
import heapq
from gluonnlp._constants import LARGE_NEGATIVE_FLOAT
import gluonnlp.data.batchify as btf
from gluonnlp.model.sequence_sampler import _expand_to_beam_size, _choose_states


# class RouteSearchSampler(object):
#     r"""Draw samples from the decoder by route search.
#
#     Parameters
#     ----------
#     decoder : callable
#         Function of the one-step-ahead decoder, should have the form::
#
#             outputs, new_states = decoder(step_input, states)
#
#         The outputs, input should follow these rules:
#
#         - step_input has shape (batch_size,),
#         - outputs has shape (batch_size, V),
#         - states and new_states have the same structure and the leading
#           dimension of the inner NDArrays is the batch dimension.
#     graph : Graph
#         It is used to index the neighbors of the node
#     """
#     def __init__(self, decoder, graph):
#         self._decoder = decoder
#         self._graph = graph
#         self._pool = ThreadPool(32)
#         self._lock = threading.Lock()
#
#     def _dijkstra(self, s, d):
#         q, seen = [(0, s.asscalar(), [])], set()
#         states = []
#         step_input = s
#         d_scalar = d.asscalar()
#         ctx = s.context
#         while q:
#             (cost, v1, path) = heapq.heappop(q)
#             if v1 not in seen:
#                 seen.add(v1)
#                 path = path + [v1]
#                 if v1 == d_scalar:
#                     return tuple([path, cost])
#                 step_input[0] = v1
#                 neighbors = self._graph.get_neighbors(v1)
#                 nd_neighbors = mx.nd.array([neighbors], ctx=ctx)
#                 with self._lock:
#                     log_probs, states = self._decoder(step_input, nd_neighbors, d, states)
#                 log_probs = log_probs.asnumpy()[0]
#                 for i in range(len(neighbors)):
#                     v2 = neighbors[i]
#                     c = - log_probs[i]
#                     if v2 not in seen:
#                         heapq.heappush(q, (cost + c, v2, path))
#         return tuple([[], float("inf")])
#
#     def _greedy(self, s, d):
#         step_input = s
#         v = s.asscalar()
#         ds = d.asscalar()
#         ctx = s.context
#         states = []
#         cost = 0
#         path = [v]
#         while v != ds:
#             step_input[0] = v
#             neighbors = self._graph.get_neighbors(v)
#             nd_neighbors = mx.nd.array([neighbors], ctx=ctx)
#             with self._lock:
#                 log_probs, states = self._decoder(step_input, nd_neighbors, d, states)
#             ind = int(mx.nd.argmax(log_probs, axis=1).asscalar())
#             v = neighbors[ind]
#             if v in path:
#                 break
#             path = path + [v]
#             cost += -log_probs[0, ind].asscalar()
#         return tuple([path, cost])
#
#     def __call__(self, sources, destinations):
#         sources = sources.astype('int32', copy=False)
#         destinations = destinations.astype('int32', copy=False)
#         samples, scores = list(zip(*self._pool.starmap(self._dijkstra, zip(sources, destinations))))
#         return samples, scores
#
#     def stop_threads(self):
#         self._pool.close()
#         self._pool.join()
#
#     def __del__(self):
#         self.stop_threads()


class _RouteSearchStepUpdate(HybridBlock):
    def __init__(self, beam_size, state_info,
                 prefix=None, params=None):
        super(_RouteSearchStepUpdate, self).__init__(prefix, params)
        self._beam_size = beam_size
        self._state_info = state_info

    def hybrid_forward(self, F, samples, destinations, valid_length, outputs, scores,
                       neighbors, valid_targets, beam_alive_mask, states,
                       target_size, batch_shift): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        F
        samples : NDArray or Symbol
            The current samples generated by beam search.
            Shape (batch_size, beam_size, L).
        destinations : NDArray or Symbol
            Destinations of the route. Shape (batch_size * beam_size,)
        valid_length : NDArray or Symbol
            The current valid lengths of the samples
        outputs : NDArray or Symbol
            Outputs from predictor. If from_logits was set to True in scorer, then it's the
            log probability of the current step. Else, it's the unnormalized outputs before
            softmax or log_softmax. Shape (batch_size * beam_size, target_size).
        scores : NDArray or Symbol
            The previous scores. Shape (batch_size, beam_size)
        neighbors : NDArray or Symbol
            The neighbor nodes of the current node. Shape (batch_size * beam_size, target_size)
        valid_targets : NDArray or Symbol
            Valid number of targets. Shape (batch_size * beam_size,)
        beam_alive_mask : NDArray or Symbol
            Shape (batch_size, beam_size)
        states : nested structure of NDArrays/Symbols
            Each NDArray/Symbol should have shape (N, ...) when state_info is None,
            or same as the layout in state_info when it's not None.
        target_size : NDArray or Symbol
            Shape (1,)
        batch_shift : NDArray or Symbol
            Contains [0, beam_size, 2 * beam_size, ..., (batch_size - 1) * beam_size].
            Shape (batch_size,)

        Returns
        -------
        new_samples : NDArray or Symbol or an empty list
            The updated samples.
            When single_step is True, it is an empty list.
            When single_step is False, shape (batch_size, beam_size, L + 1)
        new_valid_length : NDArray or Symbol
            Valid lengths of the samples. Shape (batch_size, beam_size)
        new_scores : NDArray or Symbol
            Shape (batch_size, beam_size)
        chosen_node_ids : NDArray or Symbol
            The chosen node ids of the step. Shape (batch_size, beam_size). If it's negative,
            no word will be appended to the beam.
        beam_alive_mask : NDArray or Symbol
            Shape (batch_size, beam_size)
        new_states : nested structure of NDArrays/Symbols
            Inner NDArrays have shape (batch_size * beam_size, ...)
        """
        beam_size = self._beam_size
        beam_alive_mask_bcast = F.expand_dims(beam_alive_mask, axis=2).astype(np.float32)
        outputs = F.SequenceMask(outputs,
                                 sequence_length=valid_targets,
                                 use_sequence_length=True, axis=1,
                                 value=LARGE_NEGATIVE_FLOAT)
        candidate_scores = F.broadcast_add(outputs.reshape(shape=(-4, -1, beam_size, 0)),
                                           F.expand_dims(scores, axis=-1))
        # Concat the candidate scores and the scores of the finished beams
        # The resulting candidate score will have shape (batch_size, beam_size * target_size + beam_size)
        candidate_scores = F.broadcast_mul(beam_alive_mask_bcast, candidate_scores) + \
                           F.broadcast_mul(1 - beam_alive_mask_bcast,
                                           F.ones_like(candidate_scores) * LARGE_NEGATIVE_FLOAT)
        finished_scores = F.where(beam_alive_mask,
                                  F.ones_like(scores) * LARGE_NEGATIVE_FLOAT, scores)
        candidate_scores = F.concat(candidate_scores.reshape(shape=(0, -1)),
                                    finished_scores, dim=1)
        # Get the top K scores
        new_scores, indices = F.topk(candidate_scores, axis=1, k=beam_size, ret_typ='both')
        indices = indices.astype(np.int32)
        use_prev = F.broadcast_greater_equal(indices, beam_size * target_size)
        chosen_node_ids = F.broadcast_mod(indices, target_size)
        beam_ids = F.where(use_prev,
                           F.broadcast_minus(indices, beam_size * target_size),
                           F.floor(F.broadcast_div(indices, target_size)))
        batch_beam_indices = F.broadcast_add(beam_ids, F.expand_dims(batch_shift, axis=1))
        # Update the samples and vaild_length
        selected_samples = F.take(samples.reshape(shape=(-3, 0)),
                                  batch_beam_indices.reshape(shape=(-1,)))
        chosen_neighbors = F.take(neighbors,
                                  batch_beam_indices.reshape(shape=(-1,)))
        chosen_node_ids = F.pick(chosen_neighbors,
                                 chosen_node_ids.reshape(shape=(-1, 1)),
                                 axis=1, keepdims=True).reshape(shape=(-4, -1, beam_size))
        # Determine loop
        loop = F.sum(F.broadcast_equal(F.expand_dims(chosen_node_ids, axis=2), samples), axis=2)
        chosen_node_ids = F.where(use_prev + loop,
                                  -F.ones_like(indices),
                                  chosen_node_ids)
        new_samples = F.concat(selected_samples,
                               chosen_node_ids.reshape(shape=(-1, 1)), dim=1)\
                       .reshape(shape=(-4, -1, beam_size, 0))
        new_valid_length = F.take(valid_length.reshape(shape=(-1,)),
                                  batch_beam_indices.reshape(shape=(-1,))).reshape((-1, beam_size))\
                           + 1 - use_prev
        # Update the states
        new_states = _choose_states(F, states, self._state_info, batch_beam_indices.reshape((-1,)))
        # Update the alive mask.
        beam_alive_mask = F.take(beam_alive_mask.reshape(shape=(-1,)),
                                 batch_beam_indices.reshape(shape=(-1,)))\
                              .reshape(shape=(-1, beam_size)) \
                          * (chosen_node_ids != destinations.reshape((-4, -1, beam_size)))
        beam_alive_mask = F.where(loop,
                                  F.zeros_like(beam_alive_mask),
                                  beam_alive_mask)
        # Set the score to large negative value if the loop is detected.
        new_scores = F.where(loop * beam_alive_mask,
                             F.ones_like(new_scores) * LARGE_NEGATIVE_FLOAT,
                             new_scores)
        return new_samples, new_valid_length, new_scores,\
               chosen_node_ids, beam_alive_mask, new_states


class RouteSearchSampler(object):
    r"""Draw samples from the decoder by route search.

    Parameters
    ----------
    beam_size : int
        The beam size.
    decoder : callable
        Function of the one-step-ahead decoder, should have the form::

            outputs, new_states = decoder(step_input, states)

        The outputs, input should follow these rules:

        - step_input has shape (batch_size,),
        - outputs has shape (batch_size, V),
        - states and new_states have the same structure and the leading
          dimension of the inner NDArrays is the batch dimension.
    graph : Graph
    """
    def __init__(self, beam_size, decoder, graph, state_info=None):
        self._beam_size = beam_size
        assert beam_size > 0,\
            'beam_size must be larger than 0. Received beam_size={}'.format(beam_size)
        self._decoder = decoder
        self._graph = graph
        if state_info is None:
            if hasattr(decoder, 'state_info'):
                state_info = decoder.state_info()
            else:
                state_info = None
        self._updater = _RouteSearchStepUpdate(beam_size=beam_size, state_info=state_info)
        self._updater.hybridize()
        self._pad = btf.Pad(dtype=np.int32)

    def __call__(self, sources, destinations, states):
        """Sample by beam search.

        Parameters
        ----------
        sources : NDArray
            The initial input of the decoder. Shape is (batch_size,).
        destinations : NDArray
            The destinations
        states : Object that contains NDArrays
            The initial states of the decoder.
        Returns
        -------
        samples : NDArray
            Samples draw by beam search. Shape (batch_size, beam_size, length). dtype is int32.
        scores : NDArray
            Scores of the samples. Shape (batch_size, beam_size). We make sure that scores[i, :] are
            in descending order.
        valid_length : NDArray
            The valid length of the samples. Shape (batch_size, beam_size). dtype will be int32.
        """
        batch_size = sources.shape[0]
        beam_size = self._beam_size
        ctx = sources.context
        # Tile the states and inputs to have shape (batch_size * beam_size, ...)
        if hasattr(self._decoder, 'state_info'):
            state_info = self._decoder.state_info(batch_size)
        else:
            state_info = None
        states = _expand_to_beam_size(states, beam_size=beam_size, batch_size=batch_size,
                                      state_info=state_info)
        step_input = _expand_to_beam_size(sources, beam_size=beam_size,
                                          batch_size=batch_size).astype(np.int32)
        destinations = _expand_to_beam_size(destinations, beam_size=beam_size,
                                            batch_size=batch_size).astype(np.int32)
        # All beams are initialized to alive
        # Generated samples are initialized to be the inputs
        # Except the first beam where the scores are set to be zero, all beams have -inf scores.
        # Valid length is initialized to be 1
        beam_alive_mask = mx.nd.ones(shape=(batch_size, beam_size), ctx=ctx, dtype=np.int32)
        valid_length = mx.nd.ones(shape=(batch_size, beam_size), ctx=ctx, dtype=np.int32)
        scores = mx.nd.zeros(shape=(batch_size, beam_size), ctx=ctx)
        if beam_size > 1:
            scores[:, 1:beam_size] = LARGE_NEGATIVE_FLOAT
        samples = step_input.reshape((batch_size, beam_size, 1))
        while True:
            neighbors = self._graph.get_neighbors(step_input.asnumpy())
            nd_neighbors = self._pad(neighbors).as_in_context(ctx)
            log_probs, new_states = self._decoder(step_input, nd_neighbors, destinations, states)
            if nd_neighbors.shape[1] < beam_size:
                pad_shape = (nd_neighbors.shape[0], beam_size - nd_neighbors.shape[1])
                nd_neighbors = mx.nd.concat(nd_neighbors,
                                            mx.nd.zeros(pad_shape, ctx=ctx, dtype=np.int32),
                                            dim=1)
                log_probs = mx.nd.concat(log_probs,
                                         mx.nd.ones(pad_shape, ctx=ctx) * LARGE_NEGATIVE_FLOAT,
                                         dim=1)
            target_size = mx.nd.array([log_probs.shape[1]], ctx=ctx, dtype=np.int32)
            valid_targets = mx.nd.array([len(neighbor) for neighbor in neighbors], ctx=ctx)
            batch_shift_nd = mx.nd.arange(0, batch_size * beam_size, beam_size, ctx=ctx,
                                          dtype=np.int32)
            samples, valid_length, scores, chosen_node_ids, beam_alive_mask, states = \
                self._updater(samples, destinations, valid_length, log_probs, scores,
                              nd_neighbors, valid_targets, beam_alive_mask,
                              new_states, target_size, batch_shift_nd)
            step_input = mx.nd.relu(chosen_node_ids).reshape((-1,))
            if mx.nd.sum(beam_alive_mask).asscalar() == 0:
                return samples, scores, valid_length

