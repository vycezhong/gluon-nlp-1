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

import os
import numpy as np
import random
import networkx as nx
import pickle
from scipy import spatial
from collections import defaultdict
import threading
import queue
import heapq
from graph import Graph

data_dir = '/data'


def dijkstra(g, s, d):
    """This find the shortest path between source and destination.
    :param s: int
            source node
           d: int
            destination node
    :return [float, [int]]
            shortest path length and the nodes in path
    """
    q, seen = [(0, s, [])], set()
    while q:
        (cost, v1, path) = heapq.heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = path + [v1]
            if v1 == d:
                return tuple([cost, path])

            for c, v2 in g.get(v1, ()):
                if v2 not in seen:
                    heapq.heappush(q, (cost + c, v2, path))

    return tuple([float("inf"), []])

def set_weights(edges):
    weights = []
    for i, edges_node in enumerate(edges):
        weights_node = (np.random.randint(1, 11, size=len(edges_node)) / 10).tolist()
        if weights_node == []:
            edges[i] = [i + 1]
            weights_node = [0]
        else:
            for j, e in enumerate(edges_node):
                if e < i + 1:
                    weights_node[j] = weights[e - 1][edges[e - 1].index(i + 1)]

        weights.append(weights_node)
    return weights

def generate_spatial_graph(n):
    r = 0.15
    positions = np.random.rand(n, 2)
    kdtree = spatial.KDTree(positions)
    pairs = kdtree.query_pairs(r)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(list(pairs))
    pos = dict(zip(range(n), positions))
    #nx.draw(G, pos)
    #plt.show()
    #edges = G.adjacency_list()
    #edges = [(np.array(edge)).tolist() for edge in edges]
    #weights = set_weights(positions, edges)
    #graph = Graph(positions, edges, weights)


def load_didi_graph():
    print('loading graph ...')
    with open('data/data.pkl', 'rb') as f:
        data = pickle.load(f)
    graph = Graph(data['nodes'], data['edges'], data['weights'], normalize=True)
    return graph


def generate_tracjectories(graph, sample_size):
    num_workers = 20
    samples_per_worker = sample_size // num_workers
    out_queue = queue.Queue(-1)
    threads = []

    neighbors = graph.neighbors
    weights = graph.weights
    g = defaultdict(list)
    for i, nw in enumerate(zip(neighbors, weights)):
        for n, w in zip(*nw):
            g[i].append((w, n))

    def _push_next(out_queue, graph_size, g, samples):
        source_destination_pairs = np.random.randint(graph_size, size=(samples, 2))
        for s, d in source_destination_pairs:
            if s == d:
                continue
            cost, path = dijkstra(g, s, d)
            if cost != float("inf"):
                out_queue.put(path)

    arg = (out_queue, graph.size, g, samples_per_worker)
    for _ in range(num_workers):
            thread = threading.Thread(target=_push_next, args=arg)
            threads.append(thread)
            thread.start()
    for thread in threads:
        thread.join()
    trajectories = list(out_queue.queue)
    return trajectories


def get_path(name):
        return os.path.join(data_dir, "routing_{}_syn.npz".format(name))

if __name__ == '__main__':
    graph = load_didi_graph()
    print('generating data')
    trajectories = generate_tracjectories(graph, 1000000)
    print('number of trajectories={:d}'.format(len(trajectories)))
    print('saving data')
    with open('data/synthetic.pkl', 'wb') as f:
        pickle.dump(trajectories, f, pickle.HIGHEST_PROTOCOL)


