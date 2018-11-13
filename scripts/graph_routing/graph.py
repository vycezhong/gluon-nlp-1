import numpy as np
from scipy.sparse import csr_matrix, dia_matrix

__all__ = ['Graph']


class Graph:
    """Graph for routing
    """
    def __init__(self, positions, neighbors, weights, normalize=False):
        """This model construct a Graph object
        The node id start from 1 to N.
        """
        assert len(positions) == len(neighbors) == len(weights), 'The size of positions should equal to ' \
                                       'the size of neighbors and weights. Received size of positions={}, ' \
                                       'size of neighbors={}, and size of weights={}'.format(len(positions),
                                                                                             len(neighbors),
                                                                                             len(weights))
        self._size = len(positions)
        self._positions = self.normalize(positions)
        self._neighbors = neighbors
        self._weights = weights
        self._adjacency_matrix, self._nonzeros = self._build_adjacency_matrix(normalize=normalize)

    def normalize(self, positions):
        mean = np.mean(positions, axis=0, keepdims=True)
        std = np.std(positions, axis=0, keepdims=True)
        return (positions - mean) / std

    def _build_adjacency_matrix(self, normalize=False):
        """This function construct an normalized adjacent matrix
        :return A tuple that represents a sparse tensor
        """
        # As only normalized weight is used in the model, to reduce time complexity, the normalized weight is stored.
        data = [1 / w if w != 0 else 0 for weight in self._weights for w in weight]
        row_ind = [i for i, neighbors in zip(range(self._size), self._neighbors) for _ in neighbors]
        col_ind = [neighbor for i, neighbors in zip(range(self._size), self._neighbors) for neighbor in neighbors]
        adjacency_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(self._size, self._size))
        adjacency_matrix = adjacency_matrix.transpose().tocsr()
        if normalize:
            degree_matrix = adjacency_matrix.sum(axis=1).squeeze()
            sqrt_inv_degree_matrix = dia_matrix((1 / np.sqrt(degree_matrix), 0), shape=(self._size, self._size))
            adjacency_matrix = sqrt_inv_degree_matrix * adjacency_matrix * sqrt_inv_degree_matrix
        return adjacency_matrix, len(data)

    def get_neighbors(self, nodes):
        """This function return the neighbor nodes
        :return [[int]] or [int]
        """
        if isinstance(nodes, (tuple, list, np.ndarray)):
            return [self._neighbors[node] for node in nodes]
        else:
            return self._neighbors[nodes]

    def get_neighbor_ind(self, src, tgt):
        """This function maps the tgt to the index of the neighbors of src
        :return [int] or int
        """
        if isinstance(src, (tuple, list)):
            assert len(src) == len(tgt), 'src and tgt should have the same length. ' \
                                         'len(src)={} and len(tgt)={}'.format(len(src), len(tgt))
            return [self._neighbors[src_node].index(tgt_node) for src_node, tgt_node in zip(src, tgt)]
        else:
            return self._neighbors[src].index(tgt)

    def get_weight(self, x, y):
        """This function return the link weight between two nodes x and y
        :return float
                link weight
        """

        neighbors = self._neighbors[x]
        weights = self._weights[x]

        index = neighbors.index(y)

        return weights[index]

    @property
    def size(self):
        return self._size

    @property
    def positions(self):
        return self._positions

    @property
    def adjacency_matrix(self):
        return self._adjacency_matrix

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def weights(self):
        return self._weights

    @property
    def nonzeros(self):
        return self._nonzeros
