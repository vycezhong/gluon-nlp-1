import mxnet as mx
from sampler import RouteSearchSampler


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
        self._embedding = None

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
        return self._embedding

    @embeddings.setter
    def embeddings(self, embeddings):
        self._embeddings = embeddings

