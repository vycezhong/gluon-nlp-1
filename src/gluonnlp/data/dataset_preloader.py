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

# pylint: disable=ungrouped-imports
"""DatasetLoader. An extension of Gluon data loader that allows
reading and processing multiple files on-the-fly.
"""

__all__ = ['DatasetPreLoader']

import io
import pickle
import warnings
import multiprocessing
from functools import partial
from mxnet import context
from mxnet.gluon.data.dataloader import ForkingPickler, _as_in_context
from mxnet.gluon.data.dataloader import default_mp_batchify_fn, default_batchify_fn
from .stream import _PathDataset


_datasets = None
def _initialize_batch_worker(datasets):
    global _datasets
    _datasets = datasets


def _dataset_worker_fn(urls, dataset_fn, batch_sampler_fn):
    """Function to generate datasets and batch sampler for each worker."""
    dataset = dataset_fn(urls)
    batch_sampler = batch_sampler_fn(dataset)
    return dataset, batch_sampler


def _batch_worker_fn(samples, batchify_fn, idx):
    """Function for processing data in worker process."""
    # pylint: disable=unused-argument
    # it is required that each worker process has to fork a new MXIndexedRecordIO handle
    # preserving dataset as global variable can save tons of overhead and is safe in new process
    global _datasets
    dataset = _datasets[idx]
    if isinstance(samples[0], (list, tuple)):
        batch = [batchify_fn([dataset[i] for i in shard]) for shard in samples]
    else:
        batch = batchify_fn([dataset[i] for i in samples])
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(batch)
    return buf.getvalue()


class _MultiBatchWorkerIter:
    """Internal multi-worker iterator for DataLoader."""
    def __init__(self, worker_pool, batchify_fn, datasets, batch_samplers,
                 file_sampler=None, pin_memory=False, worker_fn=_batch_worker_fn, prefetch=0):
        self._worker_pool = worker_pool
        self._batchify_fn = batchify_fn
        self._data_buffer = {}
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._worker_fn = worker_fn
        self._pin_memory = pin_memory
        self._prefetch = prefetch
        self._datasets = datasets
        self._batch_samplers = batch_samplers
        self._file_sampler = iter(file_sampler)
        self._batch_iter = None
        self._dataset_idx = None

        # pre-fetch
        for _ in range(self._prefetch):
            self._push_next()

    def _next_dataset(self):
        dataset_idx = next(self._file_sampler, None)
        if dataset_idx is None:
            return None
        self._dataset_idx = dataset_idx
        batch_sampler = self._batch_samplers[self._dataset_idx]
        return batch_sampler

    def _push_next(self):
        """Assign next batch workload to workers."""
        if self._batch_iter is not None:
            r = next(self._batch_iter, None)
        else:
            r = None
        if r is None:
            result = self._next_dataset()
            if result is None:
                return
            else:
                batch_sampler = result
                self._batch_iter = iter(batch_sampler)
                self._push_next()
        else:
            async_ret = self._worker_pool.apply_async(
                self._worker_fn, (r, self._batchify_fn, self._dataset_idx))
            self._data_buffer[self._sent_idx] = async_ret
            self._sent_idx += 1

    def __next__(self):
        self._push_next()
        if self._rcvd_idx == self._sent_idx:
            assert not self._data_buffer, 'Data buffer should be empty at this moment'
            raise StopIteration

        assert self._rcvd_idx < self._sent_idx, 'rcvd_idx must be smaller than sent_idx'
        assert self._rcvd_idx in self._data_buffer, 'fatal error with _push_next, rcvd_idx missing'
        ret = self._data_buffer.pop(self._rcvd_idx)
        batch = pickle.loads(ret.get())
        if self._pin_memory:
            batch = _as_in_context(batch, context.cpu_pinned())
        self._rcvd_idx += 1
        return batch

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


class DatasetPreLoader:
    """Loads data from a list of datasets and returns mini-batches of data.

    One dataset is loaded at a time.

    Parameters
    ----------
    file_patterns: str
        Path to the input text files.
    file_sampler : str or gluon.data.Sampler, defaults to 'random'
        The sampler used to sample a file. The following string values are supported:

        - 'sequential': SequentialSampler
        - 'random': RandomSampler
    dataset_fn : DatasetFn, callable
        Callable object to generate a gluon.data.Dataset given a url.
    batch_sampler_fn : SamplerFn, callable
        Callable object to generate a gluon.data.sampler.Sampler given a dataset.
    dataset_params : dict, default is None
        Dictionary of parameters passed to dataset_fn.
    batch_sampler_params : dict, default is None
        Dictionary of parameters passed to batch_sampler_fn.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch. Defaults to `default_batchify_fn`::

            def default_batchify_fn(data):
                if isinstance(data[0], nd.NDArray):
                    return nd.stack(*data)
                elif isinstance(data[0], tuple):
                    data = zip(*data)
                    return [default_batchify_fn(i) for i in data]
                else:
                    data = np.asarray(data)
                    return nd.array(data, dtype=data.dtype)
    num_dataset_workers : int
        Number of worker process for dataset creation.
    num_batch_workers : int
        Number of worker process for batch creation.
    pin_memory : boolean, default False
        If ``True``, the dataloader will copy NDArrays into pinned memory
        before returning them. Copying from CPU pinned memory to GPU is faster
        than from normal CPU memory. At the same time, it increases GPU memory.
    circle_length : int, default is 1
        The number of files to be read at the same time. When `circle_length` is larger than 1,
        we merge `circle_length` number of files.
    dataset_prefetch : int, default is `num_dataset_workers`
        The number of prefetching datasets only works if `num_workers` > 0.
        If `prefetch` > 0, it allow worker process to prefetch certain datasets before
        acquiring data from iterators.
        Note that using large prefetching batch will provide smoother bootstrapping performance,
        but will consume more memory. Using smaller number may forfeit the purpose of using
        multiple worker processes, try reduce `num_dataset_workers` in this case.
        By default it defaults to `num_dataset_workers`.
    batch_prefetch : int, default is `num_batch_workers * 2`
        The number of prefetching batches only works if `num_workers` > 0.
        If `prefetch` > 0, it allow worker process to prefetch certain batches before
        acquiring data from iterators.
        Note that using large prefetching batch will provide smoother bootstrapping performance,
        but will consume more shared_memory. Using smaller number may forfeit the purpose of using
        multiple worker processes, try reduce `num_batch_workers` in this case.
        By default it defaults to `num_batch_workers * 2`.
    dataset_cached : bool, default is False
        Whether or not to cache last processed dataset. Each processed dataset can
        only be cached for once. When there is no new available processed dataset to be fetched,
        we pop a cached processed dataset.
    num_max_dataset_cached : int, default is 0
        Maximum number of cached datasets. It is valid only if `dataset_cached` is True
    """
    def __init__(self, file_patterns, file_sampler,
                 dataset_fn=None, batch_sampler_fn=None,
                 dataset_params=None, batch_sampler_params=None, batchify_fn=None,
                 num_dataset_workers=0, num_batch_workers=0,
                 pin_memory=False, circle_length=1,
                 dataset_prefetch=None, batch_prefetch=None,
                 dataset_cached=False, num_max_dataset_cached=0):
        assert num_dataset_workers >= 0, \
               'num_dataset_workers must be non-negative'
        assert num_batch_workers >= 0, \
               'num_batch_workers must be non-negative'
        if num_batch_workers > 0:
            assert num_dataset_workers > 0, \
                'num_dataset_workers must be positive when num_batch_workers > 0'
        else:
            if num_dataset_workers > 0:
                warnings.warn('The multi-processing functionalities for both dataset and'
                              ' batch sampling are disabled when num_batch_workers=0 though '
                              'num_dataset_workers={} > 0'.format(num_dataset_workers))
        assert circle_length >= 1, \
               'circle_length must be larger than or equal to 1'
        if dataset_cached:
            assert num_max_dataset_cached > 0, \
                'When dataset_cached is True, num_max_dataset_cached must be positive'

        self._dataset = _PathDataset(file_patterns)
        self._file_sampler = file_sampler

        assert dataset_fn is not None, 'dataset_fn is not given.'
        assert batch_sampler_fn is not None, 'batch_sampler_fn is not given.'
        if dataset_params is not None:
            self._dataset_fn = partial(dataset_fn, **dataset_params)
        else:
            self._dataset_fn = dataset_fn
        if batch_sampler_params is not None:
            self._batch_sampler_fn = partial(batch_sampler_fn, **batch_sampler_params)
        else:
            self._batch_sampler_fn = batch_sampler_fn

        self._num_batch_workers = num_batch_workers
        self._batch_prefetch = max(0, int(batch_prefetch) \
                if batch_prefetch is not None else 2 * self._num_batch_workers)

        self._pin_memory = pin_memory

        self._datasets = {}
        self._batch_samplers = {}
        for i, idx in enumerate(iter(self._file_sampler)):
            if i > 1:
                break
            dataset, batch_sampler = _dataset_worker_fn(self._dataset[idx],
                                                        self._dataset_fn,
                                                        self._batch_sampler_fn)
            self._datasets[idx] = dataset
            self._batch_samplers[idx] = batch_sampler

        self._batch_worker_pool = None
        if self._num_batch_workers > 0:
            self._batch_worker_pool = multiprocessing.Pool(self._num_batch_workers,
                                                           initializer=_initialize_batch_worker,
                                                           initargs=[self._datasets])
        if batchify_fn is None:
            if self._num_batch_workers > 0:
                self._batchify_fn = default_mp_batchify_fn
            else:
                self._batchify_fn = default_batchify_fn
        else:
            self._batchify_fn = batchify_fn

    def __iter__(self):
        if self._num_batch_workers == 0:
            def _same_process_iter():
                urls = []
                dataset = [self._dataset[i] for i in iter(self._file_sampler)]
                for i, url in enumerate(dataset):
                    urls.append(url)
                    if i < len(dataset) - 1:
                        if len(urls) < self._circle_length:
                            continue
                    if self._circle_length == 1:
                        urls = urls[0]
                    dataset, batch_sampler = _dataset_worker_fn(urls, self._dataset_fn,
                                                                self._batch_sampler_fn)
                    for batch in batch_sampler:
                        ret = self._batchify_fn([dataset[idx] for idx in batch])
                        if self._pin_memory:
                            ret = _as_in_context(ret, context.cpu_pinned())
                        yield ret
                    urls = []
            return _same_process_iter()

        return _MultiBatchWorkerIter(self._batch_worker_pool, self._batchify_fn, self._datasets, self._batch_samplers,
                                     file_sampler=self._file_sampler, pin_memory=self._pin_memory, worker_fn=_batch_worker_fn,
                                     prefetch=self._batch_prefetch)

    def __del__(self):
        if self._batch_worker_pool:
            assert isinstance(self._batch_worker_pool, multiprocessing.pool.Pool)
            self._batch_worker_pool.terminate()
