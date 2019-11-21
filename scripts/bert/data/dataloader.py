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
"""Dataset generator."""

__all__ = ['DatasetLoader']

import io
import pickle
import logging
import warnings
import multiprocessing
import gluonnlp as nlp
from multiprocessing.managers import BaseManager
from functools import partial
from mxnet import context
from mxnet.gluon.data import ArrayDataset
from mxnet.gluon.data.dataloader import ForkingPickler, _as_in_context
from mxnet.gluon.data.dataloader import default_mp_batchify_fn, default_batchify_fn
from gluonnlp.data.stream import _PathDataset

try:
    from .create_pretraining_data import create_training_instances
except ImportError:
    from create_pretraining_data import create_training_instances


def prepare_pretrain_npy_dataset(filename, allow_pickle=False):
    """Create dataset based on the files"""
    assert not isinstance(filename, (list, tuple)), \
        'When .npy/.npz data file is loaded, filename must be a string.'
    logging.debug('start to load files %s ...', filename)
    dataset = nlp.data.NumpyDataset(filename)
    return dataset


def prepare_pretrain_text_dataset(filename, tokenizer, max_seq_length, short_seq_prob,
                                  masked_lm_prob, max_predictions_per_seq, whole_word_mask,
                                  vocab, num_workers=1, worker_pool=None):
    """Create dataset based on the files"""
    dupe_factor = 1
    if not isinstance(filename, (list, tuple)):
        filename = [filename]
    logging.debug('start to load files %s ...', filename)
    instances = create_training_instances((filename, tokenizer, max_seq_length,
                                           short_seq_prob, masked_lm_prob,
                                           max_predictions_per_seq,
                                           whole_word_mask, vocab,
                                           dupe_factor, num_workers,
                                           worker_pool, None))
    return instances


def prepare_pretrain_bucket_sampler(dataset, batch_size, shuffle=False,
                                    num_ctxes=1, num_buckets=1):
    """Create data sampler based on the dataset"""
    if isinstance(dataset, nlp.data.NumpyDataset):
        lengths = dataset.get_field('valid_lengths')
    else:
        lengths = dataset.transform(lambda input_ids, segment_ids, masked_lm_positions, \
                                           masked_lm_ids, masked_lm_weights, \
                                           next_sentence_labels, valid_lengths: \
                                        valid_lengths, lazy=False)
    # calculate total batch size for all GPUs
    batch_size = batch_size * num_ctxes
    sampler = nlp.data.FixedBucketSampler(lengths,
                                          batch_size=batch_size,
                                          num_buckets=num_buckets,
                                          ratio=0,
                                          shuffle=shuffle)
    logging.debug('Sampler created for a new dataset:\n%s', sampler.stats())
    return sampler


def _dataset_worker_fn(url, dataset_fn, batch_sampler_fn):
    """Function to generate the dataset and batch sampler for each worker."""
    dataset = dataset_fn(url)
    batch_sampler = batch_sampler_fn(dataset)
    return dataset, batch_sampler


def _batch_worker_fn(samples, batchify_fn, dataset=None):
    """Function for processing data in worker process."""
    # pylint: disable=unused-argument
    # it is required that each worker process has to fork a new MXIndexedRecordIO handle
    # preserving dataset as global variable can save tons of overhead and is safe in new process
    if isinstance(samples[0], (list, tuple)):
        batch = [batchify_fn([dataset[i] for i in shard]) for shard in samples]
    else:
        batch = batchify_fn([dataset[i] for i in samples])
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(batch)
    return buf.getvalue()


class _MultiBatchWorkerIter:
    """Internal multi-worker iterator for DataLoader."""
    def __init__(self, worker_pool, batchify_fn, dataset_iter=None,
                 pin_memory=False, worker_fn=_batch_worker_fn, prefetch=0):
        self._worker_pool = worker_pool
        self._batchify_fn = batchify_fn
        self._data_buffer = {}
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._dataset_iter = iter(self._dataset_iter)
        self._dataset_iter = dataset_iter
        self._worker_fn = worker_fn
        self._pin_memory = pin_memory
        self._prefetch = prefetch

    def _next_dataset(self):
        try:
            dataset, batch_sampler = next(self._dataset_iter)
        except StopIteration:
            return None
        return dataset, batch_sampler

    def _push_next(self):
        """Assign next batch workload to workers."""
        r = next(self._batch_iter, None)
        if r is None:
            result = self._next_dataset()
            if result is None:
                return
            else:
                dataset, batch_iter = result
                self._dataset = dataset
                self._batch_iter = batch_iter
                for _ in range(self._prefetch):
                    self._push_next()
        else:
            async_ret = self._worker_pool.apply_async(
                self._worker_fn, (r, self._batchify_fn, self._dataset))
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
        batch = pickle.loads(ret.get()) if self._dataset is None else ret.get()
        if self._pin_memory:
            batch = _as_in_context(batch, context.cpu_pinned())
        self._rcvd_idx += 1
        return batch

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


class _MultiDatasetWorkerIter:
    """Internal multi-worker iterator for DataLoader."""
    def __init__(self, worker_pool, file_sampler,
                 dataset_fn, batch_sampler_fn,
                 worker_fn=_dataset_worker_fn,
                 prefetch=0, dataset=None, circle_length=1,
                 manager=None):
        self._worker_pool = worker_pool
        self._dataset_fn = dataset_fn
        self._batch_sampler_fn = batch_sampler_fn
        self._worker_fn = worker_fn
        self._prefetch = prefetch
        self._circle_length = circle_length

        # manager for creating shared memory
        self._manager = manager

        # send and receive index for datasets
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._data_buffer = {}

        self._dataset = [dataset[i] for i in iter(file_sampler)]
        self._num_datasets = len(self._dataset)

        # pre-fetch
        for _ in range(self._prefetch):
            self._push_next_dataset()

    def _push_next_dataset(self):
        """Assign next dataset workload to workers."""
        current_dataset_idx = self._sent_idx * self._circle_length
        if current_dataset_idx < len(self._dataset):
            circle_length = min(self._circle_length,
                                len(self._dataset) - current_dataset_idx)
            if self._circle_length > 1:
                url = [self._dataset[current_dataset_idx + i] for i in range(circle_length)]
            else:
                url = self._dataset[current_dataset_idx]
        else:
            return
        # push to worker asynchronously
        async_ret = self._worker_pool.apply_async(
            self._worker_fn, (url, self._dataset_fn, self._batch_sampler_fn))
        # data buffer stores the async result
        self._data_buffer[self._sent_idx] = async_ret
        self._sent_idx += 1

    def _next_dataset(self):
        """Retrieve the next dataset. Returns None if no dataset is available."""
        if self._rcvd_idx == self._sent_idx:
            assert not self._data_buffer, 'Data buffer should be empty at this moment'
            return None

        assert self._rcvd_idx < self._sent_idx, \
               'rcvd_idx must be smaller than sent_idx'
        assert self._rcvd_idx in self._data_buffer, \
               'fatal error with _next_dataset, rcvd_idx missing'

        ret = self._data_buffer.pop(self._rcvd_idx)
        dataset, batch_sampler = ret.get()
        if self._manager:
            dataset = self._manager.ArrayDataset(dataset)
        self._rcvd_idx += 1
        return dataset, batch_sampler

    def __next__(self):
        """Next dataset"""
        self._push_next_dataset()
        result = self._next_dataset()

        if result is None:
            raise StopIteration

        return result

    def next(self):
        """Next dataset"""
        return self.__next__()

    def __iter__(self):
        """Returns the iterator object"""
        return self


def _manager_register():
    BaseManager.register('ArrayDataset', ArrayDataset)


class DatasetLoader:
    """Loads data from a list of datasets and returns mini-batches of data.

    One dataset is loaded at a time.

    Parameters
    ----------
    file_pattern: str
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
        than from normal CPU memory.
    circle_length : int, default is 1
        The number of files to be read at the same time. When circle_length is larger than 1,
        we merge circle_length files.
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
    """
    def __init__(self, file_patterns, file_sampler,
                 dataset_fn=None, batch_sampler_fn=None,
                 dataset_params=None, batch_sampler_params=None, batchify_fn=None,
                 num_dataset_workers=0, num_batch_workers=0,
                 pin_memory=False, circle_length=1,
                 dataset_prefetch=None, batch_prefetch=None):
        assert num_dataset_workers >= 0, \
               'num_dataset_workers must be non-negative'
        assert num_batch_workers >= 0, \
               'num_batch_workers must be non-negative'
        if num_batch_workers > 0:
            assert num_dataset_workers > 0, \
                'num_dataset_workers must be positive when num_batch_workers > 0'
        else:
            if num_dataset_workers > 0:
                warnings.warn('The multi-processing for both dataset and'
                              ' batch sampling is disabled when num_dataset_workers=0 though '
                              'num_batch_workers={} > 0'.format(num_batch_workers))
        assert self._circle_length >= 1, \
               'circle_length must be larger than or equal to 1'
        self._dataset = _PathDataset(file_patterns)
        self._file_sampler = file_sampler
        if dataset_fn is None:
            dataset_fn = prepare_pretrain_text_dataset
            logging.info('dataset_fn is not given. Set it to default function prepare_pretrain_text_dataset')
        if batch_sampler_fn is None:
            batch_sampler_fn = prepare_pretrain_bucket_sampler
            logging.info('batch_sampler_fn is not given. Set it to default function prepare_pretrain_bucket_sampler')
        if dataset_params is not None:
            self._dataset_fn = partial(dataset_fn, **dataset_params)
        else:
            self._dataset_fn = dataset_fn
        if batch_sampler_params is not None:
            self._batch_sampler_fn = partial(batch_sampler_fn, **batch_sampler_params)
        else:
            self._batch_sampler_fn = batch_sampler_fn
        self._num_dataset_workers = num_dataset_workers
        self._num_batch_workers = num_batch_workers
        self._dataset_prefetch \
            = max(0, int(dataset_prefetch) if dataset_prefetch is not None else self._num_dataset_workers)
        self._batch_prefetch \
            = max(0, int(batch_prefetch) if batch_prefetch is not None else 2 * self._num_batch_workers)
        self._pin_memory = pin_memory
        self._circle_length = circle_length
        self._manager = None
        self._dataset_worker_pool = None
        if self._num_dataset_workers > 0:
            _manager_register()
            self._manager = BaseManager()
            self._manager.start()
            self._dataset_worker_pool = multiprocessing.Pool(self._num_dataset_workers)
        self._batch_worker_pool = None
        if self._num_batch_workers > 0:
            self._batch_worker_pool = multiprocessing.Pool(self._num_batch_workers)
        if batchify_fn is None:
            if self._num_batch_workers > 0:
                self._batchify_fn = default_mp_batchify_fn
            else:
                self._batchify_fn = default_batchify_fn
        else:
            self._batchify_fn = batchify_fn

    def __iter__(self):
        if self._num_dataset_workers == 0:
            def _same_process_iter():
                urls = []
                dataset = [self._dataset[i] for i in iter(self._file_sampler)]
                for i, url in enumerate(dataset):
                    urls.append(url)
                    if i < len(dataset) - 1:
                        if len(urls) < self._circle_length:
                            continue
                    dataset, batch_sampler = _dataset_worker_fn(urls, self._dataset_fn, self._batch_sampler_fn)
                    for batch in batch_sampler:
                        ret = self._batchify_fn([dataset[idx] for idx in batch])
                        if self._pin_memory:
                            ret = _as_in_context(ret, context.cpu_pinned())
                        yield ret
                    urls = []
            return _same_process_iter()

        # multi-worker
        dataset_iter = _MultiDatasetWorkerIter(self._dataset_worker_pool,
                                               worker_fn=_dataset_worker_fn,
                                               dataset=self._dataset,
                                               file_sampler=self._file_sampler,
                                               dataset_fn=self._dataset_fn,
                                               batch_sampler_fn=self._batch_sampler_fn,
                                               prefetch=self._dataset_prefetch,
                                               circle_length=self._circle_length,
                                               manager=self._manager)
        return _MultiBatchWorkerIter(self._batch_worker_pool, self._batchify_fn, dataset_iter,
                                     pin_memory=self._pin_memory, worker_fn=_batch_worker_fn,
                                     prefetch=self._batch_prefetch)

    def __del__(self):
        if self._dataset_worker_pool:
            # manually terminate due to a bug that pool is not automatically terminated
            # https://bugs.python.org/issue34172
            assert isinstance(self._dataset_worker_pool, multiprocessing.pool.Pool)
            self._dataset_worker_pool.terminate()
        if self._batch_worker_pool:
            assert isinstance(self._batch_worker_pool, multiprocessing.pool.Pool)
            self._batch_worker_pool.terminate()
        if self._manager:
            self._manager.shutdown()
