# Copyright 2018 The Google AI Language Team Authors and DMLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import csv
import numpy as np

from mxnet.metric import Accuracy, F1, CompositeEvalMetric
from mxnet.gluon.data import SimpleDataset
from .classification import GlueTask
import gluonnlp as nlp


csv.field_size_limit(sys.maxsize)


class CSVDataset(SimpleDataset):
    """Text dataset that reads CSV file.

    The returned dataset includes samples, each of which can either be a list of text fields
    if field_separator is specified, or otherwise a single string segment produced by the
    sample_splitter.


    Parameters
    ----------
    filename : str or list of str
        Path to the input text file or list of paths to the input text files.
    encoding : str, default 'utf8'
        File encoding format.
    delimiter : str, default ','
        Delimiter in CSV reader
    quotechar : str, default '"'
        quotechar in CSV reader
    num_discard_samples : int, default 0
        Number of samples discarded at the head of the first file.
    field_indices : list of int or None, default None
        If set, for each sample, only fields with provided indices are selected as the output.
        Otherwise all fields are returned.
    """
    def __init__(self, filename, encoding='utf8', delimiter=',', quotechar='"',
                 num_discard_samples=0, field_indices=None):

        if not isinstance(filename, (tuple, list)):
            filename = (filename, )

        self._filenames = [os.path.expanduser(f) for f in filename]
        self._delimiter = delimiter
        self._quotechar = quotechar
        self._encoding = encoding
        self._num_discard_samples = num_discard_samples
        self._field_indices = field_indices
        super(CSVDataset, self).__init__(self._read())

    def _should_discard(self):
        discard = self._num_discard_samples > 0
        self._num_discard_samples -= 1
        return discard

    def _field_selector(self, fields):
        if not self._field_indices:
            return fields
        try:
            result = [fields[i] for i in self._field_indices]
        except IndexError as e:
            raise(IndexError('%s. Fields = %s'%(str(e), str(fields))))
        return result

    def _read(self):
        all_samples = []
        for filename in self._filenames:
            with open(filename, encoding=self._encoding) as csvfile:
                content = csv.reader(csvfile,
                                     delimiter=self._delimiter,
                                     quotechar=self._quotechar)
                num_discard_samples = self._num_discard_samples
                samples = (s for s in content if not self._should_discard())
                samples = [self._field_selector(s) for s in samples]
                self._num_discard_samples = num_discard_samples
                all_samples += samples
        return all_samples


class FinText(CSVDataset):
    def __init__(self, root, start_year, end_year, **kwargs):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        quarters = np.array([str(yr) + "_" + str(qtr)
                             for yr in range(start_year, end_year)
                             for qtr in range(1, 5) if str(yr) + "_" + str(qtr) != '2019_4'])
        filename = [os.path.join(self._root, 'MDNAwLABEL_%s.csv' % quarter)
                    for quarter in quarters]
        TICKER_IDX, YEAR_IDX, QUARTER_IDX, MEAN_RETURN_IDX, X_IDX, LABEL_IDX = 1, 2, 3, 4, 9, 5
        field_indices = [TICKER_IDX, YEAR_IDX, QUARTER_IDX, MEAN_RETURN_IDX, X_IDX, LABEL_IDX]
        super(FinText, self).__init__(filename,
                                      field_indices=field_indices,
                                      num_discard_samples=1,
                                      **kwargs)


class FinTask(GlueTask):
    def __init__(self):
        is_pair = True
        #class_labels = ['0', '1', '2']
        class_labels = ['0', '1']
        metric = CompositeEvalMetric()
        metric.add(Accuracy())
        metric.add(F1(average='micro'))
        super(FinTask, self).__init__(class_labels, metric, is_pair)

    def set_label(self, dataset, start_year, end_year, cutoff=50):
        years = np.array([x[1] for x in dataset], dtype='int')
        quarters = np.array([x[2] for x in dataset], dtype='int')
        mean_returns = np.array([x[3] for x in dataset], dtype='float32')
        for y in range(start_year, end_year):
            for q in range(1, 5):
                idx1 = np.where((years == y) & (quarters == q))[0]
                if len(idx1) > 0:
                    cut = np.percentile(mean_returns[idx1], cutoff)
                    idx2 = np.where(mean_returns > cut)[0]
                    idx2 = np.array(list(set(idx1).intersection(set(idx2))))
                    #cut = np.percentile(mean_returns[idx1], 100 - cutoff)
                    #idx3 = np.where(mean_returns < cut)[0]
                    #idx3 = np.array(list(set(idx1).intersection(set(idx3))))
                    for idx in idx1:
                        dataset[idx][-1] = '0'
                    for idx in idx2:
                        dataset[idx][-1] = '1'
                    #for idx in idx3:
                    #    dataset[idx][-1] = '0'

    def get_dataset(self, root, start_year, end_year, cutoff):
        dataset = FinText(root=root, start_year=start_year, end_year=end_year)
        dataset = SimpleDataset([x for x in dataset if x[-2] != 'FAILED'])
        self.set_label(dataset, start_year, end_year, cutoff)
        return dataset



