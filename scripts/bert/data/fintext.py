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
import numpy as np

from mxnet.metric import Accuracy, F1, CompositeEvalMetric
from .classification import GlueTask
import gluonnlp as nlp


class FinText(nlp.data.TSVDataset):
    def __init__(self, root, start_year, end_year, **kwargs):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        quarters = np.array([str(yr) + "_" + str(qtr)
                             for yr in range(start_year, end_year)
                             for qtr in range(1, 5)])
        filename = [os.path.join(self._root, 'MDNAwLABEL_%s.csv' % quarter)
                    for quarter in quarters]
        TICKER_IDX, YEAR_IDX, QUARTER_IDX, MEAN_RETURN_IDX, X_IDX, LABEL_IDX = 1, 2, 3, 4, 9, 5
        field_indices = [TICKER_IDX, YEAR_IDX, QUARTER_IDX, MEAN_RETURN_IDX, X_IDX, LABEL_IDX]
        field_separator = nlp.data.Splitter(',')
        super(FinText, self).__init__(filename,
                                      field_indices=field_indices,
                                      field_separator=field_separator,
                                      **kwargs)


class FinTask(GlueTask):
    def __init__(self):
        is_pair = True
        class_labels = ['0', '1']
        metric = CompositeEvalMetric()
        metric.add(Accuracy())
        metric.add(F1(average='micro'))
        super(FinTask, self).__init__(class_labels, metric, is_pair)

    def set_label(self, dataset, start_year, end_year, cutoff=50):
        for y in range(start_year, end_year):
            for q in range(1, 5):
                idx1 = np.where((dataset[1] == y) & (dataset[2] == q))[0]
                if len(idx1) > 0:
                    mean_returns = dataset[:, 3]
                    cut = np.percentile(mean_returns[idx1], cutoff)
                    idx2 = np.where(mean_returns > cut)[0]
                    idx2 = np.array(list(set(idx1).intersection(set(idx2))))
                    dataset[idx1, -1] = 0
                    dataset[idx2, -1] = 1

    def get_dataset(self, root, start_year, end_year, cutoff):
        dataset = FinText(root=root, start_year=start_year, end_year=end_year)
        dataset = dataset[1:]
        self.set_label(dataset, start_year, end_year, cutoff)
        return dataset



