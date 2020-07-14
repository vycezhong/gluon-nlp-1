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
# pylint:disable=redefined-outer-name,logging-format-interpolation


import argparse
import collections
import json
import logging
import os
import io
import random
import time
import warnings
import itertools
import pickle
import multiprocessing as mp
from functools import partial

import numpy as np
import mxnet as mx
from mxnet import gluon

from mxnet.gluon.loss import SoftmaxCELoss

import gluonnlp as nlp
from model.fin import BertForFin
from data.preprocessing_utils import truncate_seqs_equal, concat_sequences
from data.fintext import FinTask

np.random.seed(6)
random.seed(6)
mx.random.seed(6)

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')

parser = argparse.ArgumentParser(
    description='BERT QA example.'
    'We fine-tune the BERT model on 10-Q/K dataset.')

parser.add_argument('--only_predict',
                    action='store_true',
                    help='Whether to predict only.')

parser.add_argument('--dataset_path',
                    type=str,
                    default=None,
                    help='Path to dataset')

parser.add_argument('--model_parameters',
                    type=str,
                    default=None,
                    help='Model parameter file')

parser.add_argument('--bert_model',
                    type=str,
                    default='bert_12_768_12',
                    help='BERT model name. options are bert_12_768_12 and bert_24_1024_16.')

parser.add_argument('--bert_dataset',
                    type=str,
                    default='book_corpus_wiki_en_uncased',
                    help='BERT dataset name.'
                    'options are book_corpus_wiki_en_uncased and book_corpus_wiki_en_cased.')

parser.add_argument('--pretrained_bert_parameters',
                    type=str,
                    default=None,
                    help='Pre-trained bert model parameter file. default is None')

parser.add_argument('--uncased',
                    action='store_false',
                    help='if not set, inputs are converted to lower case.')

parser.add_argument('--output_dir',
                    type=str,
                    default='./output_dir',
                    help='The output directory where the model params will be written.'
                    ' default is ./output_dir')

parser.add_argument('--epochs',
                    type=int,
                    default=3,
                    help='number of epochs, default is 3')
parser.add_argument('--training_steps',
                    type=int,
                    help='training steps, epochs will be ignored '
                    'if trainin_steps is specified.')
parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='Batch size. Number of examples per gpu in a minibatch. default is 32')

parser.add_argument('--test_batch_size',
                    type=int,
                    default=24,
                    help='Test batch size. default is 24')

parser.add_argument('--optimizer',
                    type=str,
                    default='bertadam',
                    help='optimization algorithm. default is bertadam')

parser.add_argument('--accumulate',
                    type=int,
                    default=None,
                    help='The number of batches for '
                    'gradients accumulation to simulate large batch size. Default is None')

parser.add_argument('--lr',
                    type=float,
                    default=5e-5,
                    help='Initial learning rate. default is 5e-5')

parser.add_argument('--warmup_ratio',
                    type=float,
                    default=0.1,
                    help='ratio of warmup steps that linearly increase learning rate from '
                    '0 to target learning rate. default is 0.1')

parser.add_argument('--log_interval',
                    type=int,
                    default=50,
                    help='report interval. default is 50')

parser.add_argument('--max_seq_length',
                    type=int,
                    default=512,
                    help='The maximum total input sequence length after SentencePiece tokenization.'
                    'Sequences longer than this will be truncated, and sequences shorter '
                    'than this will be padded. default is 512')

parser.add_argument(
    '--round_to', type=int, default=None,
    help='The length of padded sequences will be rounded up to be multiple of this argument.'
         'When round to is set to 8, training throughput may increase for mixed precision'
         'training on GPUs with tensorcores.')

parser.add_argument('--gpu',
                    action='store_true',
                    help='use GPU instead of CPU')

parser.add_argument('--sentencepiece',
                    type=str,
                    default=None,
                    help='Path to the sentencepiece .model file for both tokenization and vocab.')

parser.add_argument('--dtype',
                    type=str,
                    default='float32',
                    help='Data type used for training. Either float32 or float16')

parser.add_argument('--comm_backend',
                    type=str,
                    default=None,
                    help='Communication backend. Set to horovod if horovod is used for '
                         'multi-GPU training')

parser.add_argument('--model_prefix', type=str, required=False,
                    help='load static model as hybridblock.')


args = parser.parse_args()

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

fh = logging.FileHandler(os.path.join(args.output_dir, 'finetune_squad.log'),
                         mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)
log.addHandler(fh)

log.info(args)

if args.comm_backend == 'horovod':
    import horovod.mxnet as hvd
    hvd.init()
    rank = hvd.rank()
    size = hvd.size()
    local_rank = hvd.local_rank()
else:
    rank = 0
    size = 1
    local_rank = 0

if args.dtype == 'float16':
    from mxnet.contrib import amp
    amp.init()

model_name = args.bert_model
dataset_name = args.bert_dataset
only_predict = args.only_predict
model_parameters = args.model_parameters
pretrained_bert_parameters = args.pretrained_bert_parameters
if pretrained_bert_parameters and model_parameters:
    raise ValueError('Cannot provide both pre-trained BERT parameters and '
                     'BertForQA model parameters.')
lower = args.uncased

batch_size = args.batch_size
test_batch_size = args.test_batch_size
lr = args.lr
ctx = mx.gpu(local_rank) if args.gpu else mx.cpu()

accumulate = args.accumulate
log_interval = args.log_interval * accumulate if accumulate else args.log_interval
if accumulate:
    log.info('Using gradient accumulation. Effective total batch size = {}'.
             format(accumulate*batch_size*size))

optimizer = args.optimizer
warmup_ratio = args.warmup_ratio


max_seq_length = args.max_seq_length

# vocabulary and tokenizer
if args.sentencepiece:
    logging.info('loading vocab file from sentence piece model: %s', args.sentencepiece)
    if dataset_name:
        warnings.warn('Both --dataset_name and --sentencepiece are provided. '
                      'The vocabulary will be loaded based on --sentencepiece.')
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(args.sentencepiece)
    dataset_name = None
else:
    vocab = None

pretrained = not model_parameters and not pretrained_bert_parameters and not args.sentencepiece
bert, vocab = nlp.model.get_model(
    name=model_name,
    dataset_name=dataset_name,
    vocab=vocab,
    pretrained=pretrained,
    ctx=ctx,
    use_pooler=True,
    use_decoder=False,
    use_classifier=False)

if args.sentencepiece:
    tokenizer = nlp.data.BERTSPTokenizer(args.sentencepiece, vocab, lower=lower)
else:
    tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=lower)

batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Stack(),
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], round_to=args.round_to))

# load symbolic model
model_prefix = args.model_prefix

net = BertForFin(bert=bert)
if model_parameters:
    # load complete BertForQA parameters
    nlp.utils.load_parameters(net, model_parameters, ctx=ctx, cast_dtype=True)
elif pretrained_bert_parameters:
    # only load BertModel parameters
    nlp.utils.load_parameters(bert, pretrained_bert_parameters, ctx=ctx,
                              ignore_extra=True, cast_dtype=True)
    net.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
else:
    # no checkpoint is loaded
    net.initialize(init=mx.init.Normal(0.02), ctx=ctx)

net.hybridize(static_alloc=True)

loss_function = SoftmaxCELoss()
loss_function.hybridize(static_alloc=True)

task = FinTask()
metric = task.metrics


def convert_examples_to_features(example,
                                 tokenizer=None,
                                 cls_token=None,
                                 sep_token=None,
                                 vocab=None,
                                 max_seq_length=512,
                                 class_labels=None,
                                 label_alias=None):
    """convert the examples to the BERT features"""
    label_dtype = 'int32' if class_labels else 'float32'
    # get the label
    label = example[-1]
    example = example[-2]
    #create label maps if classification task
    if class_labels:
        label_map = {}
        for (i, l) in enumerate(class_labels):
            label_map[l] = i
        if label_alias:
            for key in label_alias:
                label_map[key] = label_map[label_alias[key]]
        label = label_map[label]
    label = np.array([label], dtype=label_dtype)
    # tokenize raw text
    tokens_raw = [tokenizer(l) for l in example]
    # truncate to the truncate_length,
    tokens_trun = truncate_seqs_equal(tokens_raw, max_seq_length)
    # concate the sequences with special tokens
    tokens_trun[0] = [cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = concat_sequences(tokens_trun, [[sep_token]] * len(tokens_trun))
    # convert the token to ids
    input_ids = vocab[tokens]
    valid_length = len(input_ids)
    return input_ids, segment_ids, valid_length, label


def preprocess_data(tokenizer, root, task, batch_size, dev_batch_size, max_len, vocab):
    """Train/eval Data preparation function."""
    label_dtype = 'int32' if task.class_labels else 'float32'
    max_seq_length = max_len - 2
    trans = partial(convert_examples_to_features, tokenizer=tokenizer,
                    max_seq_length=max_seq_length,
                    cls_token=vocab.cls_token,
                    sep_token=vocab.sep_token,
                    class_labels=task.class_labels,
                    label_alias=task.label_alias,
                    vocab=vocab)

    # data train
    train_tsv = task.get_dataset(root, start_year=2010, end_year=2019, cutoff=20)
    data_train = mx.gluon.data.SimpleDataset(list(map(trans, train_tsv)))
    data_train_len = data_train.transform(lambda _, segment_ids, valid_length, label: valid_length,
                                          lazy=False)
    # bucket sampler for training
    pad_val = vocab[vocab.padding_token]
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=pad_val, round_to=args.round_to),  # input
        nlp.data.batchify.Pad(axis=0, pad_val=0, round_to=args.round_to),  # segment
        nlp.data.batchify.Stack(),  # length
        nlp.data.batchify.Stack(label_dtype))  # label
    batch_sampler = nlp.data.sampler.FixedBucketSampler(data_train_len, batch_size=batch_size,
                                                        num_buckets=10, ratio=0, shuffle=True)
    # data loader for training
    loader_train = gluon.data.DataLoader(dataset=data_train, num_workers=4,
                                         batch_sampler=batch_sampler, batchify_fn=batchify_fn)

    # data dev. For MNLI, more than one dev set is available
    dev_tsv = task.get_dataset(root, start_year=2019, end_year=2020, cutoff=20)
    dev_tsv_list = dev_tsv if isinstance(dev_tsv, list) else [dev_tsv]
    loader_dev_list = []
    for segment, data in dev_tsv_list:
        data_dev = mx.gluon.data.SimpleDataset(list(map(trans, data)))
        loader_dev = mx.gluon.data.DataLoader(data_dev, batch_size=dev_batch_size, num_workers=4,
                                              shuffle=False, batchify_fn=batchify_fn)
        loader_dev_list.append((segment, loader_dev))

    return loader_train, loader_dev_list, len(data_train)


# Get the loader.
logging.info('processing dataset...')
train_dataloader, test_dataloader, num_train_examples = preprocess_data(
    tokenizer, args.dataset_path, task, batch_size, test_batch_size, args.max_len, vocab)


def train():
    """Training function."""

    log.info('Start Training')

    optimizer_params = {'learning_rate': lr}
    param_dict = net.collect_params()
    if args.comm_backend == 'horovod':
        trainer = hvd.DistributedTrainer(param_dict, optimizer, optimizer_params)
    else:
        trainer = mx.gluon.Trainer(param_dict, optimizer, optimizer_params,
                                   update_on_kvstore=False)
    if args.dtype == 'float16':
        amp.init_trainer(trainer)

    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_steps = int(num_train_examples / step_size * args.epochs)
    if args.training_steps:
        num_train_steps = args.training_steps

    num_warmup_steps = int(num_train_steps * warmup_ratio)

    def set_new_lr(step_num, batch_id):
        """set new learning rate"""
        # set grad to zero for gradient accumulation
        if accumulate:
            if batch_id % accumulate == 0:
                step_num += 1
        else:
            step_num += 1
        # learning rate schedule
        # Notice that this learning rate scheduler is adapted from traditional linear learning
        # rate scheduler where step_num >= num_warmup_steps, new_lr = 1 - step_num/num_train_steps
        if step_num < num_warmup_steps:
            new_lr = lr * step_num / num_warmup_steps
        else:
            offset = (step_num - num_warmup_steps) * lr / \
                (num_train_steps - num_warmup_steps)
            new_lr = lr - offset
        trainer.set_learning_rate(new_lr)
        return step_num

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [p for p in param_dict.values() if p.grad_req != 'null']

    # Set grad_req if gradient accumulation is required
    if accumulate:
        for p in params:
            p.grad_req = 'add'
    net.collect_params().zero_grad()

    epoch_tic = time.time()

    total_num = 0
    log_num = 0
    batch_id = 0
    step_loss = 0.0
    tic = time.time()
    step_num = 0

    tic = time.time()
    while step_num < num_train_steps:
        for _, data in enumerate(train_dataloader):
            # set new lr
            step_num = set_new_lr(step_num, batch_id)
            # forward and backward
            _, inputs, token_types, valid_length, label = data
            num_labels = len(inputs)
            log_num += num_labels
            total_num += num_labels

            with mx.autograd.record():
                out = net(inputs.as_in_context(ctx),
                          token_types.as_in_context(ctx),
                          valid_length.as_in_context(ctx).astype('float32'))

                loss = loss_function(out, [
                    label.as_in_context(ctx).astype('float32')
                ]).sum() / num_labels

                if accumulate:
                    loss = loss / accumulate
                if args.dtype == 'float16':
                    with amp.scale_loss(loss, trainer) as l:
                        mx.autograd.backward(l)
                        norm_clip = 1.0 * size * trainer._amp_loss_scaler.loss_scale
                else:
                    mx.autograd.backward(loss)
                    norm_clip = 1.0 * size

            # update
            if not accumulate or (batch_id + 1) % accumulate == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, norm_clip)
                trainer.update(1)
                if accumulate:
                    param_dict.zero_grad()

            if args.comm_backend == 'horovod':
                step_loss += hvd.allreduce(loss, average=True).asscalar()
            else:
                step_loss += loss.asscalar()

            if (batch_id + 1) % log_interval == 0:
                toc = time.time()
                log.info('Batch: {}/{}, Loss={:.4f}, lr={:.7f} '
                         'Thoughput={:.2f} samples/s'
                         .format(batch_id % len(train_dataloader),
                                 len(train_dataloader), step_loss / log_interval,
                                 trainer.learning_rate, log_num/(toc - tic)))
                tic = time.time()
                step_loss = 0.0
                log_num = 0

            if step_num >= num_train_steps:
                break
            batch_id += 1

        log.info('Finish training step: %d', step_num)
        epoch_toc = time.time()
        log.info('Time cost={:.2f} s, Thoughput={:.2f} samples/s'.format(
            epoch_toc - epoch_tic, total_num / (epoch_toc - epoch_tic)))

    if rank == 0:
        net.save_parameters(os.path.join(output_dir, 'net.params'))


def evaluate():
    """Evaluate the model on validation dataset."""
    log.info('start prediction')

    metric.reset()

    epoch_tic = time.time()
    total_num = 0
    for data in test_dataloader:
        _, inputs, token_types, valid_length, label = data
        total_num += len(inputs)
        out = net(inputs.as_in_context(ctx),
                  token_types.as_in_context(ctx),
                  valid_length.as_in_context(ctx).astype('float32'))
        metric.update([label], [out])

    metric_nm, metric_val = metric.get()

    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]
    metric_str = 'validation metrics:' + ','.join([i + ':%.4f' for i in metric_nm])
    logging.info(metric_str, *metric_val)

    mx.nd.waitall()
    epoch_toc = time.time()
    log.info('Time cost={:.2f} s, Thoughput={:.2f} samples/s'.format(
        epoch_toc - epoch_tic, total_num / (epoch_toc - epoch_tic)))


if __name__ == '__main__':
    if not only_predict:
        train()
        evaluate()
    elif model_parameters:
        evaluate()
