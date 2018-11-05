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

import argparse
import time
import math
import os
import random
import logging
import pickle
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
import gluonnlp.data.batchify as btf
import gluonnlp as nlp
from gluonnlp.data import ShardedDataLoader
from gluonnlp.data import ExpWidthBucket, FixedBucketSampler
from mxnet.gluon.data import DataLoader, SimpleDataset
from deep_routing_model import DeepRoutingNetwork
from loss import SoftmaxCEMaskedLoss
from utils import logging_config
from graph import Graph

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

parser = argparse.ArgumentParser(description=
                                 'MXNet Deep Routing Model.')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512,
                    help='number of hidden units per layer in encoder')
parser.add_argument('--gcn_nlayers', type=int, default=3,
                    help='number of gcn layers')
parser.add_argument('--enc_nlayers', type=int, default=3,
                    help='number of encoder layers')
parser.add_argument('--nheads', type=int, default=4,
                    help='number of heads in self-attention')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1000, metavar='N',
                    help='Batch size. Number of nodes per gpu in a minibatch')
parser.add_argument('--num_buckets', type=int, default=10, help='Bucket number')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--wd', type=float, default=0,
                    help='weight decay applied to all weights')
parser.add_argument('--lr', type=float, default=1.0, help='Initial learning rate')
parser.add_argument('--warmup_steps', type=float, default=8000,
                    help='number of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument('--num_accumulated', type=int, default=1,
                    help='Number of steps to accumulate the gradients. '
                         'This is useful to mimic large batch training with limited gpu memory')
parser.add_argument('--magnitude', type=float, default=3.0,
                    help='Magnitude of Xavier initialization')
parser.add_argument('--average_checkpoint', action='store_true',
                    help='Turn on to perform final testing based on '
                         'the average of last few checkpoints')
parser.add_argument('--num_averages', type=int, default=5,
                    help='Perform final testing based on the '
                         'average of last num_averages checkpoints. '
                         'This is only used if average_checkpoint is True')
parser.add_argument('--average_start', type=int, default=5,
                    help='Perform average SGD on last average_start epochs')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--save_dir', type=str, default='routing_out',
                    help='directory path to save the final model and training log')
parser.add_argument('--eval_only', action='store_true',
                    help='Whether to only evaluate the trained model')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.'
                         '(using single gpu is suggested)')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--valid_ratio', type=float, default=0.1,
                    help='Proportion [0, 1] of training samples to use for validation set.')
args = parser.parse_args()

###############################################################################
# Load data
###############################################################################

context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
    [mx.gpu(int(x)) for x in args.gpus.split(',')]

assert args.batch_size % len(context) == 0, \
    'Total batch size must be multiple of the number of devices'

args = parser.parse_args()
logging_config(args.save_dir)
logging.info(args)

###############################################################################
# Build the model
###############################################################################

# Load the dataset


def load_routing_data():
    print('loading data ...')
    with open('data/data.pkl', 'rb') as f:
        data = pickle.load(f)
    graph = Graph(data['nodes'], data['edges'], data['weights'], normalize=True)
    trajectories = data['trajectories']
    data_train = SimpleDataset([([node[1] for node in trajectory],
                                 trajectory[-1][1]) for trajectory in trajectories])
    data_train, data_val = nlp.data.train_valid_split(data_train, args.valid_ratio)
    positions = mx.nd.array(graph.positions, ctx=context[0])
    adjacency_matrix = mx.nd.sparse.csr_matrix(graph.adjacency_matrix, dtype='float32', ctx=context[0])
    return data_train, data_val, graph, positions, adjacency_matrix


def get_data_lengths(dataset):
    return list(dataset.transform(lambda src, dest: len(src)))


class DataTransform(object):
    """Prepare the inputs
    """
    def __init__(self):
        self.pad = btf.Pad()

    def __call__(self, seq, dest):
        src = seq[:-1]
        tgt = seq[1:]
        tgt_ind = graph.get_neighbor_ind(src, tgt)
        neighbors = graph.get_neighbors(src)
        valid_target = [len(neighbor) for neighbor in neighbors]
        neighbors = self.pad(neighbors).asnumpy()
        return src, tgt_ind, neighbors, dest, len(seq), valid_target

# Construct the DataLoader. Pad data and stack label

data_train, data_val, graph, positions, adjacency_matrix = load_routing_data()
data_train_lengths = get_data_lengths(data_train)
data_val_lengths = get_data_lengths(data_val)

data_train = data_train.transform(DataTransform(), lazy=False)
data_val = data_val.transform(DataTransform(), lazy=False)

train_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(), btf.Pad(axis=(0, 1)),
                              btf.Stack(dtype='float32'), btf.Stack(dtype='float32'), btf.Pad())
val_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(), btf.Pad(axis=(0, 1)),
                            btf.Stack(dtype='float32'), btf.Stack(dtype='float32'), btf.Pad())

print('Use FixedBucketSampler')
bucket_scheme = ExpWidthBucket(bucket_len_step=1.2)
train_batch_sampler = FixedBucketSampler(lengths=data_train_lengths,
                                         batch_size=args.batch_size,
                                         num_buckets=args.num_buckets,
                                         shuffle=True,
                                         use_average_length=True,
                                         num_shards=len(context),
                                         bucket_scheme=bucket_scheme)
print(train_batch_sampler.stats())

val_batch_sampler = FixedBucketSampler(lengths=data_val_lengths,
                                       batch_size=args.batch_size * 4,
                                       num_buckets=args.num_buckets,
                                       shuffle=False,
                                       use_average_length=True,
                                       bucket_scheme=bucket_scheme)
print(val_batch_sampler.stats())

train_data_loader = ShardedDataLoader(data_train,
                                      batch_sampler=train_batch_sampler,
                                      batchify_fn=train_batchify_fn,
                                      num_workers=4)

val_data_loader = DataLoader(data_val,
                             batch_sampler=val_batch_sampler,
                             batchify_fn=val_batchify_fn,
                             num_workers=4)


# Build the model

model = DeepRoutingNetwork(graph.size, args.emsize, args.nhid,
                           args.gcn_nlayers, args.enc_nlayers,
                           graph.size, args.nheads, args.dropout)
model.initialize(init=mx.init.Xavier(magnitude=args.magnitude), ctx=context)
#model.hybridize(static_alloc=True)

loss_function = SoftmaxCEMaskedLoss()
loss_function.hybridize(static_alloc=True)

print(model)

if args.optimizer == 'sgd':
    trainer_params = {'learning_rate': args.lr,
                      'momentum': 0,
                      'wd': args.wd}
elif args.optimizer == 'adam':
    trainer_params = {'learning_rate': args.lr,
                      'wd': args.wd,
                      'beta1': 0,
                      'beta2': 0.999,
                      'epsilon': 1e-9}

trainer = gluon.Trainer(model.collect_params(), args.optimizer, trainer_params)

###############################################################################
# Training code
###############################################################################


def evaluate(data_loader, ctx=context[0]):
    """Evaluate given the data loader

    Parameters
    ----------
    data_loader : DataLoader

    Returns
    -------
    avg_loss : float
        Average loss
    real_translation_out : list of list of str
        The translation output
    """
    avg_loss_denom = 0
    avg_loss = 0.0
    embeddings = model.compute_embeddings(positions, adjacency_matrix)
    for src, tgt, neighbors, destinations, valid_length, valid_target in data_loader:
        src = src.as_in_context(ctx)
        tgt = tgt.as_in_context(ctx)
        neighbors = neighbors.as_in_context(ctx)
        destinations = destinations.as_in_context(ctx)
        valid_length = valid_length.as_in_context(ctx)
        valid_target = valid_target.as_in_context(ctx)
        # Calculating Loss
        out, _ = model(src, neighbors, destinations, valid_length - 1, embeddings)
        loss = loss_function(out, tgt, valid_length - 1, valid_target)
        loss = (loss * src.shape[1]) / (valid_length - 1)
        avg_loss += loss.sum().asscalar()
        avg_loss_denom += src.shape[0]
    avg_loss = avg_loss / avg_loss_denom
    return avg_loss


def train():
    """Training loop for deep routing model.

    """
    best_valid_loss = np.inf
    step_num = 0
    warmup_steps = args.warmup_steps
    grad_interval = args.num_accumulated
    model.collect_params().setattr('grad_req', 'add')
    average_start = (len(train_data_loader) // grad_interval) * (args.epochs - args.average_start)
    average_param_dict = None
    model.collect_params().zero_grad()
    print('start training ...')
    for epoch_id in range(args.epochs):
        log_avg_loss = 0
        log_wc = 0
        loss_denom = 0
        step_loss = 0
        log_start_time = time.time()
        for batch_id, seqs \
                in enumerate(train_data_loader):
            if batch_id % grad_interval == 0:
                step_num += 1
                new_lr = args.lr / math.sqrt(args.emsize) \
                         * min(1. / math.sqrt(step_num), step_num * warmup_steps ** (-1.5))
                trainer.set_learning_rate(new_lr)
            wc, bs = np.sum([(shard[4].sum(), shard[0].shape[0]) for shard in seqs], axis=0)
            wc = wc.asscalar()
            loss_denom += wc - bs
            seqs = [[seq.as_in_context(ctx) for seq in shard]
                    for ctx, shard in zip(context, seqs)]
            Ls = []
            with mx.autograd.record():
                embeddings = model.compute_embeddings(positions, adjacency_matrix)
                for src, tgt, neighbors, destinations, valid_length, valid_target in seqs:
                    out, _ = model(src, neighbors, destinations, valid_length - 1,
                                   embeddings.as_in_context(src.context))
                    ls = loss_function(out, tgt, valid_length - 1, valid_target).sum()
                    Ls.append((ls * src.shape[1]) / args.batch_size / 100.0)
            for L in Ls:
                L.backward()
            # ll = Ls[0].asscalar()
            # if epoch_id >= 19 and batch_id >= 60:
            #     grads = [p.grad() if p.grad_req != 'null' else None for p in model.collect_params().values()]
            #     print(grads)
            # if np.isnan(ll):
            #     save_path = os.path.join(args.save_dir, 'nan')
            #     grads = [p.grad() if p.grad_req != 'null' else None for p in model.collect_params().values()]
            #     print(grads)
            #     with open(save_path, 'wb') as f:
            #         pickle.dump(grads, f, pickle.HIGHEST_PROTOCOL)
            #     os._exit()
            if batch_id % grad_interval == grad_interval - 1 or\
                    batch_id == len(train_data_loader) - 1:
                if average_param_dict is None:
                    average_param_dict = {k: v.data(context[0]).copy() for k, v in
                                          model.collect_params().items()}
                trainer.step(float(loss_denom) / args.batch_size / 100.0)
                param_dict = model.collect_params()
                param_dict.zero_grad()
                if step_num > average_start:
                    alpha = 1. / max(1, step_num - average_start)
                    for name, average_param in average_param_dict.items():
                        average_param[:] += alpha * (param_dict[name].data(context[0]) - average_param)
            step_loss += sum([L.asscalar() for L in Ls])
            if batch_id % grad_interval == grad_interval - 1 or\
                    batch_id == len(train_data_loader) - 1:
                log_avg_loss += step_loss / loss_denom * args.batch_size * 100.0
                loss_denom = 0
                step_loss = 0
            log_wc += wc
            if (batch_id + 1) % (args.log_interval * grad_interval) == 0:
                wps = log_wc / (time.time() - log_start_time)
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, '
                             'throughput={:.2f}K wps, wc={:.2f}K'
                             .format(epoch_id, batch_id + 1, len(train_data_loader),
                                     log_avg_loss / args.log_interval,
                                     np.exp(log_avg_loss / args.log_interval),
                                     wps / 1000, log_wc / 1000))
                log_start_time = time.time()
                log_avg_loss = 0
                log_wc = 0
        mx.nd.waitall()
        valid_loss = evaluate(val_data_loader, context[0])
        logging.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}'
                     .format(epoch_id, valid_loss, np.exp(valid_loss)))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_path = os.path.join(args.save_dir, 'valid_best.params')
            logging.info('Save best parameters to {}'.format(save_path))
            model.save_parameters(save_path)
        save_path = os.path.join(args.save_dir, 'epoch{:d}.params'.format(epoch_id))
        model.save_parameters(save_path)
    save_path = os.path.join(args.save_dir, 'average.params')
    mx.nd.save(save_path, average_param_dict)


if __name__ == '__main__':
    start_pipeline_time = time.time()
    if not args.eval_only:
        train()
    if args.average_checkpoint:
        for j in range(args.num_averages):
            params = mx.nd.load(os.path.join(args.save_dir,
                                             'epoch{:d}.params'.format(args.epochs - j - 1)))
            alpha = 1. / (j + 1)
            for k, v in model._collect_params_with_prefix().items():
                for c in context:
                    v.data(c)[:] += alpha * (params[k].as_in_context(c) - v.data(c))
    elif args.average_start > 0:
        save_path = os.path.join(args.save_dir, 'average.params')
        average_param_dict = mx.nd.load(save_path)
        for k, v in model.collect_params().items():
            v.set_data(average_param_dict[k])
    else:
        model.load_parameters(os.path.join(args.save_dir, 'valid_best.params'), context)
    final_val_L = evaluate(val_data_loader, context[0])
    logging.info('Best model valid Loss={:.4f}, valid ppl={:.4f}'
                 .format(final_val_L, np.exp(final_val_L)))
    logging.info('Total time cost {:.2f}h'.format((time.time()-start_pipeline_time)/3600))
