import sys
import mxnet as mx
from mxnet.gluon import nn

grad_acc = bool(int(sys.argv[1]))
stop_step = int(sys.argv[2])
ctx = mx.cpu()

# data = [[mx.nd.array([0.132131, 0.0054324134, -1.05325, 2.042315, 3.342350, -0.552351, 0.00151235, 1.005235, 2.053253, -0.00004234]),
#          mx.nd.array((0,))],
#         [mx.nd.array([0.132131, 0.0054324134, -1.05325, 2.042315, 3.342350, -0.552351, 0.00151235, 1.005235, 2.053253, -0.00004234]),
#         mx.nd.array((1,))]]
        #[mx.nd.array([0.3, -0.5, 0.4, 1.3, 0.0001, 0.0031, 0.0031, -0.00132, 0.40032, -1.332320]),
        #mx.nd.array((1,))]]
data = [[mx.nd.array([[0.132131, 0.0054324134, -1.05325, 2.042315, 3.342350, -0.552351, 0.00151235, 1.005235, 2.053253, -0.00004234],
                    [0.132131, 0.0054324134, -1.05325, 2.042315, 3.342350, -0.552351, 0.00151235, 1.005235, 2.053253, -0.00004234]]),
        mx.nd.array((0, 1))]]
        # [mx.nd.array([[-0.0432424, 0.000432414, 0.43244234, -1.043243, 5.43242, 1.040324, -0.00423444, -4.324342, 1.43424,
        #                -0.00004234],
        #               [-1.432424, 0.43243424, -1.4234242342, 1.43242342, -3.432442, 2.432424, -14.424242, 0.0004234234, -0.00234322,
        #                -0.00004234]]),
        #  mx.nd.array((1, 0))]
        # ]
loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
model = nn.HybridSequential()
model.add(nn.Dense(1000))
model.add(nn.Activation('relu'))
model.add(nn.Dense(2))
model.initialize(init=mx.init.Xavier(magnitude=3.0), ctx=ctx)
model.load_parameters('save.params')
model.hybridize(static_alloc=True)
loss.hybridize(static_alloc=True)
trainer = mx.gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.1})
if grad_acc:
    model.collect_params().setattr('grad_req', 'add')
for v in model.collect_params().values():
    v.data()[:] /= 1e6
step = 0
for _ in range(1000):
    for sample, label in data:
        step += 1
        sample = sample.as_in_context(ctx)
        label = label.as_in_context(ctx)
        #sample = mx.nd.expand_dims(sample, axis=0)
        with mx.autograd.record():
            output = model(sample)
            ls = loss(output, label).sum()
        ls.backward()
        if not grad_acc:
            if step == 1:
                grads = {k: p.grad(ctx).copy() for k, p in model.collect_params().items()}
            else:
                for k, v in model.collect_params().items():
                    grads[k][:] += v.grad(ctx)
        if step == stop_step:
            if grad_acc:
                grads = {k: p.grad(ctx) for k, p in model.collect_params().items()}
                mx.nd.save('grad_acc.params', grads)
            else:
                mx.nd.save('grad.params', grads)
            sys.exit()