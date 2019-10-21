import mxnet as mx

grad = mx.nd.load('grad.params')
grad_acc = mx.nd.load('grad_acc.params')
for k, v in grad.items():
    v2 = grad_acc[k]
    diff = v - v2
    print(v.max())
    print(k, 'rtol:{}%, atol:{}'.format(
        mx.nd.max(mx.nd.abs(diff)/mx.nd.abs(v)).asscalar()*100,
        mx.nd.max(mx.nd.abs(diff)).asscalar()))