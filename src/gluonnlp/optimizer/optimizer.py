import math
import mxnet as mx
from mxnet.ndarray import NDArray, mean, square, sqrt, clip

__all__ = ['AdaBCE', 'AdaBCM']


@mx.optimizer.Optimizer.register
class AdaBCM(mx.optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.9, tau=1.0, alpha=0.999, c=3.0, epsilon=1e-8,
                 alpha_schedule='exp', block_schedule='output', **kwargs):
        super(AdaBCM, self).__init__(learning_rate=learning_rate, **kwargs)
        self.momentum = momentum
        self.tau = tau
        self.alpha = alpha
        self.epsilon = epsilon
        self.alpha_schedule = alpha_schedule
        self.c = c
        self.block_schedule = block_schedule

    def create_state(self, index, weight):
        name = self.idx2name[index]
        if 'query_weight' not in name and 'key_weight' not in name and 'value_weight' not in name:
            if len(weight.shape) == 1:
                if self.block_schedule == 'output' or \
                                self.block_schedule == 'kernel':
                    shape = weight.shape
                else:
                    shape = (1,)
            elif len(weight.shape) == 2:
                if self.block_schedule == 'output' or \
                                self.block_schedule == 'kernel':
                    shape = (weight.shape[0], 1)
                elif self.block_schedule == 'aggregate':
                    shape = (1, weight.shape[1])
                else:
                    shape = (1, 1)
            elif len(weight.shape) == 4:
                if self.block_schedule == 'output':
                    shape = (weight.shape[0], 1, 1, 1)
                elif self.block_schedule == 'kernel':
                    shape = (weight.shape[0], weight.shape[1], 1, 1)
                elif self.block_schedule == 'aggregate':
                    shape = (1, weight.shape[1], weight.shape[2], weight.shape[3])
                else:
                    shape = (1, 1, 1, 1)
        else:
            if len(weight.shape) == 1:
                if self.block_schedule == 'output' or \
                                self.block_schedule == 'kernel':
                    shape = weight.shape
                else:
                    shape = (8, 1)
            elif len(weight.shape) == 2:
                if self.block_schedule == 'output' or \
                                self.block_schedule == 'kernel':
                    shape = (weight.shape[0], 1)
                elif self.block_schedule == 'aggregate':
                    shape = (8, 1, weight.shape[1])
                else:
                    shape = (8, 1)
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                mx.nd.zeros(shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]

        # preprocess grad
        grad = self.rescale_grad * grad + wd * weight
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        if self.alpha_schedule == 'exp':
            alpha = self.alpha
            scale = 1 - self.alpha**t
        elif self.alpha_schedule == 'poly':
            alpha = 1 - (self.c + 1)/(t+self.c)
            scale = 1
        lr *= math.sqrt(scale)
        # update m_t and u_t
        m_t, u_t = state
        l = len(u_t.shape)
        # pre update weight
        weight[:] -= (self.tau * self.momentum) * m_t
        name = self.idx2name[index]
        if 'query_weight' not in name and 'key_weight' not in name and 'value_weight' not in name:
            if self.block_schedule == 'output':
                axes = 0
                exclude = True
            elif self.block_schedule == 'kernel':
                axes = tuple(range(l - int(l / 2)))
                exclude = True
            elif self.block_schedule == 'aggregate':
                axes = 0
                exclude = False
            else:
                axes = ()
                exclude = True
            u_t[:] = alpha * u_t + (1 - alpha) * mean(square(grad), axes,
                                                      exclude=exclude, keepdims=True)
            m_t[:] = self.momentum * m_t - lr * grad / (sqrt(u_t) + self.epsilon)
        else:
            if self.block_schedule == 'output' or \
                            self.block_schedule == 'kernel':
                u_t[:] = alpha * u_t + (1 - alpha) * mean(square(grad), 0,
                                                          exclude=True, keepdims=True)
                m_t[:] = self.momentum * m_t - lr * grad / (sqrt(u_t) + self.epsilon)
            elif self.block_schedule == 'aggregate':
                if l == 2:
                    shape = (8, -1)
                elif l == 3:
                    shape = (-4, 8, -1, 0)
                u_t[:] = alpha * u_t + (1 - alpha) * mean(square(grad).reshape(shape), 1,
                                                          keepdims=True)
                delta = lr * grad.reshape(shape) / (sqrt(u_t) + self.epsilon)
                m_t[:] = self.momentum * m_t - delta.reshape(grad.shape)
            else:
                shape = (8, -1)
                u_t[:] = alpha * u_t + (1 - alpha) * mean(square(grad).reshape(shape), 0,
                                                          exclude=True, keepdims=True)
                delta = lr * grad.reshape(shape) / (sqrt(u_t) + self.epsilon)
                m_t[:] = self.momentum * m_t - delta.reshape(grad.shape)
        # update weight
        weight[:] += (1 + self.tau * self.momentum) * m_t


@mx.optimizer.Optimizer.register
class AdaBCE(mx.optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, alpha=0.999, c=3.0, epsilon=1e-8,
                 alpha_schedule='exp', block_schedule='output', **kwargs):
        super(AdaBCE, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta = beta
        self.alpha = alpha
        self.alpha_schedule = alpha_schedule
        self.c = c
        self.epsilon = epsilon
        self.block_schedule = block_schedule

    def create_state(self, index, weight):
        name = self.idx2name[index]
        if 'query_weight' not in name and 'key_weight' not in name and 'value_weight' not in name:
            if len(weight.shape) == 1:
                if self.block_schedule == 'output' or \
                                self.block_schedule == 'kernel':
                    shape = weight.shape
                elif self.block_schedule == 'coordinate':
                    shape = weight.shape
                else:
                    shape = (1,)
            elif len(weight.shape) == 2:
                if self.block_schedule == 'output' or \
                                self.block_schedule == 'kernel':
                    shape = (weight.shape[0], 1)
                elif self.block_schedule == 'aggregate':
                    shape = (1, weight.shape[1])
                elif self.block_schedule == 'coordinate':
                    shape = weight.shape
                else:
                    shape = (1, 1)
            elif len(weight.shape) == 4:
                if self.block_schedule == 'output':
                    shape = (weight.shape[0], 1, 1, 1)
                elif self.block_schedule == 'kernel':
                    shape = (weight.shape[0], weight.shape[1], 1, 1)
                elif self.block_schedule == 'aggregate':
                    shape = (1, weight.shape[1], weight.shape[2], weight.shape[3])
                else:
                    shape = (1, 1, 1, 1)
        else:
            if len(weight.shape) == 1:
                if self.block_schedule == 'output' or \
                                self.block_schedule == 'kernel':
                    shape = weight.shape
                elif self.block_schedule == 'coordinate':
                    shape = weight.shape
                else:
                    shape = (8, 1)
            elif len(weight.shape) == 2:
                if self.block_schedule == 'output' or \
                                self.block_schedule == 'kernel':
                    shape = (weight.shape[0], 1)
                elif self.block_schedule == 'aggregate':
                    shape = (8, 1, weight.shape[1])
                elif self.block_schedule == 'coordinate':
                    shape = weight.shape
                else:
                    shape = (8, 1)
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                mx.nd.zeros(shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]

        coef1 = (1. - self.beta**t)

        # preprocess grad
        grad = self.rescale_grad * grad + wd * weight
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        if self.alpha_schedule == 'exp':
            alpha = self.alpha
            scale = 1. - self.alpha**t
        elif self.alpha_schedule == 'poly':
            alpha = 1. - (self.c + 1)/(t+self.c)
            scale = 1.
        lr *= math.sqrt(scale)/coef1
        # update m_t and u_t
        m_t, u_t = state
        l = len(u_t.shape)
        m_t[:] = self.beta * m_t + (1. - self.beta) * grad
        name = self.idx2name[index]
        if 'query_weight' not in name and 'key_weight' not in name and 'value_weight' not in name:
            if self.block_schedule == 'output':
                axes = 0
                exclude = True
            elif self.block_schedule == 'kernel':
                axes = tuple(range(l - int(l / 2)))
                exclude = True
            elif self.block_schedule == 'aggregate':
                axes = 0
                exclude = False
            elif self.block_schedule == 'coordinate':
                axes = tuple(range(l))
                exclude = True
            else:
                axes = ()
                exclude = True
            u_t[:] = alpha * u_t + (1. - alpha) * mean(square(grad), axes,
                                                       exclude=exclude, keepdims=True)
            delta = lr * m_t / (sqrt(u_t) + self.epsilon)
        else:
            if self.block_schedule == 'output' or \
                            self.block_schedule == 'kernel':
                u_t[:] = alpha * u_t + (1. - alpha) * mean(square(grad), 0,
                                                           exclude=True, keepdims=True)
                delta = lr * m_t / (sqrt(u_t) + self.epsilon)
            elif self.block_schedule == 'aggregate':
                if l == 2:
                    shape = (8, -1)
                elif l == 3:
                    shape = (-4, 8, -1, 0)
                u_t[:] = alpha * u_t + (1. - alpha) * mean(square(grad).reshape(shape), 1,
                                                           keepdims=True)
                delta = lr * m_t.reshape(shape) / (sqrt(u_t) + self.epsilon)
                delta = delta.reshape(grad.shape)
            elif self.block_schedule == 'coordinate':
                u_t[:] = alpha * u_t + (1. - alpha) * square(grad)
                delta = lr * m_t / (sqrt(u_t) + self.epsilon)
            else:
                shape = (8, -1)
                u_t[:] = alpha * u_t + (1. - alpha) * mean(square(grad).reshape(shape), 0,
                                                           exclude=True, keepdims=True)
                delta = lr * m_t.reshape(shape) / (sqrt(u_t) + self.epsilon)
                delta = delta.reshape(grad.shape)
        # update weight
        weight[:] -= delta

