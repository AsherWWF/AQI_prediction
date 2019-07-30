from mxnet import nd
from mxnet.gluon import nn, Block

class MLP(nn.HybridSequential):
    def __init__(self, hiddens, act_type, out_act, weight_initializer=None, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            for i, h in enumerate(hiddens):
                activation = None if i == len(hiddens) - 1 else act_type
                self.add(nn.Dense(h, activation=activation, weight_initializer=weight_initializer))

class HyperDense(Block):
    def __init__(self, pre_hidden_size, hidden_size, hyper_hiddens, prefix=None):
        super(HyperDense, self).__init__(prefix=prefix)
        self.pre_hidden_size = pre_hidden_size
        self.hidden_size = hidden_size
        self.act_type = 'sigmoid'
        
        with self.name_scope():
            self.w_mlp = MLP(hyper_hiddens + [self.pre_hidden_size * self.hidden_size,], act_type=self.act_type, out_act=False, prefix='w_')
            self.b_mlp = MLP(hyper_hiddens + [1,], act_type=self.act_type, out_act=False, prefix='b_')

    def forward(self, feature, data):
        """ Forward process of a HyperDense layer

        Args:
            feature: a NDArray with shape [n, d]
            data: a NDArray with shape [n, b, pre_d]

        Returns:
            output: a NDArray with shape [n, b, d]
        """
        weight = self.w_mlp(feature) # [n, pre_hidden_size * hidden_size]
        weight = nd.reshape(weight, (-1, self.pre_hidden_size, self.hidden_size))
        bias = nd.reshape(self.b_mlp(feature), shape=(-1, 1, 1)) # [n, 1, 1]
        return nd.batch_dot(data, weight) + bias