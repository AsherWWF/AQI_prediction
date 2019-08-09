import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, Block

import dgl
from dgl import DGLGraph
from functools import partial

class Graph(Block):
    @staticmethod
    def create(graph_type, dist, src, dst, hidden_size, prefix):
        """ create a graph. """
        if graph_type == 'None': return None
        elif graph_type == 'AttGraph': return AttGraph(dist, src, dst, hidden_size, prefix=prefix)
        elif graph_type == 'HyperAttGraph': return HyperAttGraph(dist, src, dst, hidden_size, prefix=prefix)
        else: raise Exception('Unknow graph: %s' % graph_type)

    @staticmethod
    def create_graphs(graph_type, graph, hidden_size, prefix):
        """ Create a list of graphs according to graph_type & graph. """
        if graph_type == 'None': 
            return None
        elif graph_type == 'HighLevelGraph':
            src_pool, dst_pool = graph['pool']
            dist_agg, src_agg, dst_agg = graph['agg']
            src_updata, dst_update = graph['update']
            dist_low, src_low, dst_low = graph['low']
            return [
                HighLevelGraph(src_pool, dst_pool, 
                 src_agg, dst_agg, dist_agg,
                 src_updata, dst_update,
                 src_low, dst_low, dist_low,
                 hidden_size),
            ]
        else:         
            dist, src, dst = graph['low']
            return [
                Graph.create(graph_type, dist, src, dst, hidden_size, prefix + 'graph_'),
            ]

    def __init__(self, dist, src, dst, hidden_size, prefix=None):
        super(Graph, self).__init__(prefix=prefix)
        self.dist = dist
        self.src = src
        self.dst = dst
        self.hidden_size = hidden_size
        self.dist = mx.nd.expand_dims(mx.nd.array(dist[src, dst]), axis=1)  #[num_edge, 1]

        # create graph
        self.num_nodes = n = dist.shape[0]

        self.ctx = []
        self.graph_on_ctx = []

        self.init_model()    

    def build_graph_on_ctx(self, ctx):
        g = DGLGraph()
        g.set_n_initializer(dgl.init.zero_initializer)
        g.add_nodes(self.num_nodes)
        g.add_edges(self.src, self.dst)
        g.edata['dist'] = self.dist.as_in_context(ctx)
        self.graph_on_ctx.append(g)
        self.ctx.append(ctx)
    
    def get_graph_on_ctx(self, ctx):
        if ctx not in self.ctx:
            self.build_graph_on_ctx(ctx)
        return self.graph_on_ctx[self.ctx.index(ctx)]

    def forward(self, state, feature=None): # first dimension of state & feature should be num_nodes
        g = self.get_graph_on_ctx(state.context)
        g.ndata['state'] = state
        # g.ndata['feature'] = feature        
        g.update_all(self.msg_edge, self.msg_reduce)
        state = g.ndata.pop('new_state')
        return state

    def init_model(self):
        raise NotImplementedError("To be implemented")

    def msg_edge(self, edge):
        """ Messege passing across edge

        Args:
            edge: a dictionary of edge data.

                edge.src['state'] and edge.dst['state']: hidden states of the nodes, which is NDArrays with shape [e, b, t, d] or [e, b, d]
                edge.src['feature'] and  edge.dst['state']: geo features of the nodes, which is NDArrays with shape [e, d]
                edge.data['dist']: distance matrix of the edges, which is a NDArray with shape [e, 1]

        Returns:
            A dictionray of messages
        """

        raise NotImplementedError("To be implemented")

    def msg_reduce(self, node):
        raise NotImplementedError("To be implemented")
        
class AttGraph(Graph):
    def __init__(self, dist, src, dst, hidden_size, prefix=None):
        super(AttGraph, self).__init__(dist, src, dst, hidden_size, prefix)

    def init_model(self):
        self.weight = self.params.get('weight', shape=(self.hidden_size * 2, self.hidden_size))
    
    def msg_edge(self, edge):
        state = nd.concat(edge.src['state'], edge.dst['state'], dim=-1)
        ctx = state.context

        alpha = nd.LeakyReLU(nd.dot(state, self.weight.data(ctx)))

        dist = edge.data['dist']
        while len(dist.shape) < len(alpha.shape):
            dist = nd.expand_dims(dist, axis=-1)

        alpha = alpha * dist 
        return { 'alpha': alpha, 'state': edge.src['state'] }

    def msg_reduce(self, node):
        state = node.mailbox['state']
        alpha = node.mailbox['alpha']
        alpha = nd.softmax(alpha, axis=1)

        new_state = nd.relu(nd.sum(alpha * state, axis=1))
        return { 'new_state': new_state }

class HyperAttGraph(Graph):
    def __init__(self, dist, src, dst, hidden_size, prefix=None):
        super(HyperAttGraph, self).__init__(dist, src, dst, hidden_size, prefix)

    def init_model(self):
        # self.weight = self.params.get('weight', shape=(self.hidden_size * 2, self.hidden_size))
        from model.basic_structure import MLP
        with self.name_scope():
            self.w_mlp = MLP([16, 2, self.hidden_size * self.hidden_size * 2], 'sigmoid', False)
    
    def msg_edge(self, edge):
        state = nd.concat(edge.src['state'], edge.dst['state'], dim=-1)
        feature = nd.concat(edge.src['feature'], edge.dst['feature'], edge.data['dist'], dim=-1)

        # generate weight by hypernetwork
        weight = self.w_mlp(feature)
        weight = nd.reshape(weight, shape=(-1, self.hidden_size * 2, self.hidden_size))

        # reshape state to [n, b * t, d] for batch_dot (currently mxnet only support batch_dot for 3D tensor)
        shape = state.shape
        state = nd.reshape(state, shape=(shape[0], -1, shape[-1]))

        alpha = nd.LeakyReLU(nd.batch_dot(state, weight))

        # reshape alpha to [n, b, t, d]
        alpha = nd.reshape(alpha, shape=shape[:-1] + (self.hidden_size,))
        return { 'alpha': alpha, 'state': edge.src['state'] }

    def msg_reduce(self, node):
        state = node.mailbox['state']
        alpha = node.mailbox['alpha']
        alpha = nd.softmax(alpha, axis=1)

        new_state = nd.relu(nd.sum(alpha * state, axis=1))
        return { 'new_state': new_state }

class PoolingGraph(Block):
    def __init__(self, src, dst, prefix=None):
        super(Graph, self).__init__(prefix=prefix)
        self.src = src
        self.dst = dst

        # create graph
        self.num_nodes = len(src) + len(set(dst))

        self.ctx = []
        self.graph_on_ctx = []

    def build_graph_on_ctx(self, ctx):
        g = DGLGraph()
        g.set_n_initializer(dgl.init.zero_initializer)
        g.add_nodes(self.num_nodes)
        g.add_edges(self.src, self.dst)
    
        self.graph_on_ctx.append(g)
        self.ctx.append(ctx)
    
    def get_graph_on_ctx(self, ctx):
        if ctx not in self.ctx:
            self.build_graph_on_ctx(ctx)
        return self.graph_on_ctx[self.ctx.index(ctx)]

    def forward(self, state): # first dimension of state & feature should be num_nodes
        g = self.get_graph_on_ctx(state.context)
        g.nodes[self.src].data['state'] = state        
        g.update_all(self.msg_edge, self.msg_reduce)
        state = g.ndata.pop('new_state')
        return state[len(src): ]
    
    def msg_edge(self, edge):
        return {'state': edge.src['state'] }

    def msg_reduce(self, node):
        state = node.mailbox['state']
        new_state = nd.mean(state, axis=1)
        return { 'new_state': new_state }

class updateGraph(Block):
    def __init__(self, src, dst, prefix=None):
        super(Graph, self).__init__(prefix=prefix)
        self.src = src
        self.dst = dst

        # create graph
        self.num_nodes = len(set(src)) + len(dst)

        self.ctx = []
        self.graph_on_ctx = []

    def build_graph_on_ctx(self, ctx):
        g = DGLGraph()
        g.set_n_initializer(dgl.init.zero_initializer)
        g.add_nodes(self.num_nodes)
        g.add_edges(self.src, self.dst)
    
        self.graph_on_ctx.append(g)
        self.ctx.append(ctx)
    
    def get_graph_on_ctx(self, ctx):
        if ctx not in self.ctx:
            self.build_graph_on_ctx(ctx)
        return self.graph_on_ctx[self.ctx.index(ctx)]

    def forward(self, state): # first dimension of state & feature should be num_nodes
        g = self.get_graph_on_ctx(state.context)
        g.nodes[list(set(self.src))].data['state'] = state       
        g.update_all(self.msg_edge, self.msg_reduce)
        state = g.ndata.pop('new_state')
        return state[: len(dst)]
    
    def msg_edge(self, edge):
        return {'state': edge.src['state'] }

    def msg_reduce(self, node):
        state = node.mailbox['state']
        new_state = nd.mean(state, axis=1)
        return { 'new_state': new_state }

class HighLevelGraph(Block):
    def __init__(self, 
                 src_pool, dst_pool, 
                 src_agg, dst_agg, dist_agg,
                 src_updata, dst_update,
                 src_low, dst_low, dist_low,
                 hidden_size):
        super(HighLevelGraph, self).__init__()
        self.pool = PoolingGraph(src_pool, dst_pool)
        self.agg = AttGraph(dist_agg, src_agg, dst_agg, hidden_size)
        self.update = updateGraph(src_updata, dst_update)
        self.low = AttGraph(dist_low, src_low, dst_low, 2 * hidden_size)

    def forward(self, state, feature):
        _state = state
        state = self.pool(state)
        state = self.agg(state)
        state = self.update(state)
        state = nd.concat(_state, state, dim=-1)
        state = self.low(state)
        return state
