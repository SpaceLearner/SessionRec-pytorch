import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F1

import dgl
import dgl.ops as F
import dgl.function as fn

from .sparsevd import LinearSVDO

class SRGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, batch_norm=False, feat_drop=0.0, threshold=3.0, activation=None, name=None):
        super().__init__()
        self.name       = name
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.dropout    = nn.Dropout(feat_drop)
        self.gru        = nn.GRUCell(2 * input_dim, output_dim)
        # self.W1         = nn.Linear(input_dim, output_dim, bias=False)
        # self.W2         = nn.Linear(input_dim, output_dim, bias=False)
        self.W1         = LinearSVDO(input_dim, output_dim, threshold=threshold, bias=False, name=name+'W1')
        self.W2         = LinearSVDO(input_dim, output_dim, threshold=threshold, bias=False, name=name+'W2')
        self.activation = activation
        
    def messager(self, edges):

        return {'m': edges.src['ft'] * edges.data['w'].unsqueeze(-1), 'w': edges.data['w']}

    def reducer(self, nodes):
        m = nodes.mailbox['m']
        w = nodes.mailbox['w']
        hn = m.sum(dim=1) / w.sum(dim=1).unsqueeze(-1)
        return {'neigh': hn}
    
    def forward(self, mg, feat):
        with mg.local_scope():
            if self.batch_norm is not None:
                feat = self.batch_norm(feat)
            mg.ndata['ft'] = self.dropout(feat)
            if mg.number_of_edges() > 0:
                mg.update_all(self.messager, self.reducer)
                neigh1 = mg.ndata['neigh']
                mg1 = mg.reverse(copy_edata=True)
                mg1.update_all(self.messager, self.reducer)
                neigh2 = mg1.ndata['neigh']
                neigh1 = self.W1(neigh1)
                neigh2 = self.W2(neigh2)
                hn = th.cat((neigh1, neigh2), dim=1)
                rst = self.gru(hn, feat)
            else:
                #rst = self.gru(th.cat((feat, feat), dim=1), feat)
                rst = feat
        if self.activation is not None:
            rst = self.activation(rst)
        return rst
    
class AttnReadout(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        batch_norm=True,
        feat_drop=0.0,
        threshold=3.0,
        activation=None,
        name='AttnReadout'
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop  = nn.Dropout(feat_drop)
        self.fc_u       = LinearSVDO(input_dim, hidden_dim, threshold=threshold, bias=False, name=name+'_fc_u')
        self.fc_v       = LinearSVDO(input_dim, hidden_dim, threshold=threshold, bias=True, name=name+'_fc_v')
        self.fc_e       = LinearSVDO(hidden_dim, 1, threshold=threshold, bias=False, name=name+'_fc_e')
        self.fc_out     = (nn.LinearSVDO(input_dim, output_dim, threshold=threshold, bias=False, name=name+'_fc_out')
            if output_dim != input_dim
            else None)
        self.activation = activation

    def forward(self, g, feat, last_nodes):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        feat_u = self.fc_u(feat)
        feat_v = self.fc_v(feat[last_nodes])
        feat_v = dgl.broadcast_nodes(g, feat_v)
        e = self.fc_e(th.sigmoid(feat_u + feat_v)) 
        alpha = F.segment.segment_softmax(g.batch_num_nodes(), e) 
        feat_norm = feat * alpha
        rst = F.segment.segment_reduce(g.batch_num_nodes(), feat_norm, 'sum')
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst

class SRGNN(nn.Module):
    
    def __init__(self, num_items, embedding_dim, num_layers, feat_drop=0.0, threshold=50.0, name="SRGNN"):
        super().__init__()
        self.name = name
        self.threshold = threshold
        self.embedding = nn.Embedding(num_items, embedding_dim)
        # self.indices = th.arange(num_items, dtype=th.long)
        self.register_buffer('indices', th.arange(num_items, dtype=th.long))
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        input_dim = embedding_dim
        for i in range(num_layers):
            layer = SRGNNLayer(
                input_dim,
                embedding_dim,
                batch_norm=None,
                feat_drop=feat_drop,
                threshold=threshold,
                name=name + "_SRGNNLayer" + "_" + str(i)
            )
            self.layers.append(layer)
        self.readout = AttnReadout(
            input_dim,
            embedding_dim,
            embedding_dim,
            batch_norm=None,
            feat_drop=feat_drop,
            threshold=threshold,
            activation=None,
            name=name+"_AttnReadout"
        )
        input_dim += embedding_dim
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_sr = LinearSVDO(input_dim, embedding_dim, threshold=threshold, bias=False, name=name+'_fc_sr')
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def forward(self, mg, sg=None):
        iid = mg.ndata['iid']
        feat = F1.normalize(self.feat_drop(self.embedding(iid)))
        
        out = feat
        for i, layer in enumerate(self.layers):
            out = layer(mg, out)

        feat = out

        last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        
        feat = F1.normalize(feat)        
        sr_g = self.readout(mg, feat, last_nodes)
        sr_l = feat[last_nodes]
        sr = th.cat([sr_l, sr_g], dim=1)
        sr = self.fc_sr(sr)
        target = self.embedding(self.indices)
        sr = F1.normalize(sr)
        target = F1.normalize(target)
        logits = sr @ target.t()
        logits = th.log(nn.functional.softmax(logits * 12, dim=-1))
        return logits# , 0
        
        
        