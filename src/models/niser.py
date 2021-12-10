import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.ops as F
import dgl.function as fn

class SRGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, batch_norm=False, feat_drop=0.0, activation=None):
        super().__init__()
        self.dropout    = nn.Dropout(feat_drop)
        self.gru        = nn.GRUCell(2 * input_dim, output_dim)
        self.W1         = nn.Linear(input_dim, output_dim, bias=False)
        self.W2         = nn.Linear(input_dim, output_dim, bias=False)
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
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim
            else None
        )
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

class NISER(nn.Module):
    
    def __init__(self, num_items, embedding_dim, num_layers, feat_drop=0.0, norm=True, scale=12):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim)
        self.register_buffer('indices', th.arange(num_items, dtype=th.long))
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.norm = norm
        self.scale = scale
        input_dim = embedding_dim
        for i in range(num_layers):
            layer = SRGNNLayer(
                input_dim,
                embedding_dim,
                batch_norm=None,
                feat_drop=feat_drop
            )
            self.layers.append(layer)
        self.readout = AttnReadout(
            input_dim,
            embedding_dim,
            embedding_dim,
            batch_norm=None,
            feat_drop=feat_drop,
            activation=None,
        )

        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_sr = nn.Linear(input_dim + embedding_dim, embedding_dim, bias=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def forward(self, mg, sg=None):
        iid = mg.ndata['iid']
        feat = self.feat_drop(self.embedding(iid))
        if self.norm:
            feat = feat.div(th.norm(feat, p=2, dim=-1, keepdim=True) + 1e-12)
        out = feat
        for i, layer in enumerate(self.layers):
            out = layer(mg, out)
            
        last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        if self.norm:
            feat = feat.div(th.norm(feat, p=2, dim=-1, keepdim=True))
        sr_g = self.readout(mg, feat, last_nodes)
        sr_l = feat[last_nodes]
        sr = th.cat([sr_l, sr_g], dim=1)
        sr = self.fc_sr(sr)
        if self.norm:
            sr = sr.div(th.norm(sr, p=2, dim=-1, keepdim=True) + 1e-12)
        target = self.embedding(self.indices)
        if self.norm:
            target = target.div(th.norm(target, p=2, dim=-1, keepdim=True) + 1e-12)
        logits = sr @ target.t()
        if self.scale:
            logits = th.log(nn.functional.softmax(self.scale * logits, dim=-1))
        else:
            logits = th.log(nn.functional.softmax(logits, dim=-1))
        return logits# , 0
        
        
        