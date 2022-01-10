import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.ops as F
import dgl.function as fn

import torchcde

class CDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = nn.Linear(hidden_channels, 128)
        self.linear2 = nn.Linear(128, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z

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

class NISER_ODE(nn.Module):
    
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
        
        self.reduce   = nn.Linear(embedding_dim, 32)
        self.recover  = nn.Linear(32, embedding_dim)
        self.cde_func = CDEFunc(33, 32)
        self.initial  = nn.Linear(33, 32)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def forward(self, mg, embeds_ids, times, num_nodes):
        iid = mg.ndata['iid']
        
        feat = self.feat_drop(self.embedding(iid))
        if self.norm:
            feat = feat.div(th.norm(feat, p=2, dim=-1, keepdim=True) + 1e-12)
        out = feat
        for i, layer in enumerate(self.layers):
            out = layer(mg, out)
            
        feat_ode = self.reduce(self.feat_drop(self.embedding(embeds_ids)))
        if self.norm:
            feat_ode = feat_ode.div(th.norm(feat_ode, p=2, dim=-1, keepdim=True) + 1e-12)
            
        feat_ode = th.cat([times.unsqueeze(-1), feat_ode], dim=-1)
        X        = torchcde.LinearInterpolation(feat_ode)
        X0       = X.evaluate(X.interval[0])
        z0       = self.initial(X0)
        # print(X.interval)
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.cde_func,
                              t=X.interval)
        
        time_embeds = self.recover(z_T[:, 1])
        
       #  out += time_embeds
            
        last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        if self.norm:
            feat = feat.div(th.norm(feat, p=2, dim=-1, keepdim=True))
        sr_g = self.readout(mg, feat, last_nodes)
        sr_l = feat[last_nodes]
        sr = th.cat([sr_l, sr_g], dim=1)
        sr = self.fc_sr(sr) + time_embeds
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
        
        
        