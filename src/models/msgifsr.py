import math

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import dgl.ops as F
import numpy as np
import torch as th
import torch.nn as nn
from scipy import sparse

from .gnn_models import GATConv

class SemanticExpander(nn.Module):
    
    def __init__(self, input_dim, reducer, order):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.order = order
        self.reducer = reducer
        self.GRUs = nn.ModuleList()
        for i in range(self.order):
            self.GRUs.append(nn.GRU(self.input_dim, self.input_dim, 1, True, True))
    
        if self.reducer == 'concat':
            self.Ws = nn.ModuleList()
            for i in range(1, self.order):
                self.Ws.append(nn.Linear(self.input_dim * (i+1), self.input_dim))
        
    def forward(self, feat):
        
        if len(feat.shape) < 3:
            return feat
        if self.reducer == 'mean':
            invar = th.mean(feat, dim=1)
        elif self.reducer == 'max':
            invar =  th.max(feat, dim=1)[0]
        elif self.reducer == 'concat':
            invar =  self.Ws[feat.size(1)-2](feat.view(feat.size(0), -1))
        var = self.GRUs[feat.size(1)-2](feat)[1].permute(1, 0, 2).squeeze()

        # return invar + var
        return 0.5 * invar + 0.5 * var
      
class MSHGNN(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout=0.0, activation=None, order=1, reducer='mean'):
        super().__init__()
     
        self.dropout = nn.Dropout(dropout)
        # self.gru = nn.GRUCell(2 * input_dim, output_dim)
        self.output_dim = output_dim
        self.activation = activation
        self.order = order
        
        conv1_modules = {'intra'+str(i+1) : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True) for i in range(self.order)}
        conv1_modules.update({'inter'     : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True)})
        self.conv1 = dglnn.HeteroGraphConv(conv1_modules, aggregate='sum')
        
        conv2_modules = {'intra'+str(i+1) : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True) for i in range(self.order)}
        conv2_modules.update({'inter'     : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True)})
        self.conv2 = dglnn.HeteroGraphConv(conv2_modules, aggregate='sum')
        
        self.lint = nn.Linear(output_dim, 1, bias=False)
        self.linq = nn.Linear(output_dim, output_dim)
        self.link = nn.Linear(output_dim, output_dim, bias=False)
        
    def forward(self, g, feat):
        
        with g.local_scope():
                
            h1 = self.conv1(g, (feat, feat))
            h2 = self.conv2(g.reverse(copy_edata=True), (feat, feat))
            h = {}
            for i in range(self.order):
                hl, hr = th.zeros(1, self.output_dim).to(self.lint.weight.device), th.zeros(1, self.output_dim).to(self.lint.weight.device)
                if 's'+str(i+1) in h1:
                    hl = h1['s'+str(i+1)]
                if 's'+str(i+1) in h2:
                    hr = h2['s'+str(i+1)]
                h['s'+str(i+1)] = hl + hr
                if len(h['s'+str(i+1)].shape) > 2:
                    h['s'+str(i+1)] = h['s'+str(i+1)].max(1)[0]
                h_mean = F.segment.segment_reduce(g.batch_num_nodes('s'+str(i+1)), feat['s'+str(i+1)], 'mean')
                h_mean = dgl.broadcast_nodes(g, h_mean, ntype='s'+str(i+1)) # adding mean maskes better
                # print(h['s'+str(i+1)].shape, h_mean.shape)
                h['s'+str(i+1)] =  h_mean + h['s'+str(i+1)]
                
        return h
                
    
class AttnReadout(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        feat_drop=0.0,
        activation=None,
        order=1,
        device=th.device('cpu')
    ):
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop)
        self.order = order
        self.device = device
        self.fc_u = nn.ModuleList()
        self.fc_v = nn.ModuleList()
        self.fc_e = nn.ModuleList()
        self.fc_p = nn.ModuleList()
        for i in range(self.order):
            self.fc_u.append(nn.Linear(input_dim, hidden_dim, bias=True))
            self.fc_v.append(nn.Linear(input_dim, hidden_dim, bias=False))
            self.fc_e.append(nn.Linear(hidden_dim, 1, bias=False))
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim
            else None
        )
        self.activation = activation
        
    def forward(self, g, feats, last_nodess):
        
        rsts = []
      
        nfeats = []
        for i in range(self.order): 
            feat = feats['s'+str(i+1)]
            feat = th.split(feat, g.batch_num_nodes('s'+str(i+1)).tolist())
            feats['s'+str(i+1)] = th.cat(feat, dim=0)
            nfeats.append(feat)
        feat_vs= th.cat(tuple(feats['s'+str(i+1)][last_nodess[i]].unsqueeze(1) for i in range(self.order)), dim=1)
        feats = th.cat([th.cat(tuple(nfeats[j][i] for j in range(self.order)), dim=0) for i in range(len(g.batch_num_nodes('s1')))], dim=0)
        batch_num_nodes = th.cat(tuple(g.batch_num_nodes('s'+str(i+1)).unsqueeze(1) for i in range(self.order)), dim=1).sum(1)
       
        idx = th.cat(tuple(th.ones(batch_num_nodes[j])*j for j in range(len(batch_num_nodes)))).long()
        for i in range(self.order):
            feat_u = self.fc_u[i](feats) 
            feat_v = self.fc_v[i](feat_vs[:, i])[idx]
            e = self.fc_e[i](th.sigmoid(feat_u + feat_v))
            alpha = F.segment.segment_softmax(batch_num_nodes, e)
            
            feat_norm = feats * alpha
            rst = F.segment.segment_reduce(batch_num_nodes, feat_norm, 'sum')
            rsts.append(rst.unsqueeze(1))
        
            if self.fc_out is not None:
                rst = self.fc_out(rst)
            if self.activation is not None:
                rst = self.activation(rst)
        rst = th.cat(rsts, dim=1)
        
        return rst

class MSGIFSR(nn.Module):
    
    def __init__(self, num_items, datasets, embedding_dim, num_layers, dropout=0.0, reducer='mean', order=3, norm=True, extra=True, fusion=True, device=th.device('cpu')):
        super().__init__()
        
        self.embeddings = nn.Embedding(num_items, embedding_dim, max_norm=1)
 
        self.num_items = num_items
        self.register_buffer('indices', th.arange(num_items, dtype=th.long))
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.layers   = nn.ModuleList()
        input_dim     = embedding_dim
        self.reducer  = reducer
        self.order    = order
        self.alpha    = nn.Parameter(th.Tensor(self.order))
        self.beta     = nn.Parameter(th.Tensor(1))
        self.norm     = norm
        self.expander = SemanticExpander(input_dim, reducer, order)
        
        self.device = device
        for i in range(num_layers):
            layer = MSHGNN(
                input_dim,
                embedding_dim,
                dropout=dropout,
                order=self.order,
                activation=nn.PReLU(embedding_dim)
            )
            self.layers.append(layer)
            
        self.readout = AttnReadout(
            input_dim,
            embedding_dim,
            embedding_dim,
            feat_drop=dropout,
            activation=None,
            order=self.order,
            device=self.device
        )
        input_dim += embedding_dim
        self.feat_drop = nn.Dropout(dropout)

        self.fc_sr = nn.ModuleList()
        for i in range(self.order):
            self.fc_sr.append(nn.Linear(input_dim, embedding_dim, bias=False))
        
        self.sc_sr = nn.ModuleList()
        for i in range(self.order):
            self.sc_sr.append(nn.Sequential(nn.Linear(embedding_dim, embedding_dim, bias=True),  nn.ReLU(), nn.Linear(embedding_dim, 2, bias=False), nn.Softmax(dim=-1)))
        self.input_dim = input_dim
        self.embedding_dim =embedding_dim
 
        # self.sr_trans1 = nn.Linear(embedding_dim, embedding_dim)
        # self.sr_trans2 = nn.Linear(embedding_dim, embedding_dim)
        self.reset_parameters()
        self.alpha.data = th.zeros(self.order)
        self.alpha.data[0] = th.tensor(1.0)
        # self.beta.data = th.zeros(1)
        self.beta.data = th.tensor(1.0)
        self.fusion = fusion
        self.extra = extra
        self.epoch = 0
        
    def inc_epoch(self):
        self.epoch += 1
          
    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def iid2rp(self, iid):
        tmp = th.sum(th.cat(tuple(th.unique(tmp, return_inverse=True)[1].unsqueeze(0) for tmp in th.unbind(iid, dim=0)), dim=0), dim=1)
        
        return tmp
        
    def residual(self, h1, res):
        
        for key in h1.keys():
            h1[key] += res[key]
        
        return h1
        
    def forward(self, mg):
        
        
        feats = {}
        for i in range(self.order):
            iid = mg.nodes['s' + str(i+1)].data['iid']
            feat = self.embeddings(iid) 
            feat = self.feat_drop(feat)
            feat = self.expander(feat)
            if th.isnan(feat).any():
                feat = feat.masked_fill(feat != feat, 0)
            if self.norm:
                feat = nn.functional.normalize(feat, dim=-1)
            feats['s' + str(i+1)] = feat
       
        h = feats
        for idx, layer in enumerate(self.layers):
            h = layer(mg, h)

        last_nodes = []
        for i in range(self.order):
            if self.norm:
                h['s'+str(i+1)] = nn.functional.normalize(h['s'+str(i+1)], dim=-1)
            last_nodes.append(mg.filter_nodes(lambda nodes: nodes.data['last'] == 1, ntype='s'+str(i+1)))
            
        feat = h
        sr_g = self.readout(mg, feat, last_nodes)                                                               

        sr_l = th.cat([feat['s'+str(i+1)][last_nodes[i]].unsqueeze(1) for i in range(self.order)], dim=1)
        sr   = th.cat([sr_l, sr_g], dim=-1)# .view(sr_l.size(0), -1)
        sr   = th.cat([self.fc_sr[i](sr).unsqueeze(1) for i, sr in enumerate(th.unbind(sr, dim=1))], dim=1)
        if self.norm:
            sr = nn.functional.normalize(sr, dim=-1)
        
        
        target = self.embeddings(self.indices)
        
        if self.norm:
            target = nn.functional.normalize(target, dim=-1)
               
        if self.extra:
            logits = sr @ target.t()
            phi = self.sc_sr[0](sr).unsqueeze(-1)
            mask = th.zeros(phi.size(0), self.num_items).to(self.device)
            iids = th.split(mg.nodes['s1'].data['iid'], mg.batch_num_nodes('s1').tolist())
            for i in range(len(mask)):
                mask[i, iids[i]] = 1

            logits_in = logits.masked_fill(~mask.bool().unsqueeze(1), float('-inf'))
            logits_ex = logits.masked_fill(mask.bool().unsqueeze(1), float('-inf'))
            score     = th.softmax(12 * logits_in.squeeze(), dim=-1)
            score_ex  = th.softmax(12 * logits_ex.squeeze(), dim=-1) 
          
            if th.isnan(score).any():
                score    = feat.masked_fill(score != score, 0)
            if th.isnan(score_ex).any():
                score_ex = score_ex.masked_fill(score_ex != score_ex, 0)
            assert not th.isnan(score).any()
            assert not th.isnan(score_ex).any()
            # print(score.shape, score_ex.shape)
            if self.order == 1:
                phi = phi.squeeze(1)
                score = (th.cat((score.unsqueeze(1), score_ex.unsqueeze(1)), dim=1) * phi).sum(1)
            else:
                score = (th.cat((score.unsqueeze(2), score_ex.unsqueeze(2)), dim=2) * phi).sum(2)
        else:
            # print("no extra ****************")
            logits = sr.squeeze() @ target.t()
            score  = th.softmax(12 * logits, dim=-1)
        
        if self.order > 1 and self.fusion:
            alpha = th.softmax(self.alpha.unsqueeze(0), dim=-1).view(1, self.alpha.size(0), 1)
            g = th.ones(score.size(0), score.size(1), 1).to(self.device)
            g = alpha.repeat(score.size(0), 1, 1)
            score = (score * g).sum(1)
        elif self.order > 1:
            score = score[:, 0]
            
        # print(score.shape)
            
        score = th.log(score)
        
        return score
        
        
