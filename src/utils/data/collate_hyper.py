from collections import Counter
import numpy as np
import torch as th
import dgl
import pickle
import numba
from numba import jit


def label_last(g, last_nid):
    is_last = th.zeros(g.num_nodes(), dtype=th.int32)
    is_last[last_nid] = 1
    g.ndata['last'] = is_last
    return g

def label_last_ccs(g, last_nid):
    for i in range(len(last_nid)):
        is_last = th.zeros(g.num_nodes('s'+str(i+1)), dtype=th.int32)
        is_last[last_nid[i]] = 1
        g.nodes['s'+str(i+1)].data['last'] = is_last
    return g

def label_last_k(g, last_nids):
    is_last = th.zeros(g.number_of_nodes(), dtype=th.int32)
    is_last[last_nids] = 1
    g.nodes['s1'].data['last'] = is_last
    return g

def seq_to_eop_multigraph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    if len(seq) > 1:
        seq_nid = [iid2nid[iid] for iid in seq]
        src = seq_nid[:-1]
        dst = seq_nid[1:]
    else:
        src = th.LongTensor([])
        dst = th.LongTensor([])
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata['iid'] = th.from_numpy(items)
    label_last(g, iid2nid[seq[-1]])
    return g

def seq_to_shortcut_graph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    seq_nid = [iid2nid[iid] for iid in seq]
    counter = Counter(
        [(seq_nid[i], seq_nid[j]) for i in range(len(seq)) for j in range(i, len(seq))]
    )
    edges = counter.keys()
    src, dst = zip(*edges)

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    return g

def seq_to_session_graph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    seq_nid = [iid2nid[iid] for iid in seq]
    counter = Counter(
        [(seq_nid[i], seq_nid[i+1]) for i in range(len(seq)-1)]
    )
    edges = counter.keys()
    if len(edges) > 0:
        src, dst = zip(*edges)
        weight = th.tensor(list(counter.values()))
    else:
        src, dst = [0], [0]
        weight = th.ones(1).long()

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    
    g.edata['w'] = weight
    # print(g.edata)
    g.ndata['iid'] = th.from_numpy(items)
    label_last(g, iid2nid[seq[-1]])

    return g

def seq_to_lg_graph(seq, order=1, coaDict=None):
 
    order1 = order
    order = min(order, len(seq))
    items = th.unique(th.tensor(seq).long(), sorted=False).numpy()
    # items, indices = np.unique(seq, return_inverse=True)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    
    num_nodes = len(items)
    seq_nid = [iid2nid[iid] for iid in seq]
    combine_seqs = []
    item_dicts = [iid2nid]
    cid2nid = {}
    item_g = []
    for i in range(1, order1):
        combine_seq = []
        item_dict = {}
        cnt = 0
        for j in range(len(seq_nid)-i):
            # item = combine(j, i+1)
            item = tuple(seq[j:j+i+1])
            if item not in item_dict:
                item_dict[item] = cnt
                cnt += 1
                combine_seq.append([seq[idx] for idx in range(j, j+i+1)])
        if len(combine_seq) == 0:
            combine_seq.append([items[0] for _ in range(i+1)])
        combine_seqs.append(combine_seq)
                
        item_dicts.append(item_dict)
    
    graph_data = {}
    cseiid2nid = {}
    cseqnid = []
    
    for idx, item1 in enumerate(items):
        for item2 in coaDict.keys():
            if len(item2) > 1:
                continue
            key = tuple(sorted([item1] + list(item2)))
            if key in coaDict:
                cseqnid.append(item2[0])
                if key not in cseiid2nid:
                    cseiid2nid[key] = len(cseiid2nid)
                if ('s1', 'link', 'g') not in graph_data:
                    graph_data[('g', 'link', 's1')] = ([cseiid2nid[key]], [item_dicts[0][item1]])
                    # graph_data[('g', 'link', 's'+str(i+1))] = ([cseiid2nid[key]], [item_dicts[i][item2]])
                    
                else:
                    # graph_data[('s1', 'link', 'g')][0].append(iid2nid[item])
                    # graph_data[('s1', 'link', 'g')][1].append(coaDict[key])
                    graph_data[('g', 'link', 's1')][1].append(iid2nid[item1])
                    graph_data[('g', 'link', 's1')][0].append(cseiid2nid[key])
                    # graph_data[('g', 'link', 's'+str(i+1))][1].append(item_dicts[i][item2])
                    # sgraph_data[('g', 'link', 's'+str(i+1))][0].append(cseiid2nid[key])

     
    # for idx, item1 in enumerate(items):
    #     for i in range(order1):
    #         citems = list(item_dicts[i].keys())
    #         for idx, item2 in enumerate(citems):
    #             if not isinstance(item2, tuple):
    #                 key = tuple(sorted([item1] + [item2]))
    #             else:
    #                 key = tuple(sorted([item1] + list(item2)))
                
    #             if  key in coaDict:
    #                 cseqnid.append(coaDict[key])
    #                 if key not in cseiid2nid:
    #                     cseiid2nid[key] = len(cseiid2nid)
    #                 if ('s1', 'link', 'g') not in graph_data:
    #                     graph_data[('g', 'link', 's1')] = ([cseiid2nid[key]], [item_dicts[0][item1]])
    #                     graph_data[('g', 'link', 's'+str(i+1))] = ([cseiid2nid[key]], [item_dicts[i][item2]])
                        
    #                 else:
    #                     # graph_data[('s1', 'link', 'g')][0].append(iid2nid[item])
    #                     # graph_data[('s1', 'link', 'g')][1].append(coaDict[key])
    #                     graph_data[('g', 'link', 's1')][1].append(iid2nid[item1])
    #                     graph_data[('g', 'link', 's1')][0].append(cseiid2nid[key])
    #                     graph_data[('g', 'link', 's'+str(i+1))][1].append(item_dicts[i][item2])
    #                     graph_data[('g', 'link', 's'+str(i+1))][0].append(cseiid2nid[key])
                        
    if ('g', 'link', 's1') not in graph_data:
        graph_data[('g', 'link', 's1')] = (th.LongTensor([]), th.LongTensor([]))
        
    # for i in range(1, order1):
    #     citems = list(item_dicts[i].keys())
    #     for item in citems:
    #         key = tuple(sorted(item))
    #         if  key in coaDict:
    #             cseqnid.append(coaDict[key])
    #             if key not in cseiid2nid:
    #                 cseiid2nid[key] = len(cseiid2nid)
    #             if ('g', 'link', 's'+str(i+1)) not in graph_data:
    #                 graph_data[('g', 'link', 's'+str(i+1))] = ([cseiid2nid[key]], [item_dicts[i][item]])
    #             else:
    #                 graph_data[('g', 'link', 's'+str(i+1))][1].append(item_dicts[i][item])
    #                 graph_data[('g', 'link', 's'+str(i+1))][0].append(cseiid2nid[key])
    
    for i in range(1):
        if ('g', 'link', 's'+str(i+1)) not in graph_data:
            graph_data[('g', 'link', 's'+str(i+1))] = (th.LongTensor([]), th.LongTensor([]))
        graph_data[('s'+str(i+1), 'link', 'g')] = (graph_data[('g', 'link', 's'+str(i+1))][1], graph_data[('g', 'link', 's'+str(i+1))][0])
    
    num_dict = {}
    num_dict['g'] = len(cseqnid)
    num_dict['s1'] = len(items)
    for i in range(1, order1):
        if len(item_dicts) > i: 
            num_dict['s'+str(i+1)] = len(item_dicts[i])
        num_dict['s'+str(i+1)] = max(1, num_dict['s'+str(i+1)])
            
    g = dgl.heterograph(graph_data, num_dict)
    g.nodes['g'].data['iid'] = th.tensor(cseqnid)
    g.nodes['s1'].data['iid'] = th.from_numpy(items)
    # print(num_dict['s2'], len(combine_seqs[0]))
    for i in range(1, order1):
        if g.num_nodes('s'+str(i+1)) == 0:
            g.add_nodes(1, ntype='s'+str(i+1))
        g.nodes['s'+str(i+1)].data['iid'] = th.from_numpy(np.array(combine_seqs[i-1]))
    # print(g)
    return g

def seq_to_ccs_graph(seq, order=1):
    # print(seq)
    order1 = order
    order = min(order, len(seq))
    # items = th.unique(th.tensor(seq).long(), sorted=False, return_inverse).numpy()
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    
    num_nodes = len(items)
    seq_nid = [iid2nid[iid] for iid in seq]
    last_k = [iid2nid[seq[-1]]]
    combine_seqs = []
    
    def com(i, order):
        item = str(seq[i:i+order])
        # for k in range(order):
        #     item += str(seq_nid[i+k])
        return item 
    
    class combine:
        def __init__(self):
            self.dict = {}
        
        def __call__(self, *input):
            return self.forward(*input)    
        
        def forward(self, i, order):
            if str(i) not in self.dict:
                self.dict[str(i)] = {}
            if order not in self.dict[str(i)]:
                self.dict[str(i)][order] = com(i, order)
            return self.dict[str(i)][order]
        
    combine = combine()  
    
    item_dicts = [iid2nid]
    cid2nid = {}
    item_g = []
    for i in range(1, order1):
        combine_seq = []
        item_dict = {}
        cnt = 0
        for j in range(len(seq_nid)-i):
            item = combine(j, i+1)
            if item not in item_dict:
                item_dict[item] = cnt
                cnt += 1
                combine_seq.append([seq[idx] for idx in range(j, j+i+1)])
    
        if len(item_dict) > 0:
            last_k.append(item_dict[item])
        else:
            last_k.append(0)
        combine_seqs.append(combine_seq)
                
        item_dicts.append(item_dict)
    
    graph_data = {}
    
    for k in range(order):
        if k == 0:
            counter = Counter([(seq_nid[i], seq_nid[i+1]) for i in range(len(seq)-1)]) ## original connect
        else:       
            counter = Counter([(item_dicts[k][combine(i, k+1)], item_dicts[k][combine(i+1, k+1)]) for i in range(len(seq)-k-1)])
        
        edges = counter.keys()
        
        if len(edges) > 0:
            src, dst = zip(*edges)
            weight = th.tensor(list(counter.values()))
        else:
            src, dst = [], []
            weight = th.ones(1).long()
        
        graph_data[('s'+str(k+1), 'intra'+str(k+1), 's'+str(k+1))] = (th.tensor(src).long(), th.tensor(dst).long())

    for k in range(1, order): 
       
        counter = Counter([(seq_nid[i], item_dicts[k][combine(i+1, k+1)]) for i in range(len(seq)-k-1)])
        
        edges = counter.keys()
        
        if len(edges) > 0:
            src, dst = zip(*edges)
            weight = th.tensor(list(counter.values()))
        else:
            src, dst = th.LongTensor([]), th.LongTensor([])
            weight = th.ones(1).long()

        graph_data[('s1', 'inter', 's'+str(k+1))] = (src, dst)
        
        counter = Counter([(item_dicts[k][combine(i, k+1)], seq_nid[i+k+1]) for i in range(len(seq)-k-1)])
        
        edges = counter.keys()
    
        if len(edges) > 0:
            src, dst = zip(*edges)
            weight = th.tensor(list(counter.values()))
        else:
            src, dst = th.LongTensor([]), th.LongTensor([])
            weight = th.ones(1).long()
        
        graph_data[('s'+str(k+1), 'inter', 's1')] = (src, dst)
    
    if order < order1:
        for i in range(order, order1):
            graph_data[('s'+str(i+1), 'intra'+str(i+1), 's'+str(i+1))]=(th.LongTensor([]), th.LongTensor([]))
            graph_data[('s'+str(i+1), 'inter', 's1')]=(th.LongTensor([]), th.LongTensor([]))
            graph_data[('s1', 'inter', 's'+str(i+1))]=(th.LongTensor([]), th.LongTensor([]))
    
    g = dgl.heterograph(graph_data)
    # print(g.num_nodes('s2'))
    if g.num_nodes('s1') == 0:
        g.add_nodes(1, ntype='s1')
    g.nodes['s1'].data['iid'] = th.from_numpy(items)
    
    if order < order1:
        for i in range(order, order1):
            if 's'+str(i+1) not in g.ntypes or g.num_nodes('s'+str(i+1)) == 0:
                g.add_nodes(1, ntype='s'+str(i+1))
                g.nodes['s'+str(i+1)].data['iid'] = th.ones(1, i+1).long() * g.nodes['s1'].data['iid'][0]
                # print(g.nodes['s'+str(i+1)].data)
    for i in range(1, order):
        if g.num_nodes('s'+str(i+1)) == 0:
            g.add_nodes(1, ntype='s'+str(i+1))
        
        g.nodes['s'+str(i+1)].data['iid'] = th.from_numpy(np.array(combine_seqs[i-1]))
    
    label_last_ccs(g, last_k)
    indices = np.array(seq_nid)
    # print(indices, last_k)
    return g, indices

def seq_to_hyper_graph(seq, order=1):
    
    order1 = order
    order = min(order, len(seq))
    # items = th.unique(th.tensor(seq).long(), sorted=False, return_inverse).numpy()
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    
    num_nodes = len(items)
    seq_nid = [iid2nid[iid] for iid in seq]
    last_k = [iid2nid[seq[-1]]]
    
    item_dicts = [iid2nid]
    cid2nid = {}
    
    graph_data = {}

    for k in range(1, order): 
        counter = Counter()
        for i in range(len(seq)-k):
            countert = Counter([(seq_nid[j], i) for j in range(i+k+1)])
            counter.update(countert)
        last_k.append(max(0, len(seq)-k-1))
        edges = counter.keys()
        
        if len(edges) > 0:
            src, dst = zip(*edges)
            weight = th.tensor(list(counter.values()))
        else:
            src, dst = th.LongTensor([]), th.LongTensor([])
            weight = th.ones(1).long()

        graph_data[('s1', 'belong', 's'+str(k+1))] = (src, dst)
    for _ in range(order1 - len(last_k)):
        last_k.append(0)

    if order < order1:
        for i in range(order, order1):
            graph_data[('s1', 'belong', 's'+str(i+1))]=(th.LongTensor([]), th.LongTensor([]))
    
    g = dgl.heterograph(graph_data)
    # print(g.num_nodes('s2'))
    if g.num_nodes('s1') == 0:
        g.add_nodes(1, ntype='s1')
    g.nodes['s1'].data['iid'] = th.from_numpy(items)
    
    # if order < order1:
    #     for i in range(order, order1):
    #         if 's'+str(i+1) not in g.ntypes or g.num_nodes('s'+str(i+1)) == 0:
    #             g.add_nodes(1, ntype='s'+str(i+1))
    #             g.nodes['s'+str(i+1)].data['iid'] = th.ones(1, i+1).long() * g.nodes['s1'].data['iid'][0]
    #             # print(g.nodes['s'+str(i+1)].data)
    for i in range(1, order1):
        if g.num_nodes('s'+str(i+1)) == 0:
            g.add_nodes(1, ntype='s'+str(i+1))
        
        g.nodes['s'+str(i+1)].data['iid'] = th.zeros(g.num_nodes('s'+str(i+1)), dtype=th.long)
    
    # print(g.num_nodes('s1'), g.num_nodes('s2'))
    
    label_last_ccs(g, last_k)
    # indices = np.array(seq_nid)
    # print(indices, last_k)
   
    return g
    
    

# def seq_to_ccs_graph(seq, order=1, coaDict=None):
    
#     order1 = order
    
#     order = min(order, len(seq))
#     items = th.unique(th.tensor(seq).long(), sorted=False).numpy()
#     # items, indices = np.unique(seq, return_inverse=True)
#     iid2nid = {iid: i for i, iid in enumerate(items)}
#     num_nodes = len(items)
#     seq_nid = [iid2nid[iid] for iid in seq]
#     last_k = [iid2nid[seq[-1]]]
#     combine_seqs = []
    
#     def com(i, order):
#         item = str(seq[i:i+order])
#         # for k in range(order):
#         #     item += str(seq_nid[i+k])
#         return item 
    
#     class combine:
#         def __init__(self):
#             self.dict = {}
        
#         def __call__(self, *input):
#             return self.forward(*input)    
        
#         def forward(self, i, order):
#             if str(i) not in self.dict:
#                 self.dict[str(i)] = {}
#             if order not in self.dict[str(i)]:
#                 self.dict[str(i)][order] = com(i, order)
#             return self.dict[str(i)][order]
        
#     combine = combine()  
    
#     item_dicts = [iid2nid]
#     cid2nid = {}
#     for i in range(1, order1):
#         combine_seq = []
#         item_dict = {}
#         cnt = 0
#         for j in range(len(seq_nid)-i):
#             item = combine(j, i+1)
#             if item not in item_dict:
#                 item_dict[item] = cnt
#                 cnt += 1
#                 combine_seq.append([seq[j:j+i+1]])
            
#             ## retrieve global item
#             # item0 = tuple(sorted(seq[j:j+i+1]))
#             # if item0 in coaDict:
#             #     if item0 not in cid2nid:
#             #         cid2nid[item0] = len(cid2nid)
#         if len(item_dict) > 0:
#             last_k.append(item_dict[item])
#         else:
#             last_k.append(0)
#         combine_seqs.append(combine_seq)
                
#         item_dicts.append(item_dict)
    
#     graph_data = {}
    
#     for k in range(order):
#         if k == 0:
#             counter = Counter([(seq_nid[i], seq_nid[i+1]) for i in range(len(seq)-1)]) ## original connect
#         else:       
#             counter = Counter([(item_dicts[k][combine(i, k+1)], item_dicts[k][combine(i+1, k+1)]) for i in range(len(seq)-k-1)])
        
#         edges = counter.keys()
        
#         if len(edges) > 0:
#             src, dst = zip(*edges)
#             weight = th.tensor(list(counter.values()))
#         else:
#             src, dst = [], []
#             weight = th.ones(1).long()
        
#         graph_data[('s'+str(k+1), 'intra'+str(k+1), 's'+str(k+1))] = (th.tensor(src).long(), th.tensor(dst).long())

#     for k in range(1, order): 
       
#         counter = Counter([(seq_nid[i], item_dicts[k][combine(i+1, k+1)]) for i in range(len(seq)-k-1)])
        
#         edges = counter.keys()
        
#         if len(edges) > 0:
#             src, dst = zip(*edges)
#             weight = th.tensor(list(counter.values()))
#         else:
#             src, dst = th.LongTensor([]), th.LongTensor([])
#             weight = th.ones(1).long()

#         graph_data[('s1', 'inter', 's'+str(k+1))] = (src, dst)
        
#         counter = Counter([(item_dicts[k][combine(i, k+1)], seq_nid[i+k+1]) for i in range(len(seq)-k-1)])
        
#         edges = counter.keys()
    
#         if len(edges) > 0:
#             src, dst = zip(*edges)
#             weight = th.tensor(list(counter.values()))
#         else:
#             src, dst = th.LongTensor([]), th.LongTensor([])
#             weight = th.ones(1).long()
        
#         graph_data[('s'+str(k+1), 'inter', 's1')] = (src, dst)
    
#     # for idx, item in enumerate(items):
#     #     for key, value in cid2nid.items():
#     #         if item in key:
#     #             if ('s1', 'link', 'g') not in graph_data:
#     #                 graph_data[('s1', 'link', 'g')] = ([iid2nid[item]], [value])
#     #                 graph_data[('g', 'link', 's1')] = ([iid2nid[item]], [value])
#     #             else:
#     #                 graph_data[('s1', 'link', 'g')][0].append(iid2nid[item])
#     #                 graph_data[('s1', 'link', 'g')][1].append(value)
#     #                 graph_data[('g', 'link', 's1')][1].append(iid2nid[item])
#     #                 graph_data[('g', 'link', 's1')][0].append(value)
    
#     # if ('s1', 'link', 'g') not in graph_data:
#     #     graph_data[('s1', 'link', 'g')] = (th.LongTensor([]), th.LongTensor([]))
#     #     graph_data[('g', 'link', 's1')] = (th.LongTensor([]), th.LongTensor([]))
    
#     # print(graph_data[('s1', 'link', 'g')])
    
#     if order < order1:
#         for i in range(order, order1):
#             graph_data[('s'+str(i+1), 'intra'+str(i+1), 's'+str(i+1))]=(th.LongTensor([]), th.LongTensor([]))
#             graph_data[('s'+str(i+1), 'inter', 's1')]=(th.LongTensor([]), th.LongTensor([]))
#             graph_data[('s1', 'inter', 's'+str(i+1))]=(th.LongTensor([]), th.LongTensor([]))
    
#     g = dgl.heterograph(graph_data)
 
#     if g.num_nodes('s1') == 0:
#         g.add_nodes(1, ntype='s1')
#     g.nodes['s1'].data['iid'] = th.from_numpy(items)
    
#     if order < order1:
#         for i in range(order, order1):
#             if 's'+str(i+1) not in g.ntypes or g.num_nodes('s'+str(i+1)) == 0:
#                 g.add_nodes(1, ntype='s'+str(i+1))
#                 g.nodes['s'+str(i+1)].data['iid'] = th.ones(1, i+1).long() * g.nodes['s1'].data['iid'][0]
#                 # print(g.nodes['s'+str(i+1)].data)
#     for i in range(1, order):
#         if g.num_nodes('s'+str(i+1)) == 0:
#             g.add_nodes(1, ntype='s'+str(i+1))
        
#         g.nodes['s'+str(i+1)].data['iid'] = th.from_numpy(np.array(combine_seqs[i-1]))
    
#     label_last_ccs(g, last_k)
#     return g

# def seq_to_ccs_graph(graph, order=1):
    
#     order1 = order
    
#     graph_data, last_k, items, combine_seqs, order = graph
    
#     g = dgl.heterograph(graph_data)
 
#     if g.num_nodes('s1') == 0:
#         g.add_nodes(1, ntype='s1')
#     g.nodes['s1'].data['iid'] = th.from_numpy(items)
    
#     if order < order1:
#         for i in range(order, order1):
#             if 's'+str(i+1) not in g.ntypes or g.num_nodes('s'+str(i+1)) == 0:
#                 g.add_nodes(1, ntype='s'+str(i+1))
#                 g.nodes['s'+str(i+1)].data['iid'] = th.ones(1, i+1).long() * g.nodes['s1'].data['iid'][0]
#                 # print(g.nodes['s'+str(i+1)].data)
#     for i in range(1, order):
#         if g.num_nodes('s'+str(i+1)) == 0:
#             g.add_nodes(1, ntype='s'+str(i+1))
        
#         g.nodes['s'+str(i+1)].data['iid'] = th.from_numpy(np.array(combine_seqs[i-1]))
    
#     label_last_ccs(g, last_k)
#     return g# , indices
    
# def seq_to_ccs_inter_graph(seq, order=2):
    
#     order = min(order, len(seq))
#     items = np.unique(seq)
#     iid2nid = {iid: i for i, iid in enumerate(items)}
    
#     num_nodes = len(items)
#     cnt = num_nodes
#     seq_nid = [iid2nid[iid] for iid in seq]
#     # last_k = [iid2nid[seq[-1]]]
    
#     def combine(i, order):
#         item = ""
#         for k in range(order):
#             item += str(seq_nid[i+k])
#         return item 
    
#     for i in range(1, order):
#         for j in range(len(seq_nid)-i):
#             item = combine(j, i+1)
#             if item not in iid2nid:
#                 iid2nid[item] = cnt
#                 cnt += 1
                
#     # counter = Counter(
#     #     [(seq_nid[i], seq_nid[i+1]) for i in range(len(seq)-1)]
#     # ) ## original connect
#     counter = Counter()
    
#     for k in range(1, order):
#         counter.update(Counter(
#             [(seq_nid[i], iid2nid[combine(i+1, k+1)]) for i in range(len(seq)-k-1)]) 
#         )
#         counter.update(Counter(
#             [(iid2nid[combine(i, k+1)], seq_nid[i+k+1]) for i in range(len(seq)-k-1)]) 
#         )
        
#     edges = counter.keys()
            
#     if len(edges) > 0:
#         src, dst = zip(*edges)
#         weight = th.tensor(list(counter.values()))
#     else:
#         src, dst = [0], [0]
#         weight = th.ones(1).long()

#     g = dgl.graph((src, dst))
    
#     g.edata['w'] = weight
#     # print(g.edata)
#     g.ndata['iid'] = th.from_numpy(np.pad(items, (0, len(iid2nid)-len(items))))
#     # label_last(g, last_k)
    
#     return g
            
def collate_fn_factory(*seq_to_graph_fns):
    def collate_fn(samples):
        seqs, labels = zip(*samples)
        inputs = []
        for seq_to_graph in seq_to_graph_fns:
            graphs = list(map(seq_to_graph, seqs))        
            bg = dgl.batch(graphs)
            inputs.append(bg)
        labels = th.LongTensor(labels)
        return inputs, labels

    return collate_fn

def collate_fn_factory_ccs(seq_to_graph_fns, order):
    # coaDict = pickle.load(open(dataset_dir + '/coaDict.pkl', 'rb'))
    def collate_fn(samples):
        seqs, labels = zip(*samples)
        labels1 = []
        for idx, label in enumerate(labels):
            if label in seqs[idx]:
                labels1.append(0)
            else:
                labels1.append(1)
        inputs = []
        graphs = []
        indices = []
        cnt = 0
        for seq_to_graph in seq_to_graph_fns:
            batch = list(map(seq_to_graph, seqs, [order for _ in range(len(seqs))]))
            if cnt == 0:
                for idx, bh in enumerate(batch):
                    graph, indice = bh
                    graphs.append(graph)
                    indices.append(indice)
                bg = dgl.batch(graphs)
                cnt = 1
            else:
                bg = dgl.batch(batch)
            inputs.append(bg)
        labels1 = th.LongTensor(labels1)
        labels = th.LongTensor(labels)
        # print(inputs[1].num_nodes('s2'), len(inputs[1].nodes['s2'].data['iid']))
        # indices = [th.randint(10, (10,)), th.randint(10, (10,))]
        # print(inputs[0].num_nodes('s2'), inputs[1].num_nodes('s2'))
        return inputs, indices, labels, labels1

    return collate_fn

def collate_fn_factory_hyper(seq_to_graph_fns, order):
    def collate_fn(samples):
        seqs, labels = zip(*samples)
        inputs = []
        graphs = []
        cnt = 0
        for seq_to_graph in seq_to_graph_fns:
            batch = list(map(seq_to_graph, seqs, [order for _ in range(len(seqs))]))
            if cnt == 0:
                for idx, bh in enumerate(batch):
                    graph = bh
                    graphs.append(graph)
                bg = dgl.batch(graphs)
                cnt = 1
            else:
                bg = dgl.batch(batch)
            inputs.append(bg)
        labels = th.LongTensor(labels)
        
        return inputs, labels
    
    return collate_fn

if __name__ == '__main__':
    
    seq = [3, 1, 3, 6, 2, 5, 1, 2, 4, 1, 2] # 2, 0, 2, 5, 1, 4, 0, 1, 3, 0, 1 
    seq0 = [250, 250, 250, 250, 3, 1, 2, 4, 1]
    # g1 = seq_to_ccs_graph(seq, order=4)
    # g2 = seq_to_ccs_graph(seq, order=2)
    collate_fn = collate_fn_factory_ccs(seq_to_ccs_graph, order=2)
    seqs = [[seq, 1], [seq0, 2]]
    print(collate_fn(seqs)[0][0].batch_num_nodes('s2'))
