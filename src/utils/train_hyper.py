import time

import torch as th
from torch import nn, optim

import numpy as np

from sklearn.metrics import accuracy_score


# ignore weight decay for parameters in bias, batch norm and activation
def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def prepare_batch(batch, device):
    inputs, labels= batch
    inputs_gpu = [x.to(device) for x in inputs]
    # indices_gpu = [th.LongTensor(x).to(device) for x in indices]
    labels_gpu = labels.to(device)
    # labels1_gpu = labels1.to(device)
    return inputs_gpu, labels_gpu


def evaluate(model, data_loader, device, cutoff=20):
    model.eval()
    mrr = 0
    hit = 0
    num_samples = 0
    with th.no_grad():
        for batch in data_loader:
            inputs, labels= prepare_batch(batch, device)
            logits, _, phi = model(*inputs)
            batch_size = logits.size(0)
            num_samples += batch_size
            topk = logits.topk(k=cutoff)[1]
            #print(topk)
            labels = labels.unsqueeze(-1)
            hit_ranks = th.where(topk == labels)[1] + 1
            hit += hit_ranks.numel()
            mrr += hit_ranks.float().reciprocal().sum().item()
            # phi = phi[:int(len(phi)/2)]
            # acc1 = accuracy_score(labels1.detach().cpu().numpy(), np.argmax(phi[:int(len(phi)/2)].detach().cpu().numpy(), axis=1))
            # acc2 = accuracy_score(labels1.detach().cpu().numpy(), np.argmax(phi[int(len(phi)/2):].detach().cpu().numpy(), axis=1))
    return mrr / num_samples, hit / num_samples# , acc1, acc2


class TrainRunner:
    def __init__(
        self,
        dataset,
        model,
        train_loader,
        test_loader,
        device,
        lr=1e-3,
        weight_decay=0,
        patience=3,
    ):
        self.model = model
        if weight_decay > 0:
            params = fix_weight_decay(model)
        else:
            params = model.parameters()
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        # self.optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.patience = patience
        self.weights = th.tensor(np.load(dataset+'/weights.npy')).float().to(self.device) + 1

    def train(self, epochs, log_interval=100):
        max_mrr = 0
        max_hit = 0
        bad_counter = 0
        t = time.time()
        mean_loss = 0
        mean_bc_loss = 0
        # mrr, hit, acc1, acc2 = evaluate(self.model, self.test_loader, self.device)
        # print(mrr, hit, acc1, acc2)
        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                inputs, labels = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()
                scores, kl_loss, phi = self.model(*inputs)
                
                # loss = nn.functional.cross_entropy(logits, labels)
                loss = nn.functional.nll_loss(scores, labels) + kl_loss
                # bc_loss = nn.functional.nll_loss(phi, labels1.repeat(2))
                
                # loss += bc_loss
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                self.optimizer.step()
                mean_loss += loss.item() / log_interval
                # mean_bc_loss += bc_loss.item() / log_interval
                mean_bc_loss = 0
                if self.batch > 0 and self.batch % log_interval == 0:
                    print(
                        f'Batch {self.batch}: Loss = {mean_loss:.4f}, BC Loss = {mean_bc_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s'
                    )
                    t = time.time()
                    mean_loss = 0
                    mean_bc_loss = 0
                self.batch += 1
            self.scheduler.step()
            mrr, hit = evaluate(self.model, self.test_loader, self.device)

            print(f'Epoch {self.epoch}: MRR = {mrr * 100:.3f}%, Hit = {hit * 100:.3f}%')
            self.model.inc_epoch()
            if mrr < max_mrr and hit < max_hit:
                bad_counter += 1
                if bad_counter == self.patience:
                    break
            else:
                bad_counter = 0
            max_mrr = max(max_mrr, mrr)
            max_hit = max(max_hit, hit)
            self.epoch += 1
        return max_mrr, max_hit
