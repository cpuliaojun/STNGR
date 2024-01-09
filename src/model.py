# -*- encoding: utf-8 -*-
"""
@file         :    model.py
@description  :
@date         :    2022/8/12 15:37
@author       :    silentpotato
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, seq_len, scaffold_maxlen, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        num = int(scaffold_maxlen)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len + num, seq_len + num))
                             .view(1, 1, seq_len + num, seq_len + num))
        self.n_head = n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()  # [batch,seq,embed]

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save


class Block(nn.Module):
    def __init__(self, nembd, nhead, seq_len, scaf_len, attn_pdrop, resid_pdrop):
        super().__init__()
        self.norm1 = nn.LayerNorm(nembd)
        self.norm2 = nn.LayerNorm(nembd)
        self.attn = CausalSelfAttention(nembd, nhead, seq_len, scaf_len, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(nembd, 4 * nembd),
            nn.GELU(),
            nn.Linear(4 * nembd, nembd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        y, attn = self.attn(self.norm1(x))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x, attn


class GPT(nn.Module):
    def __init__(self, ntoken, seq_len, scaf_len, nembd, nhead, nlayer, embd_pdrop, attn_pdrop, resid_pdrop,lstm=False,encoder=False):
        super(GPT, self).__init__()
        # para
        self.nembd = nembd
        self.nhead = nhead
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        # embedding
        self.token_emb = nn.Embedding(ntoken, nembd)
        self.type_emb = nn.Embedding(2, nembd)
        # self.prop_nn = nn.Linear(num_props,nembd)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, nembd))
        self.scaf_pos_emb = nn.Parameter(torch.zeros(1,scaf_len,nembd))
        self.drop = nn.Dropout(embd_pdrop)
        # transformer block
        self.blocks = nn.Sequential(*[Block(nembd, nhead, seq_len, scaf_len, attn_pdrop, resid_pdrop)
                                      for _ in range(nlayer)])
        # decoder head
        self.norm = nn.LayerNorm(nembd)
        self.projection = nn.Linear(nembd, ntoken, bias=False)
        self.seqlen = seq_len
        self.scaflen = scaf_len
        self.lstm = lstm
        self.encoder = encoder
        if self.lstm:
            self.lstm = nn.LSTM(input_size = nembd, hidden_size = nembd, num_layers = 2, dropout = 0.3,
                                bidirectional = False,batch_first=True)
        if self.encoder:
            self.encoderlayer = nn.TransformerEncoderLayer(d_model=128,nhead=2,dim_feedforward=64,dropout=0,batch_first=True)
            self.encoder = nn.TransformerEncoder(self.encoderlayer,1)
        self.apply(self._init_weights)

    def get_seqlen(self):
        return self.seqlen

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, src, scaf):
        batch, seq = src.size()
        assert seq <= self.seqlen, "Cannot forward, model seqlen is too small."

        # 对于src
        token_emb = self.token_emb(src)  # [batch,seq,embd]
        pos_emb = self.pos_emb[:, :seq, :]  # 可学习参数 [1,seq,nembd]
        type_emb = self.type_emb(torch.ones((batch, seq), dtype=torch.long, device=src.device))  # [batch,seq,embd]
        x = self.drop(token_emb + pos_emb + type_emb)  # [batch,seq,embd]

        # 对于scaf：
        type_emb = self.type_emb(torch.zeros((batch, 1), dtype=torch.long, device=src.device))  # [batch,1,embd]
        scaf_emb = self.token_emb(scaf)  # scaf [batch,scaflen] => [batch,scaflen,embd]

        scaf_pos_emb = self.scaf_pos_emb


        if self.lstm:
            scaf_emb = self.lstm(scaf_emb)[0]
        if self.encoder:
            scaf_emb = self.encoder(scaf_emb+scaf_pos_emb)
        scaf_emb = scaf_emb+ type_emb  # [batch,scaflen,nembd]
        x = torch.cat([scaf_emb, x], 1)  # [batch,scaflen+seq-1,embd] [512,164,256]
        for layer in self.blocks:
            x, attn = layer(x)
        x = self.norm(x)
        output = self.projection(x)

        logits = output[:, self.scaflen:, :]  # [batch,seq,ntokens]

        return F.log_softmax(logits, dim=-1), logits



@torch.no_grad()
def sample(model,vocubulary,x, steps, temperature=1.0, top_k=None, scaf=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.

    x:[batch,1] repeat
    scaffold: [batch,1] repeat

    """
    # block_size = model.get_block_size() maxlen
    maxlen = 128
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= maxlen else x[x:, -maxlen:]  # [batch,1]
        _,logits = model(src=x_cond, scaf=scaf)  # for liggpt [batch,x_cond,ntokens]
        logits = logits[:, -1, :] / temperature   #[batch,ntokens]
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)#[batch,1]
        if ix.sum() == vocubulary.dictionary['<PAD>']:
            return x
        x = torch.cat((x, ix), dim=1)

    return x

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out



