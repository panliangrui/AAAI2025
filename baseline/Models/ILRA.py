import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Exploring Low-Rank Property in Multiple Instance Learning for Whole Slide Image Classification
Jinxi Xiang et al. ICLR 2023
"""


class MultiHeadAttention(nn.Module):
    """
    multi-head attention block
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super(MultiHeadAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.gate = None
        if gated:
            self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU())

    def forward(self, Q, K,returen_attn=False):

        Q0 = Q

        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)

        A, _ = self.multihead_attn(Q, K, V)

        O = (Q + A).transpose(0, 1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if self.gate is not None:
            O = O.mul(self.gate(Q0))
        if returen_attn:
            return O,_.squeeze()
        return O


class GAB(nn.Module):
    """
    equation (16) in the paper
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(GAB, self).__init__()
        self.latent = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # low-rank matrix L

        nn.init.xavier_uniform_(self.latent)

        self.project_forward = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        """
        This process, which utilizes 'latent_mat' as a proxy, has relatively low computational complexity.
        In some respects, it is equivalent to the self-attention function applied to 'X' with itself,
        denoted as self-attention(X, X), which has a complexity of O(n^2).
        """
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H = self.project_forward(latent_mat, X)  # project the high-dimensional X into low-dimensional H
        X_hat = self.project_backward(X, H)  # recover to high-dimensional space X_hat

        return X_hat

import torch.nn as nn
import torch.distributed as dist


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

class NLP(nn.Module):
    """
    To obtain global features for classification, Non-Local Pooling is a more effective method
    than simple average pooling, which may result in degraded performance.
    """

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(NLP, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        global_embedding = self.S.repeat(X.size(0), 1, 1)
        ret,attn_softmax = self.mha(global_embedding, X,True)
        return ret,attn_softmax


class ILRA(nn.Module):
    def __init__(self, num_layers=2, feat_dim=1024, n_classes=2, hidden_feat=256, num_heads=8, topk=1, ln=True,
                 confounder_path=False, confounder_learn=False, confounder_dim=128, confounder_merge='cat'):
        super().__init__()
        # stack multiple GAB block
        gab_blocks = []
        for idx in range(num_layers):
            block = GAB(dim_in=feat_dim if idx == 0 else hidden_feat,
                        dim_out=hidden_feat,
                        num_heads=num_heads,
                        num_inds=topk,
                        ln=ln)
            gab_blocks.append(block)

        self.gab_blocks = nn.ModuleList(gab_blocks)
        self.pooling = NLP(dim=hidden_feat, num_heads=num_heads, num_seeds=topk, ln=ln)

        self.confounder_merge = confounder_merge
        self.confounder_path = None
        if confounder_path:
            print('deconfounding')
            self.confounder_path = confounder_path

            conf_tensor = torch.from_numpy(np.load(confounder_path)).view(-1, hidden_feat).float()
            conf_tensor_dim = conf_tensor.shape[-1]
            if confounder_learn:
                self.confounder_feat = nn.Parameter(conf_tensor, requires_grad=True)
            else:
                self.register_buffer("confounder_feat", conf_tensor)
            joint_space_dim = confounder_dim
            # self.confounder_W_q = nn.Linear(in_size, joint_space_dim)
            # self.confounder_W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            self.W_q = nn.Linear(hidden_feat, joint_space_dim)
            self.W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            if confounder_merge == 'cat':
                self.classifier = nn.Linear(hidden_feat+ conf_tensor_dim, n_classes)
            else:
                self.classifier = nn.Linear(in_features=hidden_feat, out_features=n_classes)
        else:
            self.classifier = nn.Linear(in_features=hidden_feat, out_features=n_classes)
        initialize_weights(self)
    def forward(self, x):
        x=torch.unsqueeze(x,0)
        for block in self.gab_blocks:
            x = block(x)

        feat,attn_softmax = self.pooling(x)
        feat = feat.squeeze(0)

        if self.confounder_path:
            device = feat.device
            bag_q = self.W_q(feat)
            conf_k = self.W_k(self.confounder_feat)
            deconf_A = torch.mm(conf_k, bag_q.transpose(0, 1))
            deconf_A = F.softmax(
                deconf_A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)),
                0)  # normalize attention scores, A in shape N x C,
            conf_feats = torch.mm(deconf_A.transpose(0, 1),
                                  self.confounder_feat)  # compute bag representation, B in shape C x V
            if self.confounder_merge == 'cat':
                feat = torch.cat((feat, conf_feats), dim=1)
            elif self.confounder_merge == 'add':
                feat = feat + conf_feats
            elif self.confounder_merge == 'sub':
                feat = feat - conf_feats

        logits = self.classifier(feat)

        return logits


if __name__ == "__main__":
    model = ILRA(feat_dim=1024, n_classes=2, hidden_feat=256, num_heads=8, topk=1)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"num of params: {num_params}")

    x = torch.randn((1600, 1024))
    model.eval()
    logits, prob, y_hat = model({"x":x})
    print(f"y shape: {logits.shape}")