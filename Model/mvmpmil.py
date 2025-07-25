import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.nystrom_attention import NystromAttention
import numpy as np
from Model.linearatt import MultiheadLinearAttention

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,d=0.3):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout= d
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class CrossLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,d=0.3):
        super().__init__()
        self.attn = MultiheadLinearAttention(embed_dim=dim,num_heads=8,dropout=d)

    def forward(self,q,k,v):

        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)
        x,attention= self.attn(q,k,v)

        return x.permute(1,0,2), attention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout1d(drop)
        self.act2 = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        #x = self.act2(x)
        return x

class optimizer_triple(nn.Module):
    def __init__(self, in_feature,out_feature,drop=0.):
        super().__init__()
        self.infeature = in_feature
        self.outfeature = out_feature
        self.drop_rate = drop
        #print(self.infeature)
        self.fc1 = nn.Linear(self.infeature,self.outfeature)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(self.drop_rate)

        self.fc2 = nn.Linear(self.outfeature,self.outfeature)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(self.drop_rate)

    def forward(self, x, mode):
        if mode == 'global':
            x = self.fc1(x)
            x = self.act1(x)
            x = self.fc2(x)
            x = self.act2(x)

        else:
            x = self.fc1(x)
            # x = self.act1(x)
            # x = self.drop1(x)
            # x = self.fc2(x)
            # x = self.act2(x)
            # x = self.drop2(x)
        
        return x
class ATTN(nn.Module):
    def __init__(self,L,D):
        super(ATTN, self).__init__()
        self.D=D
        self.L=L

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, 1)
    def forward(self,H):
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL
        return M

class MvMpMIL(nn.Module):
    def __init__(self, in_features, num_classes=2, L=512,n_pattern=2, dropout_node=0.0):
        super().__init__()
        self.in_size=in_features
        self.L = L
        self.D=128
        self.n_pattern=n_pattern

        # global lesion representation learning 
        self.m = 0.4

        self.fc=nn.Linear(in_features,512)
        self.head_n_pattern=1
        self.tail_n_pattern=1
        self.head_token = nn.Parameter(torch.randn(self.head_n_pattern, self.in_size))
        self.tail_token = nn.Parameter(torch.randn(self.tail_n_pattern, self.in_size))


        # self.triple_optimizer = optimizer_triple(in_feature=in_features,out_feature=self.L,drop=dropout_patch)

        self.W_head = nn.Linear(self.in_size, self.L)
        self.W_tail = nn.Linear(self.in_size, self.L)

        self.encoder_head_instances = nn.Sequential(
            TransLayer(dim=self.L,d=dropout_node),
            # nn.LayerNorm(self.L),
        )
        self.encoder_tail_instances = nn.Sequential(
            TransLayer(dim=self.L, d=dropout_node),
            # nn.LayerNorm(self.L),
        )

        self.H_attn=ATTN(self.L,self.D)
        self.T_attn=ATTN(self.L,self.D)


        self.H_classifier = nn.Sequential(
            nn.Linear(self.L,num_classes)
        )
        self.T_classifier = nn.Sequential(
            nn.Linear(self.L, num_classes)
        )
        self.loss_fn=nn.BCEWithLogitsLoss()

    def forward(self, x,y):
        x=x.squeeze(0)

        # x=self.fc(x)
        head_x = torch.cat((self.head_token,x),dim=0)
        head_x=self.W_head(head_x)

        tail_x = torch.cat((self.tail_token,x),dim=0)
        tail_x=self.W_tail(tail_x)

        head = self.encoder_head_instances(head_x.unsqueeze(0))
        tail = self.encoder_tail_instances(tail_x.unsqueeze(0))

        H = torch.cat((head[0,:self.head_n_pattern,:],tail[0,:self.tail_n_pattern,:]),dim=0)
        # H=head[0,:self.head_n_pattern,:]
        # T=tail[0,:self.tail_n_pattern,:]

        H_M=self.H_attn(H)
        # T_M=self.T_attn(T)

        H_Y_prob = self.H_classifier(H_M)
        # T_Y_prob = self.T_classifier(T_M)

        # loss = self.loss_fn(H_Y_prob,y) + self.loss_fn(T_Y_prob,y)+0.1*self.similarity_loss(H)+0.1*self.similarity_loss(T)
        loss = self.loss_fn(H_Y_prob,y) + +0.1*self.similarity_loss(H)

        return H_Y_prob,loss

        # return (H_Y_prob+T_Y_prob)/2, loss

    def similarity_loss(self,patterns):

        sim_matrix = F.cosine_similarity(patterns.unsqueeze(1), patterns.unsqueeze(0), dim=2).fill_diagonal_(0)
        loss_dis = sim_matrix.mean()

        return loss_dis




if __name__ == "__main__":
    milnet = MvMpMIL(1536, dropout_node=0.1)
    print(milnet)
    y=torch.tensor([0])
    logits, loss= milnet(torch.randn(1,100, 1536),y)

