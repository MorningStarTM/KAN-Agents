import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import math



class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.Q = nn.Linear(h_dim, h_dim)
        self.K = nn.Linear(h_dim, h_dim)
        self.V = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.atten_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((self.max_T, self.max_T))
        mask = torch.tril(ones).view(1, 1, self.max_T, self.max_T)

        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape

        N, D = self.n_heads, C//self.n_heads  #N=num of heads D=attention dim

        #rearrange Q, K, V as (B,T,N,D)
        Q = self.Q(x).view(B, T, N, D).transpose(1, 2)
        K = self.K(x).view(B, T, N, D).transpose(1, 2)
        V = self.V(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = Q @ K.transpose(2, 3) / math.sqrt(D)

        #casul mask applied to weights
        weights = weights.masked_fill(self.mask[..., :T, :T]==0, float('-inf')) 

        #normalized weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        #attention (B, N, T, D)
        attention = self.atten_drop(normalized_weights @ V)

        #gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out
    

class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4*h_dim),
            nn.GELU(),
            nn.Linear(4*h_dim, h_dim),
            nn.Dropout(drop_p)
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        x = x + self.attention(x)
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x
    

    

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        #transformer block
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        #projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = nn.Linear(1, h_dim)
        self.embed_state = nn.Linear(state_dim, h_dim)

        self.embed_action = nn.Embedding(act_dim, h_dim)
        use_action_tanh = False

        self.predict_rtg = nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim) 
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )

    def forward(self, timesteps, states, actions, return_to_go):
        B, T, _ = states.shape

        time_embedding = self.embed_timestep(timesteps)

        state_embedding = self.embed_state(states) + time_embedding
        action_embedding = self.embed_action(actions) + time_embedding
        rtg_embedding = self.embed_rtg(return_to_go) + time_embedding

        #stack rtg, state and actions and reshape sequence as (r0, s0, a0, s1, a1, r1)
        h = torch.stack((rtg_embedding, state_embedding, action_embedding), dime=1).permute(0,2,1,3).reshape(B, 3*T, self.h_dim)

        h = self.embed_ln(h)

        h = self.transformer(h)

        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

        return_preds = self.predict_rtg(h[:, 2]) # given r, s, a
        state_preds = self.predict_state(h[:, 2]) # given r, s, a
        action_preds = self.predict_action(h[:, 1]) #given r, s

        return return_preds, state_preds, action_preds