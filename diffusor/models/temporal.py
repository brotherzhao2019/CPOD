import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pickle
import pdb
import transformers
from transformers import TransfoXLModel, TransfoXLConfig
from .GPT2 import GPT2Model
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops.layers.torch import Rearrange
from einops import rearrange

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
    AttentionBlock
)

from torch.distributions import Bernoulli

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine = True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class GlobalMixing(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)
    

class GPT2Backbone(nn.Module):
    "Diffusor backbone with a Transformer backbone"
    def __init__(
            self,
            horizon,
            historical_obs_dim,
            transition_dim,                                 # state dimension / state + action dimension
            dim = 128,
            hidden_size = 256,
            historical_horizon = 2,
            act_dim = None,            
    ):
        super().__init__()
        self.dim = dim
        self.horizon = horizon
        self.historical_horizon = historical_horizon
        self.hidden_size = hidden_size
        self.transition_dim = transition_dim
        self.historical_obs_dim = historical_obs_dim
        #self.state_dim = transition_dim                     # TODO: obs dim?
        self.action_dim = act_dim
        config = transformers.GPT2Config(
            vocab_size= 1,
            n_embd=hidden_size,
            n_layer=4,
            n_head=2,
            n_inner=4 * 256,
            activation_function='mish',
            n_positions=1024,
            n_ctx=1023,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(2* self.dim),
            nn.Linear(2*self.dim, self.dim * 4),
            nn.Mish(),
            nn.Linear(self.dim * 4, self.hidden_size),
        )
        # no return embedding

        self.embed_obs = nn.Sequential(
            nn.LayerNorm(self.historical_obs_dim),
            nn.Linear(self.historical_obs_dim, 2 * self.hidden_size),  
            nn.Mish(),
            nn.Linear(self.hidden_size * 2, 4 * self.hidden_size),
            nn.Mish(),
            nn.Linear(4 * self.hidden_size, self.hidden_size),
        )

        #self.mask_dist = Bernoulli(probs=0.8)                                           
        self.transformer = GPT2Model(config)
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.embed_transition = nn.Linear(self.transition_dim, self.hidden_size)
        self.predict_transition = torch.nn.Linear(self.hidden_size, self.transition_dim)
        self.position_emb = nn.Parameter(torch.zeros(1, 1 + self.historical_horizon + self.horizon, self.hidden_size))
    
    def forward(self, x, time, x_condition, force=False, attention_mask=None):
        '''
        x: [ batch x horizon x transition]
        x_condition: [ batch x historical_horizon x obs_dim]                         historical observations as condtions
        因为我们暂时不采用条件生成，所以force变量在我们的模块里面暂时没有用
        '''
        t = self.time_mlp(time).unsqueeze(1)                        # batch x 1 x hidden_size
        obs_embedding = self.embed_obs(x_condition)                 # batch x 2 x hidden_size
        trans_embedding = self.embed_transition(x)                  # batch x horizon x hidden_size


        batch_size, seq_length = x.shape[0], x.shape[1]             # seq_length = 15, horizon = 15
        if not attention_mask:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=x.device)
        addition_attention_mask = torch.ones((batch_size, t.shape[1] + obs_embedding.shape[1]), dtype=torch.long, device=x.device)
        stacked_attention_mask = torch.cat((addition_attention_mask, attention_mask), dim=1)        # batch x (horizon + historical_horizon + 1)

        stacked_inputs = torch.cat((t, obs_embedding, trans_embedding), dim=1)  # batch x (historical_horizon + horizon + 1) x hidden_size

        stacked_inputs = t * stacked_inputs + self.position_emb
        stacked_inputs = self.embed_ln(stacked_inputs)                          # batch x (historical_horizon + horizon + 1) x hidden_size
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs, attention_mask=stacked_attention_mask)
        x = transformer_outputs['last_hidden_state']
        trans_preds = self.predict_transition(x[:, -seq_length:, :])
        return trans_preds
    

class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        returns_condition=False,
        condition_dropout=0.1,
        calc_energy=False,
        kernel_size=5,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        if calc_energy:
            mish = False
            act_fn = nn.SiLU()
        else:
            mish = True
            act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                        nn.Linear(1, dim),
                        act_fn,
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            embed_dim = 2*dim
        else:
            embed_dim = dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time, cond=None, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        if self.calc_energy:
            x_inp = x

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # import pdb; pdb.set_trace()

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        if self.calc_energy:
            # Energy function
            energy = ((x - x_inp)**2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
            return grad[0]
        else:
            return x

    def get_pred(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x

class MLPResNetBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim                        # 256
        self.dropout_rate = dropout_rate
        self.drop_out = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.mlp_1 = nn.Linear(self.hidden_dim, 4 * self.hidden_dim)
        self.mlp_2 = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.act = nn.Mish()
    
    def forward(self, x):
        # x: batchsize x horizon x hidden_dim
        residual = x
        x = self.drop_out(x)
        x = self.layer_norm(x)
        x = self.mlp_1(x)
        x = self.act(x)
        x = self.mlp_2(x)

        return residual + x

class MLPResNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks, hidden_dim, dropout_rate):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.in_layer = nn.Linear(self.in_dim, self.hidden_dim)
        self.out_layer = nn.Linear(self.hidden_dim, self.out_dim)
        self.resnet_block_array = nn.Sequential(*[MLPResNetBlock(self.hidden_dim, self.dropout_rate) for _ in range(self.num_blocks)])
        
        self.act = nn.ReLU()
    
    def forward(self, x):
        # x: batchsize x horizon x in_dim (in_dim = action_dim + obs_dim + t_embed_dim)
        x = self.in_layer(x)            #batchsize x horizon x hidden_dim
        x = self.resnet_block_array(x)
        # for _ in range(self.num_blocks):
        #     x = MLPResNetBlock(self.hidden_dim, self.dropout_rate, device=x.device)(x)
        x = self.act(x)
        x = self.out_layer(x)
        return x

class ResNetBackbone(nn.Module):
    def __init__(self, obs_dim, act_dim, dropout_rate=0.1, num_block=2, hidden_dim=256, time_emb_dim=100):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.num_block = num_block
        self.dropout_rate = dropout_rate
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(2 * self.time_emb_dim),
            nn.Linear(2 * self.time_emb_dim, self.time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(self.time_emb_dim * 4, self.time_emb_dim))
        self.obs_act_emb = nn.Linear(self.obs_dim + self.act_dim, self.hidden_dim - self.time_emb_dim)
        self.mlp_resnet = MLPResNet(self.hidden_dim, self.act_dim, self.num_block, self.hidden_dim, self.dropout_rate)
    
    def forward(self, obs, act, t):
        # obs: batch x horizon x obs_dim
        # act: batch x horizon x act_dim
        # t: batch x horizon
        t_emb = self.time_mlp(t)                # batch x horizon x time_emb_dim
        obs_act = torch.cat([obs, act], dim=-1)
        in_feature = torch.cat([self.obs_act_emb(obs_act), t_emb], dim=-1)
        return self.mlp_resnet(in_feature)      # batch x horizon x act_dim


class PreferenceTransformer(nn.Module):
    '''
    Transformer backbone for preference medel
    '''
    def __init__(
            self,
            horizon,
            transition_dim, 
            dim = 128,
            hidden_size = 256,
            act_dim = None,
    ):
        super().__init__()
        self.dim = dim
        self.transition_dim = transition_dim
        self.hidden_size = hidden_size
        self.horizon = horizon
        
        config = transformers.GPT2Config(
            vocab_size= 1,
            n_embd=hidden_size,
            n_layer=4,
            n_head=2,
            n_inner=4 * 256,
            activation_function='mish',
            n_positions=1024,
            n_ctx=1023,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(2* self.dim),
            nn.Linear(2*self.dim, self.dim * 4),
            nn.Mish(),
            nn.Linear(self.dim * 4, self.hidden_size),
        )

        self.transformer = GPT2Model(config)
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.embed_transition = nn.Linear(self.transition_dim, self.hidden_size)
        self.position_emb = nn.Parameter(torch.zeros(1, self.horizon, self.hidden_size))
        self.logits_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),             # batch x horizon x hidden_size
            nn.Mish(),
            nn.Linear(self.hidden_size, self.hidden_size/2),           # batch x horizon x hidden_size / 2
            nn.Mish(),
            nn.Linear(self.hidden_size/2, 1),                           
            nn.Flatten(start_dim=1),                                   # batch x horizon
            nn.Mish(),
            nn.Linear(self.horizon, 1),                                # batch x 1
        )
    
    def forward(self, x):
        '''
        x: [batch x horizon x transition_dim]
        '''
        trans_embedding = self.embed_transition(x)
        batch_size, seq_length = x.shape[0], x.shape[1]
        
        atten_mask = torch.ones((batch_size, self.horizon), dtype=torch.long, device=x.device)

        inputs_ln = self.embed_ln(trans_embedding)

        transformer_outputs = self.transformer(inputs_embeds=inputs_ln, atten_mask=atten_mask)
        x = transformer_outputs['last_hidden_state']
        return torch.squeeze(self.logits_mlp(x), dim=1)


        

        

