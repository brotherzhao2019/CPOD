from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb
import math
import einops
import time
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import diffusor.utils as utils
import random

from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

class ARInvModel(nn.Module):
    # TODO: 需要检查这个函数的输出值范围，这个很重要！！！后面根据数据集再做这件事
    def __init__(self, hidden_dim, observation_dim, action_dim, low_act=-1.0, up_act=1.0):
        super(ARInvModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.action_embed_hid = 128
        self.out_lin = 128
        self.num_bins = 80

        self.up_act = up_act
        self.low_act = low_act
        self.bin_size = (self.up_act - self.low_act) / self.num_bins
        self.ce_loss = nn.CrossEntropyLoss()

        self.state_embed = nn.Sequential(
            nn.Linear(2 * self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lin_mod = nn.ModuleList([nn.Linear(i, self.out_lin) for i in range(1, self.action_dim)])
        self.act_mod = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, self.action_embed_hid), nn.ReLU(),
                                                    nn.Linear(self.action_embed_hid, self.num_bins))])

        for _ in range(1, self.action_dim):
            self.act_mod.append(
                nn.Sequential(nn.Linear(hidden_dim + self.out_lin, self.action_embed_hid), nn.ReLU(),
                              nn.Linear(self.action_embed_hid, self.num_bins)))

    def forward(self, comb_state, deterministic=False):
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        lp_0 = self.act_mod[0](state_d)
        l_0 = torch.distributions.Categorical(logits=lp_0).sample()

        if deterministic:
            a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        else:
            a_0 = torch.distributions.Uniform(self.low_act + l_0 * self.bin_size,
                                              self.low_act + (l_0 + 1) * self.bin_size).sample()

        a = [a_0.unsqueeze(1)]

        for i in range(1, self.action_dim):
            lp_i = self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](torch.cat(a, dim=1))], dim=1))
            l_i = torch.distributions.Categorical(logits=lp_i).sample()

            if deterministic:
                a_i = self.low_act + (l_i + 0.5) * self.bin_size
            else:
                a_i = torch.distributions.Uniform(self.low_act + l_i * self.bin_size,
                                                  self.low_act + (l_i + 1) * self.bin_size).sample()

            a.append(a_i.unsqueeze(1))

        return torch.cat(a, dim=1)
    
    def calc_loss(self, comb_state, action):
        eps = 1e-8
        action = torch.clamp(action, min=self.low_act + eps, max=self.up_act - eps)
        l_action = torch.div((action - self.low_act), self.bin_size, rounding_mode='floor').long()
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        loss = self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])

        for i in range(1, self.action_dim):
            loss += self.ce_loss(self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)),
                                     l_action[:, i])

        return loss/self.action_dim

class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, historical_horizon, observation_dim, 
                 action_dim, dist_temperature, n_timesteps=200, n_timesteps_finetune_start = 50,
                 gpt_backbone=True, regulizer_lambda = .5, 
                 loss_type='l1', clip_denoised=False, predict_epsilon=True, hidden_dim=256, 
                 loss_discount=1.0, loss_weights=None, returns_condition=False,
                 condition_guidance_w=0.1, ar_inv=False, train_only_inv=False, fix_inv=False):
        super().__init__()
        self.horizon = horizon
        self.historical_horizon = historical_horizon
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.gpt_backbone = gpt_backbone
        self.regulizer_lambda = regulizer_lambda
        #self.transition_dim = action_dim + observation_dim
        self.model = model
        self.ar_inv = ar_inv
        self.fix_inv = fix_inv
        self.train_only_inv = train_only_inv
        if self.ar_inv:                                         #TODO: 这里需要针对任务重新检查InDyn函数
            self.inv_model = ARInvModel(hidden_dim=hidden_dim, observation_dim=observation_dim, action_dim=action_dim)
        else:
            self.inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim)
            )
        self.return_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.dist_temperature = dist_temperature

        self.n_timesteps = int(n_timesteps)
        self.n_timesteps_finetune_start = n_timesteps_finetune_start
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses[loss_type](loss_weights)
    
    # TODO: 注意，对于GPTbackbone，horizon 仅仅代指预测轨迹的长度， 而对于Unet backbone，horrizon 长度包括了historical
    
    def get_loss_weights(self, discount, gpt_back_bone=True):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        if not self.gpt_backbone:
            discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        else:
            discounts = discount ** torch.arange(self.horizon - self.historical_horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        # Cause things are conditioned on t=0 for Unet backbone
        if not gpt_back_bone:
            loss_weights[0:self.historical_horizon, :] = 0
        
        return loss_weights
    
    #------------------------------------------ sampling ------------------------------------------#
    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
            这里的 x_t 对于GPTbackbone来说是合理的
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def q_posterior(self, x_start, x_t, t):
        '''
        根据 x_t 和 x_0 给出 x_(t-1) 的后验分布 
        '''
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, x_cond, t, returns=None):
        '''
        x:          batch x horizon x obs_dim
        x_cond:     batch x historical_horizon x obs_dim 
        t           batch
        '''
        if self.return_condition:
            assert RuntimeError()
        else:
            if self.gpt_backbone:
                epsilon = self.model(x, t, x_cond)
            else:
                epsilon = self.model(x, t)
        
        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        # 计算的是 逆向（去噪）过程的后验分布的 均值 和 方差
        return model_mean, posterior_variance, posterior_log_variance
    
    #@torch.no_grad()
    def p_sample(self, x, x_cond, t, returns=None):
        assert returns == None
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, x_cond=x_cond, t=t)
        noise = 0.5*torch.randn_like(x)
        # no noise when t ==0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond, returns=None, verbose=True, return_diffusion=False):
        '''
            x_cond: [batch x historical_horizon x obs_dim]
        '''
        assert returns == None
        device = self.betas.device                      #TODO: 这个device怎么回事？
        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)

        if not self.gpt_backbone:
            # apply conditions for Unet backbone
            x[:, 0: self.historical_horizon, :] = x_cond
        
        if return_diffusion: diffusion = []

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            with torch.no_grad():
                x = self.p_sample(x, x_cond, timesteps)

            if not self.gpt_backbone:
                x[:, 0: self.historical_horizon, :] = x_cond
            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)
        
        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
    
    @torch.no_grad()
    def conditional_sample(self, x_cond):
        '''
            x_cond: [batch x historical_horizon x obs_dim]
        '''
        assert self.predict_epsilon
        batch_size = x_cond.shape[0]
        if not self.gpt_backbone:
            shape = (batch_size, self.horizon, self.observation_dim)
        else:
            shape = (batch_size, self.horizon - self.historical_horizon, self.observation_dim)

        return self.p_sample_loop(shape, x_cond)

    #------------------------------------------ training ------------------------------------------#
    def q_sample(self, x_start, t, noise=None):
        '''
            从初始 x_0 预测任意时间 t 的前向扩散结果 x_t
        '''
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample
    
    def p_losses(self, x_start, x_cond, t):
        '''
            x_start: [batch x horizon x obs_dim]
            x_cond: [batch x historical_horizon x obs_dim]
            如果使用 unet backbone， x_start 中也包括了historical observation， 如果使用 gpt backbone， x_start 中仅包括 预测 traj
        '''
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t=t, noise=noise)                      # batch x 15 x obs_dim for gpt backbone

        if not self.gpt_backbone:
            x_noisy[:, 0: self.historical_horizon, :] = x_cond
            model_out = self.model(x_noisy, t)
        else:
            model_out = self.model(x_noisy, t, x_cond)                          # batch x 15 x obs_dim for gpt backbone
        
        # 如果选用了 unet backbone， 在计算loss时已经将historical observation mask掉了，这是依靠 loss_weight 实现的
        if self.predict_epsilon:
            loss, info = self.loss_fn(model_out, noise)
        else:
            if not self.gpt_backbone:
                model_out[:, 0: self.historical_horizon, :] = x_cond
            loss, info = self.loss_fn(model_out, x_start)
        
        return loss, info
    
    def inv_action_loss(self, x, cond):
        '''
            x: [batch x horizon x (action_dim + obs_dim)]
        '''
        x_t = x[:, :-1, self.action_dim:]
        a_t = x[:, :-1, : self.action_dim]
        x_t_p = x[:, 1:, self.action_dim:]
        x_comb_t = torch.cat([x_t, x_t_p], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
        a_t = a_t.reshape(-1, self.action_dim)
        if self.ar_inv:
            loss = self.inv_model.calc_loss(x_comb_t, a_t)
            info = {'inv_dynamic_loss': loss.cpu().detach().numpy()}
        else:
            pred_a_t = self.inv_model(x_comb_t)
            loss = F.mse_loss(pred_a_t, a_t)
            info = {'inv_dynamic_loss': loss.cpu().detach().numpy()}
        return loss, info
    
    def loss(self, x, cond):
        '''
            x: [batch x horizon x (action_dim + obs_dim)]           
            注意： 对于gpt_backbone，输入数据的 轨迹 长度应该是 horizon + historical_horizon, 
            对于unet_backbone，输入的轨迹长度应该是horizon，horizon中包含了historical_horizon
        '''
        assert self.predict_epsilon
        
        if self.train_only_inv:
            loss, info = self.inv_action_loss(x, cond)
        else:
            inv_action_loss, info = self.inv_action_loss(x, cond)
            if self.fix_inv:
                inv_action_loss.detach()
            batch_size = x.shape[0]
            t = torch.randint(0, self.n_timesteps, (batch_size, ), device=x.device).long()
            if self.gpt_backbone:
                x_state_trj = x[:, self.historical_horizon:, self.action_dim:]          # batch x (horizon - 1) x obs_dim
                x_state_cond = x[:, 0:self.historical_horizon, self.action_dim:]        # batch x 1 x obs_dim
                diffuse_loss, _ = self.p_losses(x_state_trj, x_state_cond, t)
            else:
                x_state_trj = x[:, :, self.action_dim:]
                x_state_cond = x[:, 0:self.historical_horizon, self.action_dim:]
                diffuse_loss, _ = self.p_losses(x_state_trj, x_state_cond, t)
            
            loss = (diffuse_loss + inv_action_loss) * 0.5
            info['diffusion_loss'] = diffuse_loss.cpu().detach().numpy()
        info['total_loss'] = loss.cpu().detach().numpy()
        
        return loss, info
    
    def forward(self, x_cond):
        '''
        x_cond: [batch, historical_horizon, obs_dim]
        '''
        assert self.predict_epsilon
        assert x_cond.shape[1] == self.historical_horizon and x_cond.shape[2] == self.observation_dim
        return self.conditional_sample(x_cond=x_cond)
    
    #------------------------------------------ Finetunning ------------------------------------------#
    def finetue_DPPO_loss(self, x_p, x_n, scores, dist_temperature=None):
        '''
            x_p: positive trajectory            [batch, horizon, action_dim + observation_dim]
            x_n: negative trajectory            [batch, horizon, action_dim + observation_dim]
            scores:                             [batch]   0 表示x_p好
            先按照 gpt_backbone 实现ReFL
        '''
        assert self.predict_epsilon
        # assert self.gpt_backbone
        device = self.betas.device

        if self.gpt_backbone:
            x_state_trj_1 = x_p[:, self.historical_horizon:, self.action_dim:]
            x_state_trj_2 = x_n[:, self.historical_horizon:, self.action_dim:]
            x_state_cond_1 = x_p[:, 0: self.historical_horizon, self.action_dim:]
            x_state_cond_2 = x_n[:, 0: self.historical_horizon, self.action_dim:]
        else:
            x_state_trj_1 = x_p[:, :, self.action_dim:]
            x_state_trj_2 = x_n[:, :, self.action_dim:]
            x_state_cond_1 = x_p[:, 0: self.historical_horizon, self.action_dim:]
            x_state_cond_2 = x_n[:, 0: self.historical_horizon, self.action_dim:]

        t1 = random.randint(1, self.n_timesteps_finetune_start)
        t2 = random.randint(1, self.n_timesteps_finetune_start)

        batch_size = x_p.shape[0]
        shape = x_state_trj_1.shape
        # 先做正常的去噪过程 trj_p
        x1 = 0.5*torch.randn(shape, device=device)
        if not self.gpt_backbone:
            x1[:, 0:self.historical_horizon, :] = x_state_cond_1
        
        for i in reversed(range(t1, self.n_timesteps)):
            timesteps_1 = torch.full((batch_size,), i, device=device)
            with torch.no_grad():
                x1 = self.p_sample(x1, x_state_cond_1, timesteps_1)
                if not self.gpt_backbone:
                    x1[:, 0:self.historical_horizon, :] = x_state_cond_1

        timesteps_1 = torch.full((batch_size,), t1 - 1, device=device)
        # 再加一步微调
        if self.gpt_backbone:
            epsilon1 = self.model(x1, timesteps_1, x_state_cond_1)
        else:
            epsilon1 = self.model(x1, timesteps_1)
        
        timesteps_1 = timesteps_1.detach().to(torch.int64)
        x1_0 = self.predict_start_from_noise(x1, t=timesteps_1, noise=epsilon1)

        x2 = 0.5*torch.randn(shape, device=device)
        if not self.gpt_backbone:
            x2[:, 0:self.historical_horizon, :] = x_state_cond_2
        
        for i in reversed(range(t2, self.n_timesteps)):
            timesteps_2 = torch.full((batch_size,), i, device=device)
            with torch.no_grad():
                x2 = self.p_sample(x2, x_state_cond_2, timesteps_2)
                if not self.gpt_backbone:
                    x2[:, 0:self.historical_horizon, :] = x_state_cond_2
        
        timesteps_2 = torch.full((batch_size,), t2 - 1, device=device)
        if self.gpt_backbone:
            epsilon2 = self.model(x2, timesteps_2, x_state_cond_2)
        else:
            epsilon2 = self.model(x2, timesteps_2)
        
        timesteps_2 = timesteps_2.detach().to(torch.int64)
        x2_0 = self.predict_start_from_noise(x2, t=timesteps_2, noise=epsilon2)

        sum_mask = torch.ones(shape, dtype=torch.long, device=device)
        if not self.gpt_backbone:
            sum_mask[:, 0:self.historical_horizon, :] = 0

        x_1_distance = torch.sum((x1_0 - x_state_trj_1) ^ 2 * sum_mask, dim=-1)             # [batch, horizon]
        x_1_distance = torch.sum(x_1_distance, dim=-1) / self.horizon / self.dist_temperature            # batch

        x_2_distance = torch.sum((x2_0 - x_state_trj_2) ^ 2 * sum_mask, dim=-1)
        x_2_distance = torch.sum(x_2_distance, dim=-1) / self.horizon / self.dist_temperature

        
        x_distance = torch.cat((x_1_distance, x_2_distance), dim=1)                         # [batch, 2]

        score_1_over_2 = torch.log(torch.exp(-x_distance[:, 0]) / (torch.exp(-x_distance[:, 0]) 
                                                                   + torch.exp(- self.regulizer_lambda * x_distance[:, 1])))  #[batch,]
        score_2_over_1 = torch.log(torch.exp(-x_distance[:, 1]) / (torch.exp(-x_distance[:, 1]) 
                                                                   + torch.exp(- self.regulizer_lambda * x_distance[:, 0])))

        loss = - torch.mean((1 - scores) * score_1_over_2 + scores * scores * score_2_over_1)
        return loss
