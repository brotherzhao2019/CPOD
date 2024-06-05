import os
import copy
import numpy as np
import torch
# import einops
import pdb
import random
from .arrays import batch_to_device, to_np, to_device, apply_dict, to_torch
from .timer import Timer
#import metaworld
import time
import gym
from torch.optim import lr_scheduler
# import d4rl
import statistics
DTYPE = torch.float
from collections import namedtuple
from diffusor.models import ActionDiffusion
#from ml_logger import logger
#import diffuser.utils as utils
DTBatch = namedtuple('DTBatch', 'actions rtg observations timestep mask')
DEVICE = 'cuda'
from diffusor.datasets.pref_sequence import WLBatch
from torch.utils.tensorboard import SummaryWriter
from diffusor.envs import MetaWorldSawyerEnv

def cycle(dl):
    while True:
        for data in dl:
            yield data

def to_torch(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
        # import pdb; pdb.set_trace()
    return torch.tensor(x, dtype=dtype, device=device)


class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def add_pref_label(batch_1, batch_2):
    batch_1_trj, batch_1_return = batch_1[0], batch_1[2]
    batch_2_trj, batch_2_return = batch_2[0], batch_2[2]
    bsz = min(batch_2_trj.shape[0], batch_1_trj.shape[0])
    traj_w, traj_l = torch.zeros_like(batch_1_trj), torch.zeros_like(batch_2_trj)

    for ind in range(bsz):
        if batch_1_return[ind][0] > batch_2_return[ind][0]:
            traj_w[ind, :, :] = batch_1_trj[ind, :, :]
            traj_l[ind, :, :] = batch_2_trj[ind, :, :]
        else:
            traj_w[ind, :, :] = batch_2_trj[ind, :, :]
            traj_l[ind, :, :] = batch_1_trj[ind, :, :]
    return WLBatch(traj_w, traj_l)


class Trainer(object):
    def __init__(self,
                diffusor,
                dataset,
                dataset_p,
                renderer,
                dataset_inv,
                dataset_val = None,
                val_freq = 0,
                #ref_model = None,           
                action_plan = False,
                finetune_coef=[0.5, 0.5],
                init_model_dir = None,
                ema_decay=0.995,
                train_batch_size=32,
                train_lr=2e-5,
                gradient_accumulate_every=2,
                step_start_ema=2000,
                update_ema_every=10,
                log_freq=100,
                skip_steps = 400 * 1000,
                sample_freq=1000,
                save_freq=1000,
                label_freq=100000,
                save_parallel=False,
                with_ref_model = True,
                no_label_bc=False,
                results_folder='./results',
                pref_model = None,
                #n_reference=8,
                bucket=None,
                is_gpt_bacbone=True,
                train_device='cuda',
                horizon=16,
                ismetaworld = False,
                env_name = '',
                bc_update_period = None,
                bc_update_mulplier = None,
                lr_decay_beta=0,
                weight_decay=0.0,
                total_finetune_steps=10000,
                lr_schedule='exp'
                ):
        super().__init__()
        self.model = diffusor
        self.with_ref_model = with_ref_model
        self.ref_model = copy.deepcopy(diffusor) if self.with_ref_model else None                  # ref_model is the model in diffusor. (e.g. gpt_backbone)
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.device = train_device
        self.horizon = horizon
        self.train_lr = train_lr
        self.finetune_coef = finetune_coef
        self.train_batch_size = train_batch_size
        self.is_gpt_backbone = is_gpt_bacbone
        self.action_plan = action_plan
        self.no_label_bc = no_label_bc
        self.skip_steps = skip_steps
        self.pref_model = pref_model
        if self.pref_model:
            assert self.pref_model.query_len == self.horizon

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel
        self.save_root_dir = results_folder
        self.log_dir = os.path.join(self.save_root_dir, 'training_log')
        self.fine_log_dir = os.path.join(self.save_root_dir, 'finetune_log')
        
        self.weight_dir = os.path.join(self.save_root_dir, 'weights')
        self.save_avg_return_record = 0
        self.ismetaworld = ismetaworld
        if self.ismetaworld:
            assert env_name
            self.env_name = env_name

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
        if not os.path.exists(self.fine_log_dir):
            os.makedirs(self.fine_log_dir)

        if dataset_inv:
            
            self.log_inv_dir = os.path.join(self.save_root_dir, 'training_inv_log')
            if not os.path.exists(self.log_inv_dir):
                os.makedirs(self.log_inv_dir)
            self.writer_inv = SummaryWriter(self.log_inv_dir)

        #self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        # Initiate two dataset here...
        self.dataset = dataset
        self.dataset_p = dataset_p
        self.dataset_inv = dataset_inv
        self.dataset_val = dataset_val
        self.val_freq = val_freq
        
        if self.dataset_val:
            self.dataloader_val = cycle(torch.utils.data.DataLoader(
                self.dataset_val, batch_size=1000, num_workers=0, shuffle=True, pin_memory=True))
        
        if self.dataset_inv:
            self.dataloader_inv = cycle(torch.utils.data.DataLoader(
                self.dataset_inv, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True))
        
        if self.dataset:
            self.dataloader = cycle(torch.utils.data.DataLoader(
                self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True))
        
        if self.dataset_p:
            self.dataloader_p = cycle(torch.utils.data.DataLoader(
                self.dataset_p, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True))
        
        self.observation_dim = self.dataset.observation_dim if self.dataset else self.dataset_p.observation_dim
        self.action_dim = self.dataset.action_dim if self.dataset else self.dataset_p.action_dim

        
        # Declare optimizer here...
        self.optimizer = torch.optim.Adam(diffusor.parameters(), lr=train_lr, weight_decay=weight_decay)
        self.writer = SummaryWriter(self.log_dir)
        self.writer_finetune = SummaryWriter(self.fine_log_dir)

        if init_model_dir:
            self.load(init_model_dir)

        self.reset_parameters()
        self.step = 0
        #self.fintune_step = 0
        self.bc_update_period = bc_update_period
        self.bc_update_mulplier = bc_update_mulplier
        self.lr_schedule = lr_schedule

        if self.lr_schedule == 'cosine':
            self.lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=total_finetune_steps)
        if self.lr_schedule == 'exp':
            self.lr_decay_beta = lr_decay_beta

        # self.num_eval = 200
        # self.env_list = [MetaWorldSawyerEnv(self.env_name, randomize_hand=False) for _ in range(self.num_eval)]
    
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def load_inv_parameters(self, model_dir):
        assert not self.action_plan
        data = torch.load(model_dir, map_location=self.device)
        self.model.inv_model.load_state_dict(data['inv_model'])
    
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)
    
    def train(self, n_train_steps):
        # This function is used to train a Diffusor policy on the offline dataset
        timer = Timer()
        self.model.train_only_inv = False
        self.model.model.train()
        for _ in range(n_train_steps):
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.writer.add_scalar('Loss/total_loss', infos['total_loss'], global_step=self.step)
            # if not self.action_plan:
            #     self.writer.add_scalar('Loss/a0_loss', infos['inv_dynamic_loss'], global_step=self.step)
            if 'diffusion_loss' in infos:
                self.writer.add_scalar('Loss/diffusion_loss', infos['diffusion_loss'], global_step=self.step)

            if self.step % self.update_ema_every == 0:
                self.step_ema()
                
            # TODO: save and evaluate model
            
            if self.step % self.log_freq == 0 and self.step > self.skip_steps:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)
                if not isinstance(self.model, ActionDiffusion):
                    avg_return, _, avg_norm_score, std_norm_score = self.evaluate()
                else:
                    if self.ismetaworld:
                        avg_return, _, avg_norm_score, std_norm_score, success_rate = self.evaluate_action_diffusion_mw()
                    else:
                        avg_return, _, avg_norm_score, std_norm_score = self.evaluate_action_diffusion()
                self.writer.add_scalar('Evaluation/Avg_return', avg_return, global_step=self.step)
                self.writer.add_scalar('Evaluation/Avg_norm_score', avg_norm_score, global_step=self.step)
                self.writer.add_scalar('Evaluation/std_norm_score', std_norm_score, global_step=self.step)
                if self.ismetaworld:
                    self.writer.add_scalar('Evaluation/success_rate', success_rate, global_step=self.step)
                if avg_return > self.save_avg_return_record:
                    self.save_avg_return_record = avg_return
                    self.save(0)
            
            self.step += 1
    
    def finetune_dpo(self, n_train_steps):
        timer = Timer()
        self.model.train_only_inv = False
        self.model.model.train()
        print('In finetune function...')

        if self.step == 0:
            if not isinstance(self.model, ActionDiffusion):
                avg_return, _, avg_norm_score, std_norm_score = self.evaluate()
            else:
                if self.ismetaworld:
                    avg_return, _, avg_norm_score, std_norm_score, success_rate = self.evaluate_action_diffusion_mw()
                else:
                    avg_return, _, avg_norm_score, std_norm_score = self.evaluate_action_diffusion()
            if self.ismetaworld:
                self.writer_finetune.add_scalar('Evaluation/success_rate', success_rate, global_step=self.step)
            self.writer_finetune.add_scalar('Evaluation/Avg_return', avg_return, global_step=self.step)
            self.writer_finetune.add_scalar('Evaluation/Avg_norm_score', avg_norm_score, global_step=self.step)
            self.writer_finetune.add_scalar('Evaluation/std_norm_score', std_norm_score, global_step=self.step)
            self.save_avg_return_record = avg_return

        for _ in range(n_train_steps):
            if self.bc_update_mulplier and self.bc_update_period:
                if self.step > 0  and self.step % self.bc_update_period == 0:
                    self.model.bc_coef = self.model.bc_coef * self.bc_update_mulplier
            acc = 0        
            for _ in range(self.gradient_accumulate_every):
                if self.no_label_bc:
                    batch_pref = next(self.dataloader_p)
                    batch_pref = batch_to_device(batch_pref, device=self.device)
                    traj_w, traj_l = batch_pref
                    batch_nolabel = next(self.dataloader)
                    batch_nolabel = batch_to_device(batch_nolabel, device=self.device)
                    batch_nolabel = batch_nolabel[0]
                    loss, info = self.model.finetune_loss_with_nolabel_data(traj_w, traj_l, batch_nolabel, self.ref_model)
                else:
                    batch = next(self.dataloader_p)
                    batch = batch_to_device(batch, device=self.device)
                    traj_w, traj_l = batch                  #batch['traj_w'], batch['traj_l']
                    loss, info = self.model.finetune_loss(traj_w, traj_l, self.ref_model)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                acc += info['avg_implict_acc'] / self.gradient_accumulate_every
            
            self.optimizer.step()
            # lr decay
            if self.lr_schedule == 'cosine':
                self.lr_scheduler.step()
            if self.lr_schedule == 'exp':
                acc = torch.clip(acc-0.5, min=0)
                new_learning_rate = self.train_lr * torch.exp(-self.lr_decay_beta * acc)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_learning_rate.item()

            self.optimizer.zero_grad()
            
            self.writer_finetune.add_scalar('Training/lr', self.optimizer.param_groups[0]['lr'], self.step)
            self.writer_finetune.add_scalar('Training/bc_coef', self.model.bc_coef, self.step)

            self.writer_finetune.add_scalar('Training/loss', loss.cpu().detach().numpy(), self.step)
            for key in info.keys():
                self.writer_finetune.add_scalar('Training/'+key, info[key], self.step)
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()
                
            if self.val_freq > 0:
                if self.step % self.val_freq == 0 and self.step > self.skip_steps:
                    print('now is validating')
                    batch = next(self.dataloader_val)
                    batch = batch_to_device(batch, device=self.device)
                    traj_w, traj_l = batch
                    acc = self.model.validate_acc(traj_w, traj_l, self.ref_model)
                    self.writer_finetune.add_scalar('Training/ val_implict_acc', acc, global_step=self.step)
                
            if self.step % self.log_freq == 0 and self.step > self.skip_steps:
                print(f'{self.step}: {loss:8.4f} | ------ | t: {timer():8.4f}', flush=True)
                if not isinstance(self.model, ActionDiffusion):
                    avg_return, _, avg_norm_score, std_norm_score = self.evaluate()
                else:
                    if self.ismetaworld:
                        avg_return, _, avg_norm_score, std_norm_score, success_rate = self.evaluate_action_diffusion_mw()
                    else:
                        avg_return, _, avg_norm_score, std_norm_score = self.evaluate_action_diffusion()
                if self.ismetaworld:
                    self.writer_finetune.add_scalar('Evaluation/success_rate', success_rate, global_step=self.step)
                self.writer_finetune.add_scalar('Evaluation/Avg_return', avg_return, global_step=self.step)
                self.writer_finetune.add_scalar('Evaluation/Avg_norm_score', avg_norm_score, global_step=self.step)
                self.writer_finetune.add_scalar('Evaluation/std_norm_score', std_norm_score, global_step=self.step)
                if avg_return > self.save_avg_return_record:
                    self.save_avg_return_record = avg_return
                    self.save(0, False, True)
                # self.save(self.step, False, True)
            
            self.step += 1
    # 需要修改 dataloader_p, 1) no padding, 2)need return = True
    def add_pref_label(self, batch_1, batch_2):
        batch_1_trj, batch_1_return = batch_1[0], batch_1[2]
        batch_2_trj, batch_2_return = batch_2[0], batch_2[2]
        bsz = min(batch_2_trj.shape[0], batch_1_trj.shape[0])
        traj_w, traj_l = torch.zeros_like(batch_1_trj), torch.zeros_like(batch_2_trj)
        log = {}
        
        if self.pref_model:
            batch_1_act = batch_1_trj[:, :, 0: self.action_dim]
            batch_1_obs = batch_1_trj[:, :, self.action_dim:]
            batch_2_obs = batch_2_trj[:, :, self.action_dim:]
            batch_2_act = batch_2_trj[:, :, 0: self.action_dim]
            time_step = np.tile(np.arange(1, self.horizon + 1).astype(np.int32), (bsz, 1))
            batch_1_act = to_torch(batch_1_act, dtype=torch.float32, device=self.device)
            batch_1_obs = to_torch(batch_1_obs, dtype=torch.float32, device=self.device)
            batch_2_act = to_torch(batch_2_act, dtype=torch.float32, device=self.device)
            batch_2_obs = to_torch(batch_2_obs, dtype=torch.float32, device=self.device)
            time_step = to_torch(time_step, dtype=torch.int32, device=self.device)
            predict_result = self.pref_model.predict(batch_1_obs, batch_1_act, time_step, batch_2_obs, batch_2_act, time_step)

            count = 0
            for ind in range(bsz):
                if predict_result[ind] == 0:
                    traj_w[ind, :, :] = batch_1_trj[ind, :, :]
                    traj_l[ind, :, :] = batch_2_trj[ind, :, :]
                    if batch_1_return[ind][0] > batch_2_return[ind][0]:
                        count += 1
                else:
                    traj_w[ind, :, :] = batch_2_trj[ind, :, :]
                    traj_l[ind, :, :] = batch_1_trj[ind, :, :]
                    if batch_1_return[ind][0] <= batch_2_return[ind][0]:
                        count += 1
            
            log['pref_acc'] = count / bsz
                    
            
        else:   
            for ind in range(bsz):
                if batch_1_return[ind][0] > batch_2_return[ind][0]:
                    traj_w[ind, :, :] = batch_1_trj[ind, :, :]
                    traj_l[ind, :, :] = batch_2_trj[ind, :, :]
                else:
                    traj_w[ind, :, :] = batch_2_trj[ind, :, :]
                    traj_l[ind, :, :] = batch_1_trj[ind, :, :]
        return WLBatch(traj_w, traj_l), log

    def finetune_dpo_extend(self, n_train_steps):
        timer = Timer()
        self.model.train_only_inv = False
        self.model.model.train()
        if self.step == 0:
            if not isinstance(self.model, ActionDiffusion):
                avg_return, _, avg_norm_score, std_norm_score = self.evaluate()
            else:
                if self.ismetaworld:
                    avg_return, _, avg_norm_score, std_norm_score, success_rate = self.evaluate_action_diffusion_mw()
                else:
                    avg_return, _, avg_norm_score, std_norm_score = self.evaluate_action_diffusion()
            if self.ismetaworld:
                self.writer_finetune.add_scalar('Evaluation/success_rate', success_rate, global_step=self.step)
            self.writer_finetune.add_scalar('Evaluation/Avg_return', avg_return, global_step=self.step)
            self.writer_finetune.add_scalar('Evaluation/Avg_norm_score', avg_norm_score, global_step=self.step)
            self.writer_finetune.add_scalar('Evaluation/std_norm_score', std_norm_score, global_step=self.step)
            self.save_avg_return_record = avg_return
        for _ in range(n_train_steps):
            if self.bc_update_mulplier and self.bc_update_period:
                if self.step > 0  and self.step % self.bc_update_period == 0:
                    self.model.bc_coef = self.model.bc_coef * self.bc_update_mulplier
            for _ in range(self.gradient_accumulate_every):
                while True:
                    batch_1 = next(self.dataloader_p)
                    batch_2 = next(self.dataloader_p)
                    if batch_1[0].shape[0] != batch_2[0].shape[0]:
                        continue
                    #batch = add_pref_label(batch_1, batch_2)
                    batch, log_pref = self.add_pref_label(batch_1, batch_2)
                    batch = batch_to_device(batch, device=self.device)
                    traj_w, traj_l = batch                                      #batch['traj_w'], batch['traj_l']
                    break
                loss, info = self.model.finetune_loss(traj_w, traj_l, self.ref_model)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.writer_finetune.add_scalar('Training/loss', loss.cpu().detach().numpy(), self.step)
            for key in info.keys():
                self.writer_finetune.add_scalar('Training/'+key, info[key], self.step)
            
            for key in log_pref.keys():
                self.writer_finetune.add_scalar('Training/'+key, log_pref[key], self.step)
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            
            if self.step % self.log_freq == 0:
                print(f'{self.step}: {loss:8.4f} | ------ | t: {timer():8.4f}', flush=True)
                if not isinstance(self.model, ActionDiffusion):
                    avg_return, _, avg_norm_score, std_norm_score = self.evaluate()
                else:
                    if self.ismetaworld:
                        avg_return, _, avg_norm_score, std_norm_score, success_rate = self.evaluate_action_diffusion_mw()
                    else:
                        avg_return, _, avg_norm_score, std_norm_score = self.evaluate_action_diffusion()
                if self.ismetaworld:
                    self.writer_finetune.add_scalar('Evaluation/success_rate', success_rate, global_step=self.step)
                self.writer_finetune.add_scalar('Evaluation/Avg_return', avg_return, global_step=self.step)
                self.writer_finetune.add_scalar('Evaluation/Avg_norm_score', avg_norm_score, global_step=self.step)
                self.writer_finetune.add_scalar('Evaluation/std_norm_score', std_norm_score, global_step=self.step)
                if avg_return > self.save_avg_return_record:
                    self.save_avg_return_record = avg_return
                    self.save(0, False, True)
            
            self.step += 1

    def finetune(self, n_train_steps):
        timer = Timer()
        self.model.train_only_inv = False
        self.model.model.train()
        for _ in range(n_train_steps):
            for _ in range(self.gradient_accumulate_every):
                batch_1 = next(self.dataloader)
                batch_2 = next(self.dataloader_p)
                batch_1 = batch_to_device(batch_1, device=self.device)
                batch_2 = batch_to_device(batch_2, device=self.device)
                loss_unlabel, info_unlabel = self.model.loss(*batch_1)
                loss_pref, info_pref = self.model.finetune_loss_v2(*batch_2)
                loss_sum = self.finetune_coef[0] * loss_unlabel + self.finetune_coef[1] * loss_pref
                loss_sum.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.writer_finetune.add_scalar('Loss/total_loss', loss_sum.cpu().detach().numpy(), self.step)
            self.writer_finetune.add_scalar('Loss/unlabel_loss', loss_unlabel.cpu().detach().numpy(), self.step)
            self.writer_finetune.add_scalar('Loss/prefer_loss', loss_pref.cpu().detach().numpy(), self.step)

            for key in info_unlabel.keys():
                self.writer_finetune.add_scalar('Unlabel_info/'+key, info_unlabel[key], self.step)
            
            for key in info_pref.keys():
                self.writer_finetune.add_scalar('Pref_info/'+key, info_pref[key], self.step)
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            
            if self.step % self.save_freq == 0 and self.step:
                self.save(self.step, False, True)

            if self.step % self.log_freq == 0 and self.step > self.skip_steps:
                print(f'{self.step}: {loss_sum:8.4f} | ------ | t: {timer():8.4f}', flush=True)
                avg_return, _, avg_norm_score, std_norm_score = self.evaluate()
                self.writer_finetune.add_scalar('Evaluation/Avg_return', avg_return, global_step=self.step)
                self.writer_finetune.add_scalar('Evaluation/Avg_norm_score', avg_norm_score, global_step=self.step)
                self.writer_finetune.add_scalar('Evaluation/std_norm_score', std_norm_score, global_step=self.step)           
            self.step += 1
            

    def train_inv(self, n_train_steps):
        assert self.dataloader_inv
        assert not self.action_plan
        timer = Timer()
        self.model.train_only_inv = True
        for _ in range(n_train_steps):
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dataloader_inv)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.writer_inv.add_scalar('Loss/inv_phase_total_loss', infos['total_loss'], global_step=self.step)
            self.writer_inv.add_scalar('Loss/inv_phase_a0_loss', infos['inv_dynamic_loss'], global_step=self.step)

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            # TODO: save and evaluate model
            
            if self.step % self.save_freq == 0 and self.step:
                self.save(self.step, True)
            
            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)
            
            self.step += 1
        self.model.train_only_inv = False

    def load(self, dir):
        '''
        load model from disk
        '''
        data = torch.load(dir)
        assert 'ema' in data
        del data['ema']['loss_fn.weights']
        self.model.load_state_dict(data['ema'], strict=False)
        self.ema_model.load_state_dict(data['ema'], strict=False)
        if self.with_ref_model:
            self.ref_model.load_state_dict(data['ema'], strict=False)
        print('Load model from ' + dir)


    def save(self, epoch, only_inv=False, finetune=False):
        '''
            saves model and ema to disk;
        '''
        if not only_inv:
            data = {
                'step': self.step,
                'model': self.model.state_dict(),
                'ema': self.ema_model.state_dict()
            }
            if finetune:
                savepath = os.path.join(self.weight_dir, f'fine_model_{epoch}.pt')
            else:
                savepath = os.path.join(self.weight_dir, f'model_{epoch}.pt')
        else:
            data = {
                'step': self.step,
                'inv_model': self.model.inv_model.state_dict()
            }
            savepath = os.path.join(self.weight_dir, f'inv_model_{epoch}.pt')
        try:
            torch.save(data, savepath)
            print(f'[ utils/training ] Saved model to {savepath}', flush=True)
            # savepath_double = os.path.join(self.weight_dir, f'model_bk.pt')
            # torch.save(data, savepath_double)
        except:
            pass
        return savepath

    def evaluate(self,):
        num_eval = 10
        dataset = self.dataset if self.dataset else self.dataset_p
        observation_dim = dataset.observation_dim
        env_list = [gym.make(dataset.env_name) for _ in range(num_eval)]
        for i, item in enumerate(env_list):
            item.seed(i*10)
        dones = [0 for _ in range(num_eval)]
        episode_rewards = [0 for _ in range(num_eval)]

        #t = 0
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)

        conditions = torch.zeros([obs.shape[0], self.model.historical_horizon, obs.shape[-1]], device = self.device)
        self.model.model.eval()
        while sum(dones) < num_eval:
            obs = dataset.normalizer.normalize(obs, 'observations')
            conditions = torch.cat([conditions[:, 1:, :], to_torch(obs, device=self.device).unsqueeze(1)], dim=1)
            samples = self.ema_model.conditional_sample(conditions)
            if not self.action_plan:
                obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
                obs_comb = obs_comb.reshape(-1, 2 * observation_dim)
                if torch.any(obs_comb.isnan()):
                #break
                    print('##################NAN values, stop evaluation##################')
                    return 0, 0, 0, 0
                action = self.ema_model.inv_model(obs_comb)
            else:
                action = samples[:, 0, :]
            action = to_np(action)
            action = dataset.normalizer.unnormalize(action, 'actions')

            obs_list = []
            for i in range(num_eval):
                this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
                obs_list.append(this_obs[None])
                if this_done:
                    if dones[i] == 1:
                        pass
                    else:
                        dones[i] = 1
                        episode_rewards[i] += this_reward
                else:
                    if dones[i] == 1:
                        pass
                    else:
                        episode_rewards[i] += this_reward
            
            obs = np.concatenate(obs_list, axis=0)
        
        episode_rewards = np.array(episode_rewards)
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        normalized_scores = [env_list[0].get_normalized_score(s) for s in episode_rewards]
        avg_norm_score = env_list[0].get_normalized_score(avg_reward)
        std_norm_score = np.std(normalized_scores)
        self.model.model.train()
        return avg_reward, std_reward, avg_norm_score, std_norm_score
        
    def evaluate_action_diffusion(self,):
        num_eval = 10
        #observation_dim = self.dataset.observation_dim
        dataset = self.dataset if self.dataset else self.dataset_p
        env_list = [gym.make(dataset.env_name) for _ in range(num_eval)]
        for i, item in enumerate(env_list):
            item.seed(i*10)
        dones = [0 for _ in range(num_eval)]
        episode_rewards = [0 for _ in range(num_eval)]
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)

        self.model.model.eval()
        while sum(dones) < num_eval:
            obs = dataset.normalizer.normalize(obs, 'observations')
            conditions = to_torch(obs, device=self.device)
            action = self.ema_model.conditional_sample(conditions)         # batch x act_dim
            action  = to_np(action)
            action = dataset.normalizer.unnormalize(action, 'actions')

            obs_list = []
            for i in range(num_eval):
                this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
                obs_list.append(this_obs[None])
                if this_done:
                    if dones[i] == 1:
                        pass
                    else:
                        dones[i] = 1
                        episode_rewards[i] += this_reward
                else:
                    if dones[i] == 1:
                        pass
                    else:
                        episode_rewards[i] += this_reward
            
            obs = np.concatenate(obs_list, axis=0)
        
        episode_rewards = np.array(episode_rewards)
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        normalized_scores = [env_list[0].get_normalized_score(s) for s in episode_rewards]
        avg_norm_score = env_list[0].get_normalized_score(avg_reward)
        std_norm_score = np.std(normalized_scores)        

        self.model.model.train()
        return avg_reward, std_reward, avg_norm_score, std_norm_score
    
    def evaluate_action_diffusion_mw(self, ):
        num_eval = 100
        env_list = [MetaWorldSawyerEnv(self.env_name, randomize_hand=True) for _ in range(num_eval)]
        for i, item in enumerate(env_list):
            item.seed(i * 10)
        dones = [0 for _ in range(num_eval)]
        episode_rewards = [0 for _ in range(num_eval)]
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        dataset = self.dataset if self.dataset else self.dataset_p
        
        self.model.model.eval()
        success_num = np.zeros((num_eval,))
        while sum(dones) < num_eval:
            obs = dataset.normalizer.normalize(obs, 'observations')
            conditions = to_torch(obs, device=self.device)
            action = self.ema_model.conditional_sample(conditions)
            action  = to_np(action)
            action = dataset.normalizer.unnormalize(action, 'actions')
            
            obs_list = []
            for i in range(num_eval):
                this_obs, this_reward, this_done, info = env_list[i].step(action[i])
                obs_list.append(this_obs[None])
                if this_done:
                    if dones[i] == 1:
                        pass
                    else:
                        dones[i] = 1
                        episode_rewards[i] += this_reward
                else:
                    if dones[i] == 1:
                        pass
                    else:
                        episode_rewards[i] += this_reward
                if info['success'] > 1e-8:
                    success_num[i] = 1.
            
            obs = np.concatenate(obs_list, axis=0)
            
        episode_rewards = np.array(episode_rewards)
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        #normalized_scores = [env_list[0].get_normalized_score(s) for s in episode_rewards]          
        #avg_norm_score = env_list[0].get_normalized_score(avg_reward)
        #std_norm_score = np.std(normalized_scores)
        success_rate = np.sum(success_num) / num_eval
        self.model.model.train() 
        return avg_reward, std_reward, 0, 0, success_rate
    

class PrefTrainer(object):
    def __init__(self,
                pref_model,
                obs_dim,
                act_dim,
                query_len,
                dataset_in,
                dataset_regulizer_in,
                dataset_regulizer_out,
                dataset_eval,
                ema_decay=0.995,
                train_batch_size=32,
                train_lr = 2e-5,
                gradient_accumulate_every=2,
                step_start_ema=2000,
                update_ema_every=10,
                log_freq=100,
                train_device = 'cuda',
                result_folder='./results' 
                 ):
        super().__init__()
        self.pref_model = pref_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.pref_model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.train_batch_size = train_batch_size
        self.train_lr = train_lr
        self.gradient_accumulate_every = gradient_accumulate_every
        self.log_freq = log_freq
        self.train_device = train_device
        self.save_root_dir = result_folder
        # self.transition_dim = transition_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.query_len = query_len

        self.log_dir = os.path.join(self.save_root_dir, 'pref_training_log')
        self.weight_dir = os.path.join(self.save_root_dir, 'pref_weights')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
        
        self.optimizer = torch.optim.Adam(self.pref_model.parameters(), lr=self.train_lr)
        self.writer = SummaryWriter(self.log_dir)

        self.dataset_in = dataset_in
        self.dataset_regulizer_in = dataset_regulizer_in
        self.dataset_regulizer_out = dataset_regulizer_out
        self.dataset_eval = dataset_eval

        self.dataloader_in = cycle(torch.utils.data.DataLoader(
            self.dataset_in, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True))
        
        self.dataloader_regulizer_in = cycle(torch.utils.data.DataLoader(
            self.dataset_regulizer_in, batch_size=train_batch_size // 2, num_workers=0, shuffle=True, pin_memory=True))
        
        self.dataloader_regulizer_out = cycle(torch.utils.data.DataLoader(
            self.dataset_regulizer_out, batch_size=train_batch_size // 2, num_workers=0, shuffle=True, pin_memory=True))
        
        self.dataloader_eval = cycle(torch.utils.data.DataLoader(
            self.dataset_eval, batch_size=100, num_workers=0, shuffle=True, pin_memory=True))
        
        self.step = 0
        self.avg_acc = 0
        
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.pref_model.state_dict())
    
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.pref_model)
    
    def train(self, n_train_steps):
        for _ in range(n_train_steps):
            for _ in range(self.gradient_accumulate_every):
                in_batch = next(self.dataloader_in)
                in_batch = batch_to_device(in_batch, device=self.train_device)
                regulizer_batch_in = next(self.dataloader_regulizer_in)
                regulizer_batch_in = batch_to_device(regulizer_batch_in, device=self.train_device)
                regulizer_batch_out = next(self.dataloader_regulizer_out)
                regulizer_batch_out = batch_to_device(regulizer_batch_out, device=self.train_device)
                
                in_obs1, in_act1, in_time1, in_obs2, in_act2, in_time2, label = in_batch
                re_in_obs, re_in_act, re_in_time, re_in_obs_biased, re_in_act_biased, re_in_time_biased = regulizer_batch_in
                re_out_obs, re_out_act, re_out_time, re_out_obs_biased, re_out_act_biased, re_out_time_biased = regulizer_batch_out
                re_obs = torch.cat((re_in_obs, re_out_obs), dim=0)
                re_act = torch.cat((re_in_act, re_out_act), dim=0)
                re_time = torch.cat((re_in_time, re_out_time), dim=0)
                re_obs_biased = torch.cat((re_in_obs_biased, re_out_obs_biased), dim=0)
                re_act_biased = torch.cat((re_in_act_biased, re_out_act_biased), dim=0)
                re_time_biased = torch.cat((re_in_time_biased, re_out_time_biased), dim=0)
                loss, infos = self.pref_model.pref_predict_loss(in_obs1, in_act1, in_time1, in_obs2, in_act2, in_time2, label,
                                                                re_obs, re_act, re_time, re_obs_biased, re_act_biased, re_time_biased)
                #loss, infos = self.pref_model.loss(*in_batch, *out_batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            for key in infos.keys():
                self.writer.add_scalar('Training/'+key, infos[key], global_step=self.step)
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            
            if self.step % self.log_freq == 0:
                avg_acc, num = self.evaluate()
                self.writer.add_scalar('Evaluation/Acc', avg_acc, global_step=self.step)
                self.writer.add_scalar('Evaluation/sum_num', num, global_step=self.step)
                if avg_acc > self.avg_acc:
                    self.save(0)
                    self.avg_acc = avg_acc

            self.step += 1

    def save(self, epoch):
        data = {
            'step': self.step,
            'model': self.pref_model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.weight_dir, f'pref_model_{epoch}.pt')
        torch.save(data, savepath)
        print(f'Save model to {savepath}', flush=True)

    def evaluate(self,):
        num_eval = 100
        acc_num = 0
        sum_num = 0
        
        for idx in range(num_eval):
            while True:
                batch_1 = next(self.dataloader_eval)
                batch_2 = next(self.dataloader_eval)
                batch = add_pref_label(batch_1, batch_2)
                batch = batch_to_device(batch, device=self.train_device)
                traj_w, traj_l = batch
                if traj_w.shape[0] == traj_l.shape[0]:
                    break
            
            batch_obs_1 = traj_w[:, :, self.act_dim:]
            batch_act_1 = traj_w[:, :, 0: self.act_dim]
            batch_obs_2 = traj_l[:, :, self.act_dim:]
            batch_act_2 = traj_l[:, :, 0: self.act_dim]
            batch_size = batch_obs_1.shape[0]
            time_step = np.tile(np.arange(1, self.query_len + 1).astype(np.int32), (batch_size, 1))
            #time_step = batch_to_device(time_step)
            time_step = to_torch(time_step, dtype=torch.int32, device=self.train_device)
            predict_result = self.pref_model.predict(batch_obs_1, batch_act_1, time_step, batch_obs_2, batch_act_2, time_step) 
            acc_num += torch.sum(predict_result == 0)
            sum_num += batch_size
            
        return acc_num / (sum_num + 1e-6), sum_num

class PrefTrainerMW(object):
    def __init__(self,
                 pref_model,
                 obs_dim,
                 act_dim,
                 query_len,
                 dataset_in,
                 dataset_out,
                 dataset_eval,
                 ema_decay=0.995,
                 train_batch_size=32,
                 train_lr = 2e-5,
                 gradient_accumulate_every=2,
                 step_start_ema=2000,
                 update_ema_every=10,
                 log_freq=100,
                 train_device = 'cuda',
                 result_folder='./results'
                 ):
        super().__init__()
        self.pref_model = pref_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.pref_model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.train_batch_size = train_batch_size
        self.train_lr = train_lr
        self.gradient_accumulate_every = gradient_accumulate_every
        self.log_freq = log_freq
        self.train_device = train_device
        self.save_root_dir = result_folder
        # self.transition_dim = transition_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.query_len = query_len
        
        self.log_dir = os.path.join(self.save_root_dir, 'pref_training_log')
        self.weight_dir = os.path.join(self.save_root_dir, 'pref_weights')
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
        
        self.optimizer = torch.optim.Adam(self.pref_model.parameters(), lr=self.train_lr)
        self.writer = SummaryWriter(self.log_dir)      
        
        self.dataset_in = dataset_in
        self.dataset_out = dataset_out
        self.dataset_eval = dataset_eval
        
        self.dataloader_in = cycle(torch.utils.data.DataLoader(
            self.dataset_in, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True))
        
        self.dataloader_out = cycle(torch.utils.data.DataLoader(
            self.dataset_out, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True))
        
        self.dataloader_eval = cycle(torch.utils.data.DataLoader(
            self.dataset_eval, batch_size=100, num_workers=0, shuffle=True, pin_memory=True))
        
        self.step = 0
        self.avg_acc = 0
        
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.pref_model.state_dict())
    
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.pref_model)
    
    def train(self, n_train_steps):
        for _ in range(n_train_steps):
            for _ in range(self.gradient_accumulate_every):
                self.pref_model.train()
                in_batch = next(self.dataloader_in)
                in_batch = batch_to_device(in_batch, device=self.train_device)
                out_batch = next(self.dataloader_out)
                out_batch = batch_to_device(out_batch, device=self.train_device)
                
                in_obs1, in_act1, in_time1, in_obs2, in_act2, in_time2, label = in_batch
                out_obs, out_act, out_time, out_obs_biased, out_act_biased, out_time_biased = out_batch
                
                loss, infos = self.pref_model.pref_predict_loss(in_obs1, in_act1, in_time1, in_obs2, in_act2, in_time2, label,
                                                                out_obs, out_act, out_time, out_obs_biased, out_act_biased, out_time_biased)
                
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            for key in infos.keys():
                self.writer.add_scalar('Training/'+key, infos[key], global_step=self.step)
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            
            if self.step % self.log_freq == 0:
                avg_acc, num = self.evaluate()
                self.writer.add_scalar('Evaluation/Acc', avg_acc, global_step=self.step)
                self.writer.add_scalar('Evaluation/sum_num', num, global_step=self.step)
                if avg_acc > self.avg_acc:
                    self.save(0)
                    self.avg_acc = avg_acc

            self.step += 1
    
    def save(self, epoch):
        data = {
            'step': self.step,
            'model': self.pref_model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.weight_dir, f'pref_model_{epoch}.pt')
        torch.save(data, savepath)
        print(f'Save model to {savepath}', flush=True)

    def evaluate(self,):
        num_eval = 100
        acc_num = 0
        sum_num = 0
        
        self.pref_model.eval()
        for idx in range(num_eval):
            while True:
                batch_1 = next(self.dataloader_eval)
                batch_2 = next(self.dataloader_eval)
                batch = add_pref_label(batch_1, batch_2)
                batch = batch_to_device(batch, device=self.train_device)
                traj_w, traj_l = batch
                if traj_w.shape[0] == traj_l.shape[0]:
                    break
            
            batch_obs_1 = traj_w[:, :, self.act_dim:]
            batch_act_1 = traj_w[:, :, 0: self.act_dim]
            batch_obs_2 = traj_l[:, :, self.act_dim:]
            batch_act_2 = traj_l[:, :, 0: self.act_dim]
            batch_size = batch_obs_1.shape[0]
            time_step = np.tile(np.arange(1, self.query_len + 1).astype(np.int32), (batch_size, 1))
            #time_step = batch_to_device(time_step)
            time_step = to_torch(time_step, dtype=torch.int32, device=self.train_device)
            predict_result = self.pref_model.predict(batch_obs_1, batch_act_1, time_step, batch_obs_2, batch_act_2, time_step) 
            acc_num += torch.sum(predict_result == 0)
            sum_num += batch_size
        
            
        return acc_num / (sum_num + 1e-6), sum_num
                
                
         
        