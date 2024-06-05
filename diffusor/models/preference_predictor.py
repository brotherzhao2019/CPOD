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
import os


class PreferenceModel(nn.Module):
    def __init__(self, model, nu=1., query_len=16):
        super().__init__()
        self.model = model
        self.query_len = query_len
        self.classify_loss = nn.BCEWithLogitsLoss()
        self.smooth_loss = nn.MSELoss()
        self.nu = nu
        
    def load(self, dir):
        data = torch.load(dir, map_location=next(self.model.parameters()).device)
        self.load_state_dict(data['ema'])

    def pref_predict_loss(self, obs_1, act_1, timestep_1, obs_2, act_2, timestep_2, label, 
                          out_obs_1, out_act_1, out_time_1, out_obs_2, out_act_2, out_time_2):
        '''
            obs_1: batch x horizon x obs_dim
            act_1: batch x horizon x act_dim
            time_step: batch x hoziron
            label: batch x 2
        '''
        self.model.train()
        prediction_in_1, _ = self.model(obs_1, act_1, timestep_1, attn_mask=None, reverse=True, use_weighted_sum=False)
        prediction_in_2, _ = self.model(obs_2, act_2, timestep_2, attn_mask=None, reverse=True, use_weighted_sum=False)
        prediction_out_1, _ = self.model(out_obs_1, out_act_1, out_time_1, attn_mask=None, reverse=True, use_weighted_sum=False)
        prediction_out_2, _ = self.model(out_obs_2, out_act_2, out_time_2, attn_mask=None, reverse=True, use_weighted_sum=False)
        in_logits_1 = torch.sum(prediction_in_1['value'].squeeze(), dim=1)      # batch 
        in_logits_2 = torch.sum(prediction_in_2['value'].squeeze(), dim=1)
        out_logits_1 = torch.sum(prediction_out_1['value'].squeeze(), dim=1)
        out_logits_2 = torch.sum(prediction_out_2['value'].squeeze(), dim=1)
        
        in_logits = torch.cat((in_logits_1.unsqueeze(1), in_logits_2.unsqueeze(1)), dim=1)                # batch x 2
        
        bce_loss = self.classify_loss(in_logits, label)                         # label 中可能会包含(0.5 , 0.5)这样的标签
        
        smooth_loss = self.smooth_loss(out_logits_1, out_logits_2)
        
        total_loss = bce_loss + self.nu * smooth_loss
        
        with torch.no_grad():
            draw_mask = label[:, 0] == 0.5                                              # batch
            acc_raw = torch.argmax(in_logits, dim=1) == torch.argmax(label, dim=1)      # batch
            corr = torch.sum((~draw_mask) * acc_raw) / (torch.sum(~draw_mask) + 1e-6)
        
        logs = {}
        logs['total_loss'] = total_loss
        logs['bce_loss'] = bce_loss
        logs['smooth_loss'] = smooth_loss
        logs['training_acc'] = corr
        
        return total_loss, logs
    
    def predict(self, obs_1, act_1, timestep_1, obs_2, act_2, timestep_2):
        self.model.eval()
        with torch.no_grad():
            prediction_in_1, _ = self.model(obs_1, act_1, timestep_1, attn_mask=None, reverse=True, use_weighted_sum=False)
            prediction_in_2, _ = self.model(obs_2, act_2, timestep_2, attn_mask=None, reverse=True, use_weighted_sum=False)
            in_logits_1 = torch.sum(prediction_in_1['value'].squeeze(), dim=1)      # batch 
            in_logits_2 = torch.sum(prediction_in_2['value'].squeeze(), dim=1)
            
            in_logits = torch.cat((in_logits_1.unsqueeze(1), in_logits_2.unsqueeze(1)), dim=1)
            
            result = torch.argmax(in_logits, dim=1)                     # batch
        
        return result