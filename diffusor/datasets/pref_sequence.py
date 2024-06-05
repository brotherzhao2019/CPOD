from collections import namedtuple
import numpy as np
import torch
import pdb
# import d4rl
import os
import pickle
from tqdm import tqdm

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
from .sequence import sequence_dataset_mw

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
PrefBatch = namedtuple('BathPref', 'traj1 traj2 pref_label')
WLBatch = namedtuple('WLBatch', 'traj_w traj_l')
PrefTimeBatch = namedtuple('PrefTimeBatch', 'obs1 act1 timestep1 obs2 act2 timestep2 label')
DoubleBatch = namedtuple('DoubleBatch', 'obs, act, timestep1, obs_biased, act_biased, timestep2')


class PrefSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 env = 'walker2d-medium-expert-v2', 
                 horizon=16,                     
                 query_len = 100,
                 normalizer='LimitsNormalizer',
                 preprocess_fns=[],
                 w_l_format=False,
                 script_teacher = False,
                 label_base_dir = './diffusor/datasets/human_label/'):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env_name = env
        self.env = env = load_environment(env)
        #self.env = env
        self.horizon = horizon
        self.query_len = query_len
        self.label_base_dir = label_base_dir
        self.w_l_format=w_l_format
        self.script_teacher = script_teacher
        
        ori_dataset = d4rl.qlearning_dataset(self.env)
        self.dataset = dict(
            observations = ori_dataset['observations'],
            actions = ori_dataset['actions'],
            next_observations = ori_dataset['next_observations'],
            rewards = ori_dataset['rewards'],
            dones = ori_dataset['terminals'].astype(np.float32))

        # normalize, aligned with sequencedataset
        self.normalizer = DatasetNormalizer(self.dataset, normalizer)
        self.normalize()
        self.observation_dim = self.dataset['observations'].shape[-1]
        self.action_dim = self.dataset['actions'].shape[-1]
        self.data_with_preference, self.num_query = self.load_queries_with_indices()
        self.indices = self.make_indices()
    
    def normalize(self, keys=['observations', 'actions']):
        for key in keys:
            array = self.dataset[key]
            assert len(array.shape) <= 2
            normed = self.normalizer(array, key)
            self.dataset[f'normed_{key}'] = normed

    def load_queries_with_indices(self,):
        human_indices_2_file, human_indices_1_file, human_labels_file = sorted(os.listdir(os.path.join(self.label_base_dir, self.env_name)))
        with open(os.path.join(self.label_base_dir, self.env_name, human_indices_1_file), 'rb') as fp:
            human_indices_1 = pickle.load(fp)
        with open(os.path.join(self.label_base_dir, self.env_name, human_indices_2_file), 'rb') as fp:
            human_indices_2 = pickle.load(fp)
        with open(os.path.join(self.label_base_dir, self.env_name, human_labels_file), 'rb') as fp:
            human_labels = pickle.load(fp)
        
        saved_indices = [human_indices_1, human_indices_2]
        saved_labels = np.array(human_labels)
        num_query = len(human_labels)

        total_obs_seq_1, total_obs_seq_2 = np.zeros((num_query, self.query_len, self.observation_dim)), np.zeros((num_query, self.query_len, self.observation_dim))
        total_act_seq_1, total_act_seq_2 = np.zeros((num_query, self.query_len, self.action_dim)), np.zeros((num_query, self.query_len, self.action_dim))
        sum_reward_seq_1, sum_reward_seq_2 = np.zeros((num_query,)), np.zeros((num_query,))

        query_range = np.arange(num_query)

        for i in query_range:
            temp_count = 0
            while(temp_count < 2):
                start_idx = saved_indices[temp_count][i]
                end_idx = start_idx + self.query_len
                obs_seq = self.dataset['normed_observations'][start_idx: end_idx]
                act_seq = self.dataset['normed_actions'][start_idx: end_idx]
                reward_seq = self.dataset['rewards'][start_idx: end_idx]
                #timestep_seq = np.arange(1, self.query_len + 1)

                if temp_count == 0:
                    total_obs_seq_1[i] = obs_seq
                    total_act_seq_1[i] = act_seq
                    sum_reward_seq_1[i] = np.sum(reward_seq)
                
                else:
                    total_obs_seq_2[i] = obs_seq
                    total_act_seq_2[i] = act_seq
                    sum_reward_seq_2[i] = np.sum(reward_seq)
                
                temp_count += 1
        
        seg_obs_1 = total_obs_seq_1.copy()
        seg_obs_2 = total_obs_seq_2.copy()
        seg_reward_1 = sum_reward_seq_1.copy()

        seg_act_1 = total_act_seq_1.copy()
        seg_act_2 = total_act_seq_2.copy()
        seg_reward_2 = sum_reward_seq_2.copy()

        #seg_timestep_1 = total_timestep_1.copy()
        #seg_timestep_2 = total_timestep_2.copy()

        batch = {}
        
        reserved_idxes = saved_labels != -1

        seg_obs_1 = seg_obs_1[reserved_idxes, :, :]
        seg_obs_2 = seg_obs_2[reserved_idxes, :, :]

        seg_act_1 = seg_act_1[reserved_idxes, :, :]
        seg_act_2 = seg_act_2[reserved_idxes, :, :]
        #prefer_labels = prefer_labels[reserved_idxes, :]
        prefer_labels = saved_labels[reserved_idxes].astype(np.int64)

        batch['seg_observations_1'] = seg_obs_1
        batch['seg_observations_2'] = seg_obs_2
        batch['seg_actions_1'] = seg_act_1
        batch['seg_actions_2'] = seg_act_2

        if not self.script_teacher:
            batch['pref_labels'] = prefer_labels
        else:
            batch['pref_labels'] = (seg_reward_1 < seg_reward_2).astype(np.int64)
        
        return batch, prefer_labels.shape[0]

    def make_indices(self,):
        indices = []
        assert self.query_len >= self.horizon
        for i in range(self.num_query):
            max_start = self.query_len - self.horizon
            for start in range(max_start + 1):
                end = start + self.horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices
    
    def __len__(self,):
        return len(self.indices)
    
    def __getitem__(self, idx):
        seg_ind, start, end = self.indices[idx]
        traj_seg_1 = np.concatenate([self.data_with_preference['seg_actions_1'][seg_ind, start: end],
                                    self.data_with_preference['seg_observations_1'][seg_ind, start: end]], axis=-1)
        traj_seg_2 = np.concatenate([self.data_with_preference['seg_actions_2'][seg_ind, start: end],
                                    self.data_with_preference['seg_observations_2'][seg_ind, start: end]], axis=-1)
        preference_label = self.data_with_preference['pref_labels'][seg_ind]
        if not self.w_l_format:
            batch = PrefBatch(traj_seg_1, traj_seg_2, preference_label)
        else:
            if preference_label:
                batch = WLBatch(traj_seg_2, traj_seg_1)
            else:
                batch = WLBatch(traj_seg_1, traj_seg_2)
        return batch

class PrefSequenceDatasetMW(torch.utils.data.Dataset):
    def __init__(self,
                 pref_dataset_dir,
                 normalizer,         # Use a safe normalizer here...
                 horizon = 64,
                 query_len = 64,
                 #w_l_format=False,
                 label_key = 'rl_sum',
                 mode = 'comparison',
                 keep_indx=None):
        self.pref_dataset_dir = pref_dataset_dir
        self.horizon = horizon
        self.query_len = query_len
        #self.w_l_format = w_l_format
        self.label_key = label_key
        self.mode = mode
        self.keep_indx = keep_indx
        
        raw_dataset = np.load(self.pref_dataset_dir)
        
        self.normalizer = normalizer
        
        self.dataset = dict(
            observations = raw_dataset['obs'],
            actions = raw_dataset['action'],
            rewards = raw_dataset['reward'],
            dones = raw_dataset['done'],)
        self.dataset[self.label_key] = raw_dataset[label_key]
        
        if self.keep_indx is not None:
            #assert type(self.keep_indx) =
            self.dataset['observations'] = self.dataset['observations'][self.keep_indx, :, :]
            self.dataset['actions'] = self.dataset['actions'][self.keep_indx, :, :]
            self.dataset['rewards'] = self.dataset['rewards'][self.keep_indx]
            self.dataset['dones'] = self.dataset['dones'][self.keep_indx]
            self.dataset[self.label_key] = self.dataset[self.label_key][self.keep_indx]
        
        self.observation_dim = self.dataset['observations'].shape[-1]
        self.action_dim = self.dataset['actions'].shape[-1]
        obs_reshaped = self.dataset['observations'].reshape(-1, self.observation_dim)
        act_reshaped = self.dataset['actions'].reshape(-1, self.action_dim)
        self.dataset['normed_observations'] = self.normalizer(obs_reshaped, 'observations').reshape(self.dataset['observations'].shape)
        self.dataset['normed_actions'] = self.normalizer(act_reshaped, 'actions').reshape(self.dataset['actions'].shape)
        
        #self.dataset['normed_observations'] = self.normalizer(self.dataset['observations'], 'observations')
        #self.dataset['normed_actions'] = self.normalizer(self.dataset['actions'], 'actions')
        

        self.seg_num = self.dataset['observations'].shape[0]
        self.indices = self.make_indices()
        
    def make_indices(self):
        if self.mode == 'comparison':
            indices = np.arange(0, self.seg_num // 2)
        else:
            indices = np.arange(0, self.seg_num)
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if self.mode == 'comparison':
            seg_idx_1 = idx
            seg_idx_2 = idx + self.seg_num // 2

        else:
            seg_idx_1 = idx
            seg_idx_2 = np.random.randint(0, self.seg_num)
           
        if self.query_len > self.horizon:
            start_1 = np.random.randint(0, self.query_len - self.horizon + 1)
            end_1 = start_1 + self.horizon
            start_2 = np.random.randint(0, self.query_len - self.horizon + 1)
            end_2 = start_2 + self.horizon
        else:
            start_1 = 0
            end_1 = self.horizon
            start_2 = 0
            end_2 = self.horizon
        
        obs_seg_1 = self.dataset['normed_observations'][seg_idx_1, start_1: end_1, :]
        act_seg_1 = self.dataset['normed_actions'][seg_idx_1, start_1: end_1, :]
        label_1 = self.dataset[self.label_key][seg_idx_1]
        
        obs_seg_2 = self.dataset['normed_observations'][seg_idx_2, start_2: end_2, :]
        act_seg_2 = self.dataset['normed_actions'][seg_idx_2, start_2: end_2, :]
        label_2 = self.dataset[self.label_key][seg_idx_2]
        
        traj_seg_1 = np.concatenate([act_seg_1, obs_seg_1], axis=-1)
        traj_seg_2 = np.concatenate([act_seg_2, obs_seg_2], axis=-1)
        
        if label_1.size > 1:
            label_1 = np.sum(label_1)
            label_2 = np.sum(label_2)
            
        if label_1 > label_2:
            return WLBatch(traj_seg_1, traj_seg_2)
        else:
            return WLBatch(traj_seg_2, traj_seg_1)

class PrefPredictSequenceDatasetMW(torch.utils.data.Dataset):
    def __init__(self,
                 pref_dataset_dir,
                 normalizer,
                 horizon = 64,
                 query_len = 64,
                 label_key = 'rl_sum',
                 mode = 'comparison'):
        self.pref_dataset_dir = pref_dataset_dir
        self.horizon = horizon
        self.query_len = query_len
        self.label_key = label_key
        self.mode = mode
        self.normalizer = normalizer
        raw_dataset = np.load(self.pref_dataset_dir)
        
        self.dataset = dict(
            observations = raw_dataset['obs'],
            actions = raw_dataset['action'],
            rewards = raw_dataset['reward'],
            dones = raw_dataset['done'],)
        self.dataset[self.label_key] = raw_dataset[label_key]
        self.observation_dim = self.dataset['observations'].shape[-1]
        self.action_dim = self.dataset['actions'].shape[-1]
        
        obs_reshaped = self.dataset['observations'].reshape(-1, self.observation_dim)
        act_reshaped = self.dataset['actions'].reshape(-1, self.action_dim)
        self.dataset['normed_observations'] = self.normalizer(obs_reshaped, 'observations').reshape(self.dataset['observations'].shape)
        self.dataset['normed_actions'] = self.normalizer(act_reshaped, 'actions').reshape(self.dataset['actions'].shape)
        
        self.seg_num = self.dataset['observations'].shape[0]
        self.indices = self.make_indices()
    
    def make_indices(self):
        if self.mode == 'comparison':
            indices = np.arange(0, self.seg_num // 2)
        else:
            indices = np.arange(0, self.seg_num)
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if self.mode == 'comparison':
            seg_idx_1 = idx
            seg_idx_2 = idx + self.seg_num // 2

        else:
            seg_idx_1 = idx
            seg_idx_2 = np.random.randint(0, self.seg_num)
           
        if self.query_len > self.horizon:
            start_1 = np.random.randint(0, self.query_len - self.horizon + 1)
            end_1 = start_1 + self.horizon
            start_2 = np.random.randint(0, self.query_len - self.horizon + 1)
            end_2 = start_2 + self.horizon
        else:
            start_1 = 0
            end_1 = self.horizon
            start_2 = 0
            end_2 = self.horizon
        
        obs_seg_1 = self.dataset['normed_observations'][seg_idx_1, start_1: end_1, :]
        act_seg_1 = self.dataset['normed_actions'][seg_idx_1, start_1: end_1, :]
        label_1 = self.dataset[self.label_key][seg_idx_1]
        
        obs_seg_2 = self.dataset['normed_observations'][seg_idx_2, start_2: end_2, :]
        act_seg_2 = self.dataset['normed_actions'][seg_idx_2, start_2: end_2, :]
        label_2 = self.dataset[self.label_key][seg_idx_2]
        
        label = np.zeros((2, ))
        
        if label_1.shape[0] > 1:
            label_1 = np.sum(label_1)
            label_2 = np.sum(label_2)
            
        if label_1 > label_2:
            label[0] = 1
        elif label_1 < label_2:
            label[1] = 1
        else:
            label[0] = label[1] = 0.5
            
        timestep1 = np.arange(1, self.horizon + 1).astype(np.int32)
        timestep2 = np.arange(1, self.horizon + 1).astype(np.int32)
        
        batch = PrefTimeBatch(obs_seg_1, act_seg_1, timestep1, obs_seg_2, act_seg_2, timestep2, label)
        
        return batch
    
class DoubleSequenceDatasetMW(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, horizon = 64,
                 normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=300,
                 smooth_sigma=10):
        self.dataset_dir = dataset_dir
        #self.preprocess_fn = get_preprocess_fn(preprocess_fns)
        self.horizon = horizon
        self.smooth_sigma = smooth_sigma
        self.max_path_length = max_path_length
        self.dataset_raw = np.load(self.dataset_dir)
        
        itr = sequence_dataset_mw(self.dataset_raw)
        
        fields = ReplayBuffer(10000, self.max_path_length, 0)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        
        fields.finalize()
        
        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = np.arange(0, fields.n_episodes)
        
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
    
    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
            TODO: 注意，这里要注意要和 preference finetune 部分对齐，注意分别查清楚 observation 和 action normalize 的方式
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        seg_length = self.path_lengths[idx]
        start_1 = np.random.randint(0, seg_length - self.horizon + 1)
        end_1 = start_1 + self.horizon
        bias = np.round(np.random.randn() * self.smooth_sigma).astype(int)
        if start_1 + bias < 0:
            start_2 = 0
            end_2 = self.horizon
        elif end_1 + bias > seg_length:
            end_2 = seg_length
            start_2 = end_2 - self.horizon
        else:
            start_2, end_2 = start_1 + bias, end_1 + bias
        
        timestep1 = np.arange(1, self.horizon + 1).astype(np.int32)
        timestep2 = np.arange(1, self.horizon + 1).astype(np.int32)
        obs = self.fields.normed_observations[idx, start_1: end_1, :]
        act = self.fields.normed_actions[idx, start_1: end_1, :]
        obs_biased = self.fields.normed_observations[idx, start_2: end_2, :]
        act_biased = self.fields.normed_actions[idx, start_2: end_2, :]
        batch = DoubleBatch(obs, act, timestep1, obs_biased, act_biased, timestep2)
        return batch
    
class PrefPredictSequenceDataset(torch.utils.data.Dataset):
    def __init__(self,
                 env = 'walker2d-medium-expert-v2',
                 seq_len = 16,
                 query_len = 100,
                 normalizer = 'LimitsNormalizer',
                 preprocess_fns=[],
                 script_teacher = True,
                 label_base_dir = './diffusor/datasets/human_label/'):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env_name = env
        self.env = env = load_environment(env)
        
        self.seq_len = seq_len
        self.query_len = query_len
        self.label_base_dir = label_base_dir
        self.script_teacher = script_teacher
        
        ori_dataset = d4rl.qlearning_dataset(self.env)
        self.dataset = dict(
            observations = ori_dataset['observations'],
            actions = ori_dataset['actions'],
            next_observations = ori_dataset['next_observations'],
            rewards = ori_dataset['rewards'],
            dones = ori_dataset['terminals'].astype(np.float32))

        self.normalizer = DatasetNormalizer(self.dataset, normalizer)
        self.normalize()
        self.observation_dim = self.dataset['observations'].shape[-1]
        self.action_dim = self.dataset['actions'].shape[-1]
        self.data_with_preference, self.num_query = self.load_queries_with_indices()
        self.indices = self.make_indices()
    
    def normalize(self, keys=['observations', 'actions']):
        for key in keys:
            array = self.dataset[key]
            assert len(array.shape) <= 2
            normed = self.normalizer(array, key)
            self.dataset[f'normed_{key}'] = normed
        
    def load_queries_with_indices(self, ):
        human_indices_2_file, human_indices_1_file, human_labels_file = sorted(os.listdir(os.path.join(self.label_base_dir, self.env_name)))
        with open(os.path.join(self.label_base_dir, self.env_name, human_indices_1_file), 'rb') as fp:
            human_indices_1 = pickle.load(fp)
        with open(os.path.join(self.label_base_dir, self.env_name, human_indices_2_file), 'rb') as fp:
            human_indices_2 = pickle.load(fp)
        with open(os.path.join(self.label_base_dir, self.env_name, human_labels_file), 'rb') as fp:
            human_labels = pickle.load(fp)
            
        saved_indices = [human_indices_1, human_indices_2]
        saved_labels = np.array(human_labels)
        num_query = len(human_labels)
        
        total_obs_seq_1, total_obs_seq_2 = np.zeros((num_query, self.query_len, self.observation_dim)), np.zeros((num_query, self.query_len, self.observation_dim))
        total_act_seq_1, total_act_seq_2 = np.zeros((num_query, self.query_len, self.action_dim)), np.zeros((num_query, self.query_len, self.action_dim))
        sum_reward_seq_1, sum_reward_seq_2 = np.zeros((num_query,)), np.zeros((num_query,))
        
        query_range = np.arange(num_query)
        
        for i in query_range:
            temp_count = 0
            while(temp_count < 2):
                start_idx = saved_indices[temp_count][i]
                end_idx = start_idx + self.query_len
                obs_seq = self.dataset['normed_observations'][start_idx: end_idx]
                act_seq = self.dataset['normed_actions'][start_idx: end_idx]
                reward_seq = self.dataset['rewards'][start_idx: end_idx]
                #timestep_seq = np.arange(1, self.query_len + 1)

                if temp_count == 0:
                    total_obs_seq_1[i] = obs_seq
                    total_act_seq_1[i] = act_seq
                    sum_reward_seq_1[i] = np.sum(reward_seq)
                
                else:
                    total_obs_seq_2[i] = obs_seq
                    total_act_seq_2[i] = act_seq
                    sum_reward_seq_2[i] = np.sum(reward_seq)
                
                temp_count += 1
        
        seg_obs_1 = total_obs_seq_1.copy()
        seg_obs_2 = total_obs_seq_2.copy()
        #seg_reward_1 = sum_reward_seq_1.copy()

        seg_act_1 = total_act_seq_1.copy()
        seg_act_2 = total_act_seq_2.copy()
        #seg_reward_2 = sum_reward_seq_2.copy()
        
        batch = {}
        
        pref_labels = np.zeros((num_query, 2))
        
        if self.script_teacher:
            for i in range(num_query):
                if sum_reward_seq_1[i] > sum_reward_seq_2[i]:
                    pref_labels[i, 0] = 1.0
                else:
                    pref_labels[i, 1] = 1.0
        else:
            pref_labels[saved_labels == 0, 0] = 1
            pref_labels[saved_labels == 1, 1] = 1
            pref_labels[saved_labels == -1, :] = 0.5
            
        batch['seg_observations_1'] = seg_obs_1
        batch['seg_observations_2'] = seg_obs_2
        batch['seg_actions_1'] = seg_act_1
        batch['seg_actions_2'] = seg_act_2
        batch['pref_labels'] = pref_labels
        
        return batch, pref_labels.shape[0]
    
    def make_indices(self,):
        indices = []
        assert self.query_len >= self.seq_len
        for i in range(self.num_query):
            max_start = self.query_len - self.seq_len
            for start in range(max_start + 1):
                end = start + self.seq_len
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices
    
    def __len__(self, ):
        return len(self.indices)
    
    def __getitem__(self, idx):
        seg_ind, start, end = self.indices[idx]
        obs_1 = self.data_with_preference['seg_observations_1'][seg_ind, start: end]
        act_1 = self.data_with_preference['seg_actions_1'][seg_ind, start: end]
        time_1 = np.arange(1, self.seq_len + 1).astype(np.int32)
        obs_2 = self.data_with_preference['seg_observations_2'][seg_ind, start: end]
        act_2 = self.data_with_preference['seg_actions_2'][seg_ind, start: end]
        time_2 = np.arange(1, self.seq_len + 1).astype(np.int32)
        label = self.data_with_preference['pref_labels'][seg_ind]
        
        batch = PrefTimeBatch(obs_1, act_1, time_1, obs_2, act_2, time_2, label)
        
        return batch
    
class DoubleSequenceInDataset(torch.utils.data.Dataset):
    def __init__(self, env='hopper-medium-replay', horizon=16,
                 normalizer='LimitsNormalizer', preprocess_fns=[], in_query_len=100,
                 smooth_sigma=10, label_base_dir='./diffusor/datasets/human_label/'):
        self.env_name = env
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon
        #self.max_path_length = max_path_length
        self.in_query_len = in_query_len
        #self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.label_base_dir = label_base_dir
        self.smooth_sigma = smooth_sigma
        
        ori_dataset = d4rl.qlearning_dataset(self.env)
        self.dataset = dict(
            observations = ori_dataset['observations'],
            actions = ori_dataset['actions'],
            next_observations = ori_dataset['next_observations'],
            rewards = ori_dataset['rewards'],
            dones = ori_dataset['terminals'].astype(np.float32))
        
        self.observation_dim = self.dataset['observations'].shape[1]
        self.action_dim = self.dataset['actions'].shape[1]
        
        self.normalizer = DatasetNormalizer(self.dataset, normalizer)
        self.normalize()
        
        dones_float = np.zeros_like(self.dataset['rewards'])
        
        for i in range(len(dones_float) - 1):
            if np.linalg.norm(self.dataset['observations'][i + 1]-self.dataset['next_observations'][i]) > 1e-5 or self.dataset['dones'][i] == 1.0:
                dones_float[i] = 1.
            else:
                dones_float[i] = 0.
        dones_float[-1] = 1.
        
        self.dataset['dones_float'] = dones_float
        
        # build sequence indices
        ending_points = np.where(self.dataset['dones_float'] > 0)[0]
        ending_points = np.concatenate([[-1], ending_points])
        self.traj_lens = np.diff(ending_points)
        self.traj_starting_points = ending_points[:-1] + 1
        
        self.traj_num = len(self.traj_lens)
        
        human_indices_2_file, human_indices_1_file, human_labels_file = sorted(os.listdir(os.path.join(self.label_base_dir, self.env_name)))
        with open(os.path.join(self.label_base_dir, self.env_name, human_indices_1_file), 'rb') as fp:
            human_indices_1 = pickle.load(fp)
        with open(os.path.join(self.label_base_dir, self.env_name, human_indices_2_file), 'rb') as fp:
            human_indices_2 = pickle.load(fp)
        
        in_start_indices = human_indices_1.tolist() + human_indices_2.tolist()
        in_start_indices_extend = []
        for item in in_start_indices:
            for i in range(self.in_query_len - self.horizon + 1):
                in_start_indices_extend.append(item + i)
        
        in_start_indices_extend = set(in_start_indices_extend)
        seq_indices = []
        seq_in_indices = []
        seq_traj_indices = []
        
        seq_indices_cutting_points = [0]
        traj_returns = []
        traj_complete = []
        last_done = -1
        for i, done in tqdm(enumerate(dones_float), total=len(dones_float), desc="calc seq indices"):
            seq_start = max(last_done + 1, i - self.horizon + 1)
            seq_end = i + 1
            if done > 0:
                traj_complete.append(True if self.dataset["dones"][i] else False)
                traj_returns.append(self.dataset["rewards"][last_done+1:i+1].sum())
                last_done = i
            if seq_end - seq_start < self.horizon:
                continue
            seq_indices.append([seq_start, seq_end])
            seq_traj_indices.append(len(seq_indices_cutting_points) - 1)
            
            if seq_start in in_start_indices_extend:
                seq_in_indices.append(len(seq_indices) - 1)
            
            if done > 0:
                seq_indices_cutting_points.append(len(seq_indices))
        
        self.seq_indices = np.array(seq_indices)
        self.seq_traj_indices = np.array(seq_traj_indices)
        self.seq_size = len(self.seq_indices)
        self.seq_indices_starting_points = np.array(seq_indices_cutting_points[:-1])
        self.seq_indices_ending_points = np.array(seq_indices_cutting_points[1:])
        self.traj_complete = np.array(traj_complete)
        self.traj_returns = np.array(traj_returns)
        self.seq_in_indices = np.array(seq_in_indices)
        #self.seq_size = len(seq_in_indices)
        
        
        self.seq_observations = np.zeros((self.seq_size, self.horizon, self.observation_dim), np.float32)
        self.seq_actions = np.zeros((self.seq_size, self.horizon, self.action_dim), np.float32)
        self.seq_rewards = np.zeros((self.seq_size, self.horizon), np.float32)
        self.seq_masks = np.zeros((self.seq_size, self.horizon), np.float32)
        self.seq_timesteps = np.zeros((self.seq_size, self.horizon), np.int32)
        
        for i in tqdm(range(self.seq_size), total=self.seq_size, desc='build seq data'):
            seq_start, seq_end = self.seq_indices[i]
            seq_len_i = seq_end - seq_start
            assert seq_len_i == self.horizon
            self.seq_observations[i, :seq_len_i, :] = self.dataset['normed_observations'][seq_start:seq_end, :]
            self.seq_actions[i, :seq_len_i, :] = self.dataset['normed_actions'][seq_start:seq_end, :]
            self.seq_rewards[i, :seq_len_i,] = self.dataset['rewards'][seq_start:seq_end]
            self.seq_masks[i, :seq_len_i] = 1
            timestep_start = 1
            timestep_end = timestep_start + seq_len_i
            self.seq_timesteps[i, :seq_len_i] = np.arange(timestep_start, timestep_end, dtype=np.int32)
            
        # build seq bounds
        seq_indx_low, seq_indx_high = [], []
        for i, traj_idx in enumerate(self.seq_traj_indices):
            if i == 0:
                cur_low = 0
                cur_high = 0
                cur_traj_idx = traj_idx
            else:
                if cur_traj_idx == traj_idx:
                    cur_high = i
                else:
                    seq_indx_low += [cur_low] * (cur_high - cur_low + 1)
                    seq_indx_high += [cur_high] * (cur_high - cur_low + 1)
                    cur_traj_idx = traj_idx
                    cur_low, cur_high = i, i
        
        seq_indx_low += [cur_low] * (cur_high - cur_low + 1)
        seq_indx_high += [cur_high] * (cur_high - cur_low + 1)
        
        self.seq_indx_low = np.array(seq_indx_low)
        self.seq_indx_high = np.array(seq_indx_high)

    
    def normalize(self, keys=['observations', 'actions', 'next_observations']):
        for key in keys:
            array = self.dataset[key]
            assert len(array.shape) <= 2
            normed = self.normalizer(array, key)
            self.dataset[f'normed_{key}'] = normed
    
    def __len__(self):
        return self.seq_in_indices.shape[0]
    
    def __getitem__(self, idx):
        obs = self.seq_observations[self.seq_in_indices[idx], :, :]
        act = self.seq_actions[self.seq_in_indices[idx], :, :]
        time1 = self.seq_timesteps[self.seq_in_indices[idx], :]
        low_bound = self.seq_indx_low[self.seq_in_indices[idx]]
        up_bound = self.seq_indx_high[self.seq_in_indices[idx]]
        bias = np.round(np.random.randn() * self.smooth_sigma).astype(int)
        idx_biased = np.clip(self.seq_in_indices[idx] + bias, low_bound, up_bound)
        
        obs_biased = self.seq_observations[idx_biased, :, :]
        act_biased = self.seq_actions[idx_biased, :, :]
        time2 = self.seq_timesteps[idx_biased, :]
        batch = DoubleBatch(obs, act, time1, obs_biased, act_biased, time2)
        return batch

class DoubleSequenceOutDataset(torch.utils.data.Dataset):
    def __init__(self, env='hopper-medium-replay', horizon=16,
                 normalizer='LimitsNormalizer', preprocess_fns=[], in_query_len=100,
                 smooth_sigma=10, ):
        self.env_name = env
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon
        #self.max_path_length = max_path_length
        self.in_query_len = in_query_len
        #self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        #self.label_base_dir = label_base_dir
        self.smooth_sigma = smooth_sigma
        
        ori_dataset = d4rl.qlearning_dataset(self.env)
        self.dataset = dict(
            observations = ori_dataset['observations'],
            actions = ori_dataset['actions'],
            next_observations = ori_dataset['next_observations'],
            rewards = ori_dataset['rewards'],
            dones = ori_dataset['terminals'].astype(np.float32))
        
        self.observation_dim = self.dataset['observations'].shape[1]
        self.action_dim = self.dataset['actions'].shape[1]
        
        self.normalizer = DatasetNormalizer(self.dataset, normalizer)
        self.normalize()
        
        dones_float = np.zeros_like(self.dataset['rewards'])
        
        for i in range(len(dones_float) - 1):
            if np.linalg.norm(self.dataset['observations'][i + 1]-self.dataset['next_observations'][i]) > 1e-5 or self.dataset['dones'][i] == 1.0:
                dones_float[i] = 1.
            else:
                dones_float[i] = 0.
        dones_float[-1] = 1.
        
        self.dataset['dones_float'] = dones_float
        
        # build sequence indices
        ending_points = np.where(self.dataset['dones_float'] > 0)[0]
        ending_points = np.concatenate([[-1], ending_points])
        self.traj_lens = np.diff(ending_points)
        self.traj_starting_points = ending_points[:-1] + 1
        
        self.traj_num = len(self.traj_lens)
        
        seq_indices = []
        seq_traj_indices = []
        
        seq_indices_cutting_points = [0]
        traj_returns = []
        traj_complete = []
        last_done = -1
        for i, done in tqdm(enumerate(dones_float), total=len(dones_float), desc="calc seq indices"):
            seq_start = max(last_done + 1, i - self.horizon + 1)
            seq_end = i + 1
            if done > 0:
                traj_complete.append(True if self.dataset["dones"][i] else False)
                traj_returns.append(self.dataset["rewards"][last_done+1:i+1].sum())
                last_done = i
            if seq_end - seq_start < self.horizon:
                continue
            seq_indices.append([seq_start, seq_end])
            seq_traj_indices.append(len(seq_indices_cutting_points) - 1)
            if done > 0:
                seq_indices_cutting_points.append(len(seq_indices))
        
        self.seq_indices = np.array(seq_indices)
        self.seq_traj_indices = np.array(seq_traj_indices)
        self.seq_size = len(self.seq_indices)
        self.seq_indices_starting_points = np.array(seq_indices_cutting_points[:-1])
        self.seq_indices_ending_points = np.array(seq_indices_cutting_points[1:])
        self.traj_complete = np.array(traj_complete)
        self.traj_returns = np.array(traj_returns)
        
        self.seq_observations = np.zeros((self.seq_size, self.horizon, self.observation_dim), np.float32)
        self.seq_actions = np.zeros((self.seq_size, self.horizon, self.action_dim), np.float32)
        self.seq_rewards = np.zeros((self.seq_size, self.horizon), np.float32)
        self.seq_masks = np.zeros((self.seq_size, self.horizon), np.float32)
        self.seq_timesteps = np.zeros((self.seq_size, self.horizon), np.int32)
        
        for i in tqdm(range(self.seq_size), total=self.seq_size, desc='build seq data'):
            seq_start, seq_end = self.seq_indices[i]
            seq_len_i = seq_end - seq_start
            assert seq_len_i == self.horizon
            self.seq_observations[i, :seq_len_i, :] = self.dataset['normed_observations'][seq_start:seq_end, :]
            self.seq_actions[i, :seq_len_i, :] = self.dataset['normed_actions'][seq_start:seq_end, :]
            self.seq_rewards[i, :seq_len_i,] = self.dataset['rewards'][seq_start:seq_end]
            self.seq_masks[i, :seq_len_i] = 1
            timestep_start = 1
            timestep_end = timestep_start + seq_len_i
            self.seq_timesteps[i, :seq_len_i] = np.arange(timestep_start, timestep_end, dtype=np.int32)
            
        # build seq bounds
        seq_indx_low, seq_indx_high = [], []
        for i, traj_idx in enumerate(self.seq_traj_indices):
            if i == 0:
                cur_low = 0
                cur_high = 0
                cur_traj_idx = traj_idx
            else:
                if cur_traj_idx == traj_idx:
                    cur_high = i
                else:
                    seq_indx_low += [cur_low] * (cur_high - cur_low + 1)
                    seq_indx_high += [cur_high] * (cur_high - cur_low + 1)
                    cur_traj_idx = traj_idx
                    cur_low, cur_high = i, i
        
        seq_indx_low += [cur_low] * (cur_high - cur_low + 1)
        seq_indx_high += [cur_high] * (cur_high - cur_low + 1)
        
        self.seq_indx_low = np.array(seq_indx_low)
        self.seq_indx_high = np.array(seq_indx_high)

    
    def normalize(self, keys=['observations', 'actions', 'next_observations']):
        for key in keys:
            array = self.dataset[key]
            assert len(array.shape) <= 2
            normed = self.normalizer(array, key)
            self.dataset[f'normed_{key}'] = normed
    
    def __len__(self):
        return self.seq_size
    
    def __getitem__(self, idx):
        obs = self.seq_observations[idx, :, :]
        act = self.seq_actions[idx, :, :]
        time1 = self.seq_timesteps[idx, :]
        low_bound = self.seq_indx_low[idx]
        up_bound = self.seq_indx_high[idx]
        bias = np.round(np.random.randn() * self.smooth_sigma).astype(int)
        idx_biased = np.clip(idx + bias, low_bound, up_bound)
        
        obs_biased = self.seq_observations[idx_biased, :, :]
        act_biased = self.seq_actions[idx_biased, :, :]
        time2 = self.seq_timesteps[idx_biased, :]
        batch = DoubleBatch(obs, act, time1, obs_biased, act_biased, time2)
        return batch    