#from diffuser.utils import watch
from diffusor.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]


# change in your experiment
logbase = '../logs'
human_label_base = '../diffusor/datasets/human_label/'

base = {
    'diffusion': {
        'device': 'cuda:0',
        'is_dlc': False,
        'seed': 100,
        'save_path': None,
        'exp_save_prefix': 'exp_door-open',     
         ## model
        'model': 'models.ResNetBackbone',
        'horizon': 64,
        'historical_horizon': 1,
        'check_point_dir': '',
        'finetune_coef': 0.9,
        'dist_temperature': .5, 
        'init_model_dir':'../weights/door-open-v2-pretrain.pt',
        'pretrain_only': False,
        'regulizer_lambda': 1.,
        'bc_coef':1.,
        'with_ref_model': False,       
        'no_label_bc': False,         
        'no_pretrain': False,        
        'ismetaworld': True,
        'bc_update_period': 1000000,
        'bc_update_mulplier': 1.,
        'lr_decay_beta': 20,
        'weight_decay': 0.0,
        'loss_bias': 0.0,

         # other parameters
        'diffusion': 'models.ActionDiffusion',
        'n_diffusion_steps': 200,
        'loss_discount': 1.0,
        'predict_epsilon': True,
        'gpt_backbone': True,
        'action_plan': True,
        'clip_denoised': True,
        
        # dataset for pretraining
        'loader': 'datasets.SequenceDataset',
        'dataset_root': '../dataset/mw/data',       
        'dataset_dir': '',
        'env_name': 'door-open-v2',
        'normalizer': 'SafeLimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 1000,
        'termination_penalty': -100,
        'bc_percentage': None,
        'dpo_beta': 500,
        'script_teacher': True,
        'pref_model_dir': '',
        'label_key': 'rl_sum',
        
        # for perference predict model
        'smooth_sigma': 10.,
        'smooth_nu': 1.0,
        'pref_batch_size': 64,
        'pref_lr':2e-4,

        # dataset for finetuning
        'loader_fine': 'datasets.PrefSequenceDataset',
        'query_len': 100,
        'log_base': logbase,
        'label_base_dir': human_label_base,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),
        'pref_dataset_root': '../dataset/mw/pref',
        'pref_dataset_dir': '',
        'mode': 'comparison',

        'n_steps_per_epoch': 10000,
        'loss_type': 'state_l2',
        'n_train_steps': 300 * 1000,
        'val_freq': 0,
        'skip_steps': 0,
        'n_fintune_steps': 400000,
        'n_train_inv_steps': 1e5,
        'batch_size': 32,
        'learning_rate': 2e-5,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'log_freq': 2000,
        'save_freq': 2000,
        'sample_freq': 20000,
        'n_saves': 5,
        'save_parallel': False,
        'lr_schedule': 'exp'
    },
}


#------------------------ overrides ------------------------#

