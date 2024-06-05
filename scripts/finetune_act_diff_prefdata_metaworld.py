import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import diffusor.utils as utils
import datetime
from diffusor.datasets import PrefPredictSequenceDataset, DoubleSequenceInDataset, DoubleSequenceOutDataset, SequenceDataset, SequenceDatasetMW, PrefSequenceDatasetMW
from diffusor.models import PreferenceModel, TransRewardModel, ResNetBackbone, ActionDiffusion
from diffusor.utils.training import PrefTrainer

class Parser(utils.Parser):
    config: str = 'config.metaworld'
    dataset: str = 'test-'

args = Parser().parse_args('diffusion')

utils.set_seed(args.seed)

args.log_base = '../logs'
args.label_base_dir = '../diffusor/datasets/human_label/'
args.dataset_root = '../dataset/mw/data'
args.pref_dataset_root = '../dataset/mw/pref'

dataset_dir_1 = os.path.join(args.dataset_root, 'mw_' + args.env_name + '_ep2500_n0.3')
dataset_file_name, _ = sorted(os.listdir(dataset_dir_1))
assert dataset_file_name[-3:] == 'npz'
args.dataset_dir = os.path.join(dataset_dir_1, dataset_file_name)

args.pref_dataset_dir = os.path.join(args.pref_dataset_root, 'mw_' + args.env_name + '_ep2500_n0.3.npz')

if not args.exp_save_prefix:
    save_prefix = 'metaworld-debug-' + args.env_name + '-'
else:
    save_prefix = args.exp_save_prefix + '-'
log_root_dir = os.path.join(args.log_base, save_prefix + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))

if not os.path.exists(log_root_dir):
    os.makedirs(log_root_dir)
    

dataset = SequenceDatasetMW(dataset_dir=args.dataset_dir,
                            horizon=64, normalizer=args.normalizer,
                            include_returns=False)

dataset_fine = PrefSequenceDatasetMW(pref_dataset_dir=args.pref_dataset_dir, normalizer=dataset.normalizer,
                                     horizon=64, query_len=64, mode=args.mode)

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


diffusor_network = ResNetBackbone(obs_dim = observation_dim, act_dim = action_dim).to(args.device)


diffusor = ActionDiffusion(model=diffusor_network, horizon=args.horizon, observation_dim = observation_dim, action_dim = action_dim,
                           dpo_beta = args.dpo_beta, regulizer_lambda=args.regulizer_lambda, bc_coef = args.bc_coef, loss_type=args.loss_type,
                           clip_denoised=args.clip_denoised, loss_bias=args.loss_bias).to(args.device)

trainer = utils.Trainer(diffusor = diffusor, dataset = dataset, dataset_p = dataset_fine, dataset_inv = None, renderer=None, horizon=args.horizon,
                        train_batch_size = args.batch_size, train_lr = args.learning_rate, pref_model = None,
                        log_freq = args.log_freq, skip_steps = args.skip_steps, with_ref_model = args.with_ref_model,
                        no_label_bc = args.no_label_bc, save_freq=args.save_freq, init_model_dir = args.init_model_dir, 
                        gradient_accumulate_every=args.gradient_accumulate_every, results_folder=log_root_dir, ismetaworld=True, env_name=args.env_name,
                        bc_update_period=args.bc_update_period, bc_update_mulplier=args.bc_update_mulplier, lr_decay_beta=args.lr_decay_beta,
                        weight_decay=args.weight_decay, total_finetune_steps=args.n_fintune_steps, lr_schedule=args.lr_schedule)


trainer.step = 0

n_epochs = int(args.n_fintune_steps // args.n_steps_per_epoch)
assert args.init_model_dir

trainer.load(args.init_model_dir)

for i in range(n_epochs):
    trainer.finetune_dpo(n_train_steps = args.n_steps_per_epoch)
    print(i)

print('Finetune end')