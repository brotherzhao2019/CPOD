import metaworld
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
from metaworld.policies.sawyer_push_v2_policy import SawyerPushV2Policy
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from metaworld.policies.sawyer_door_open_v2_policy import SawyerDoorOpenV2Policy
from metaworld.policies.sawyer_drawer_open_v2_policy import SawyerDrawerOpenV2Policy
from metaworld.policies.sawyer_drawer_close_v2_policy import SawyerDrawerCloseV2Policy
from metaworld.policies.sawyer_button_press_topdown_v2_policy import SawyerButtonPressTopdownV2Policy
from metaworld.policies.sawyer_peg_insertion_side_v2_policy import SawyerPegInsertionSideV2Policy
from metaworld.policies.sawyer_window_open_v2_policy import SawyerWindowOpenV2Policy
from metaworld.policies.sawyer_window_close_v2_policy import SawyerWindowCloseV2Policy
import random
from tqdm import trange
import pickle

policy_dict = {"reach-v2": SawyerReachV2Policy, 
               "push-v2": SawyerPushV2Policy,
               "pick-place-v2": SawyerPickPlaceV2Policy,
               "door-open-v2": SawyerDoorOpenV2Policy,
               "drawer-open-v2": SawyerDrawerOpenV2Policy,
               "drawer-close-v2": SawyerDrawerCloseV2Policy,
               "button-press-topdown-v2": SawyerButtonPressTopdownV2Policy,
               "peg-insert-side-v2": SawyerPegInsertionSideV2Policy,
               "window-open-v2": SawyerWindowOpenV2Policy,
               "window-close-v2": SawyerWindowCloseV2Policy}
# sample expert data from MT-10 tasks
mt10 = metaworld.MT10() # Construct the benchmark, sampling tasks
training_envs, env_names = [], []
for name, env_cls in mt10.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in mt10.train_tasks
                        if task.env_name == name])
  env.set_task(task)
  env_names.append(name)
  training_envs.append(env)

import numpy as np
for i in range(len(training_envs)):
    data_buffer = {}
    task_list, obs_list, act_list, reward_list, next_obs_list, done_list = [], [], [], [], [], []
    env = training_envs[i]
    env_name = env_names[i]
    policy = policy_dict[env_name]()
    for k in trange(200, desc='traj'):
        obs = env.reset()[0]  # Reset environment
        done = False
        for _ in range(env.max_path_length):
            a = policy.get_action(obs)
            next_obs, reward, done, _ , info = env.step(a)  # Step the environment with the sampled random action
            task_list.append(k)
            obs_list.append(obs)
            act_list.append(a)
            reward_list.append(reward)
            next_obs_list.append(next_obs)
            done_list.append(done)
            obs = next_obs
    data_buffer['task_ids'] = np.array(task_list)
    data_buffer['observations'] = np.array(obs_list)
    data_buffer['actions'] = np.array(act_list)
    data_buffer['rewards'] = np.array(reward_list)
    data_buffer['next_observations'] = np.array(next_obs_list)
    data_buffer['dones'] = np.array(done_list)

    filename = f'./data/mt10_{env_name}.pkl'

    with open(filename, 'wb') as file:
        pickle.dump(data_buffer, file)

    print(f'Data buffer saved to {filename}')