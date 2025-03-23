import pickle
import os
from typing import Optional

import gym
from gym.wrappers import RescaleAction
from gym.wrappers.pixel_observation import PixelObservationWrapper

from continuous_control import wrappers
from continuous_control.wrappers import VideoRecorder


def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True) -> gym.Env:
    # Check if the env is in gym.
    #all_envs = gym.envs.registry.all()
    all_envs = gym.envs.registry.values()
    env_ids = [env_spec.id for env_spec in all_envs]

    if env_name in env_ids:
        env = gym.make(env_name)
    else:
        domain_name, task_name = env_name.split('-')
        env = wrappers.DMCEnv(domain_name=domain_name,
                              task_name=task_name,
                              task_kwargs={'random': seed})

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = VideoRecorder(env, save_folder=save_folder)

    if from_pixels:
        if env_name in env_ids:
            camera_id = 0
        else:
            camera_id = 2 if domain_name == 'quadruped' else 0
        env = PixelObservationWrapper(env,
                                      pixels_only=pixels_only,
                                      render_kwargs={
                                          'pixels': {
                                              'height': image_size,
                                              'width': image_size,
                                              'camera_id': camera_id
                                          }
                                      })
        env = wrappers.TakeKey(env, take_key='pixels')
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def save_agent(agent, path: str):
    os.makedirs(path, exist_ok=True)

    data = {
        'actor': {
            'params': agent.actor.params,
            'opt_state': agent.actor.opt_state,
        },
        'critic': {
            'params': agent.critic.params,
            'opt_state': agent.critic.opt_state,
        },
        'target_critic': {
            'params': agent.target_critic.params,
        },
        'temp': {
            'params': agent.temp.params,
            'opt_state': agent.temp.opt_state,
        },
        'rng': agent.rng,
        'step': agent.step
    }

    with open(os.path.join(path, 'agent.pkl'), 'wb') as f:
        pickle.dump(data, f)

    print(f"✅ Agent saved at {path}/agent.pkl")

def load_agent(path: str, agent_class, **init_kwargs):
    with open(os.path.join(path, 'agent.pkl'), 'rb') as f:
        data = pickle.load(f)

    agent = agent_class(**init_kwargs)

    agent.actor = agent.actor.replace(
        params=data['actor']['params'],
        opt_state=data['actor']['opt_state']
    )

    agent.critic = agent.critic.replace(
        params=data['critic']['params'],
        opt_state=data['critic']['opt_state']
    )

    agent.target_critic = agent.target_critic.replace(
        params=data['target_critic']['params']
    )

    agent.temp = agent.temp.replace(
        params=data['temp']['params'],
        opt_state=data['temp']['opt_state']
    )

    agent.rng = data['rng']
    agent.step = data['step']

    print(f"✅ Agent loaded from {path}/agent.pkl")
    return agent
