import os
import random
import wandb
import numpy as np
import tqdm
import pdb
from absl import app, flags
from ml_collections import config_flags
import jax
import jax.numpy as jnp
from continuous_control.agents import SACLearner
from continuous_control.datasets import ReplayBuffer
from continuous_control.evaluation import evaluate
from continuous_control.utils import make_env, save_agent, load_agent

FLAGS = flags.FLAGS

flags.DEFINE_string('exp', '', 'Experiment description (not actually used).')
flags.DEFINE_string('env_name', 'quadruped-run', 'Environment name.')
flags.DEFINE_string('save_dir', './out/', 'Logging dir.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 100, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(2e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                    'Number of training steps to start training.')
flags.DEFINE_integer('reset_interval', int(2e5), 'Periodicity of resets.')
flags.DEFINE_boolean('resets', False, 'Periodically reset the agent networks.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')

flags.DEFINE_integer('distill_steps', 25000, '_') # 5000 10k, 25k
flags.DEFINE_integer('lambda_decay_rate', 1000, '_')
flags.DEFINE_float('tau', 0.005, '')
flags.DEFINE_string('job_id', os.getenv("SLURM_JOB_ID", "unknown"), '_')


flags.DEFINE_integer('save_agent_at', -1, 'Save agent at specific timestep. -1 to disable.')
flags.DEFINE_integer('load_agent_step', -1, 'If > 0, load agent saved at this timestep.')

config_flags.DEFINE_config_file(
    'config',
    'configs/sac.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

def collect_expert_data(teacher_agent, env, kwargs, replay_buffer_size, num_samples=50000):
    """Collects expert data by running the SAC teacher policy in the environment."""
    expert_buffer = ReplayBuffer(env.observation_space, env.action_space.shape[0], num_samples)
    
    observation, done = env.reset(), False
    for _ in range(num_samples):
        action = teacher_agent.sample_actions(observation)  # Teacher selects action
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0
        expert_buffer.insert(observation, action, reward, mask, float(done), next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False

    return expert_buffer


def distill(teacher_agent, env, kwargs, replay_buffer_size, replay_buffer, i):
    # Create student agent using SACLearner
    student_agent = SACLearner(
        FLAGS.seed + 1000,  # Ensure a different seed # TODO What happned if we keep the same seed?
        env.observation_space.sample()[np.newaxis],
        env.action_space.sample()[np.newaxis],
        **kwargs
    )
    # 
    # expert_buffer = collect_expert_data(teacher_agent, env, kwargs, replay_buffer_size)

    kl_losses = []
    for step in range(FLAGS.distill_steps):
        # Sample batch from the replay buffer
        batch = replay_buffer.sample(FLAGS.batch_size)  # TODO replace with batch_size
        states, actions, rewards, next_states, dones = batch
        
        # === Actor Distillation (KL Divergence Loss) ===
        # Get teacher actions and log probabilities
        teacher_actions = teacher_agent.sample_actions(states)
        teacher_dist = teacher_agent.actor.apply({'params': teacher_agent.actor.params}, states)
        teacher_log_probs = teacher_dist.log_prob(teacher_actions)

        # Get student actions and log probabilities
        student_actions = student_agent.sample_actions(states)
        student_dist = student_agent.actor.apply({'params': student_agent.actor.params}, states)
        student_log_probs = student_dist.log_prob(student_actions)

        analytical_kl_div = jnp.mean(teacher_dist.kl_divergence(student_dist))

        
        # Compute KL divergence loss
        epsilon = 1e-8
        #kl_loss = jnp.mean(teacher_log_probs * (jnp.log(teacher_log_probs + epsilon) - jnp.log(student_log_probs + epsilon)))   
        kl_loss = jnp.mean(teacher_log_probs - student_log_probs)
   
        kl_losses.append(kl_loss)

        if step % 100 == 0:
            print(f"Step {step}: KL Loss = {kl_loss}")
            # wandb.log({'distill_step': step, 'kl_loss': kl_loss}) 
            # wandb.log({'distill_step': step + FLAGS.reset_interval, 'kl_loss': float(kl_loss)})
            wandb.log({'timestep': i + step, 'kl_loss': float(kl_loss)})
            wandb.log({'timestep': i + step, 'ana_kl_div': float(kl_loss)})

        # layer-wise parameter difference printouts during distillation, helping you verify 
        # that the student is actually changing over time and not stuck
        if step % 500 == 0:
            for key in teacher_agent.actor.params.keys():
                t = jnp.ravel(teacher_agent.actor.params[key])
                s = jnp.ravel(student_agent.actor.params[key])
                diff = jnp.linalg.norm(t - s)
                print(f"[Step {step}] Param L2 diff for layer {key}: {diff:.4f}")


        # Perform SAC update on student agent (Actor and Critic updates handled inside)
        student_agent.update(batch)
        
        # === Target Network Update (Soft Update for Stability) ===
        new_target_critic_params = jax.tree_util.tree_map(
            lambda student_param, teacher_param: FLAGS.tau * teacher_param + (1 - FLAGS.tau) * student_param,
            student_agent.target_critic.params,
            teacher_agent.target_critic.params
        )
        student_agent.target_critic = student_agent.target_critic.replace(params=new_target_critic_params)

    return student_agent

def main(_):
    num_distill = 0
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    wandb.login()
    # Initialize WandB
    wandb.init(
        project="M_2",
        name=f"{FLAGS.job_id}_TestCheckLoadingAgent_{FLAGS.env_name}_seed{FLAGS.seed}",
        config=FLAGS.flag_values_dict()
    )
    # Define metric to align all logs on the same x-axis
    wandb.define_metric("timestep")  # x-axis
    wandb.define_metric("eval_return", step_metric="timestep")
    wandb.define_metric("reset_event", step_metric="timestep")  # Marker

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env = make_env(FLAGS.env_name, FLAGS.seed, video_train_folder)
    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    all_kwargs = FLAGS.flag_values_dict()
    all_kwargs.update(all_kwargs.pop('config'))

    kwargs = dict(FLAGS.config)
    assert kwargs.pop('algo') == 'sac'
    updates_per_step = kwargs.pop('updates_per_step')
    replay_buffer_size = kwargs.pop('replay_buffer_size')

    agent = SACLearner(FLAGS.seed,
                        env.observation_space.sample()[np.newaxis],
                        env.action_space.sample()[np.newaxis], **kwargs)
    
    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or FLAGS.max_steps)
    

    eval_returns = []
    observation, done = env.reset(), False
    start_step = 1
    if FLAGS.load_agent_step > 0:
        # pdb.set_trace()
        print(f"[DEBUG] Replay buffer size: {replay_buffer.size}")
        start_step = FLAGS.load_agent_step
        load_path = os.path.join(FLAGS.save_dir, f'agent_step_{FLAGS.load_agent_step}')
        agent = load_agent(load_path,
                        SACLearner,
                        seed=FLAGS.seed,
                        observations=env.observation_space.sample()[np.newaxis],
                        actions=env.action_space.sample()[np.newaxis],
                        **kwargs)
        replay_buffer.load(os.path.join(load_path, 'buffer.pkl')) # will the 4 replay buffers load? Check when you run teh experiment. 
        print(f"[DEBUG] Loaded replay buffer size: {replay_buffer.size}")
       
    # pdb.set_trace()
    for i in tqdm.tqdm(range(start_step, FLAGS.max_steps + 1),
                    smoothing=0.1,
                    disable=not FLAGS.tqdm):
        timestep = i
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask, float(done),
                            next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False

        if i >= FLAGS.start_training:
            for _ in range(updates_per_step):
                batch = replay_buffer.sample(FLAGS.batch_size)
                agent.update(batch)
            info = agent.update(batch)
            print("[Step %d] Critic Loss: %.4f | Actor Entropy: %.4f | Alpha: %.4f" % (
                agent.step, info['critic_loss'], info['entropy'], info['temperature']
            ))

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)
            #
            eval_return = eval_stats['return']
            timestep = i
            #
            eval_returns.append(
                (timestep, eval_return))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                    eval_returns,
                    fmt=['%d', '%.1f'])
            wandb.log({'timestep': i, 'eval_return': eval_return})

        if FLAGS.resets and i % FLAGS.reset_interval == 0:
            # create a completely new agent
            agent = SACLearner(FLAGS.seed + i,
                               env.observation_space.sample()[np.newaxis],
                               env.action_space.sample()[np.newaxis], **kwargs)
            wandb.log({"timestep": i, "reset_event": 1})

        if i == FLAGS.save_agent_at:
            save_path = os.path.join(FLAGS.save_dir, f'agent_step_{i}')
            save_agent(agent, save_path)
            replay_buffer.save(os.path.join(save_path, 'buffer.pkl'))

    # Finish WandB run
    wandb.finish()
    
if __name__ == '__main__':
    app.run(main)
