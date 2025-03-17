    import os
    import random
    import wandb
    import numpy as np
    import tqdm
    from absl import app, flags
    from ml_collections import config_flags

    from continuous_control.agents import SACLearner
    from continuous_control.datasets import ReplayBuffer
    from continuous_control.evaluation import evaluate
    from continuous_control.utils import make_env

    FLAGS = flags.FLAGS

    flags.DEFINE_string('exp', '', 'Experiment description (not actually used).')
    flags.DEFINE_string('env_name', 'quadruped-run', 'Environment name.')
    flags.DEFINE_string('save_dir', './out/', 'Logging dir.')
    flags.DEFINE_integer('seed', 0, 'Random seed.')
    flags.DEFINE_integer('eval_episodes', 10,
                        'Number of episodes used for evaluation.')
    flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
    flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
    flags.DEFINE_integer('max_steps', int(2e6), 'Number of training steps.')
    flags.DEFINE_integer('start_training', int(1e4),
                        'Number of training steps to start training.')
    flags.DEFINE_integer('reset_interval', int(2e5), 'Periodicity of resets.')
    flags.DEFINE_boolean('resets', False, 'Periodically reset the agent networks.')
    flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
    flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')

    flags.DEFINE_integer('distill_steps', 10000, '_') # 5000
    flags.DEFINE_integer('lambda_decay_rate', 1000, '_')
    flags.DEFINE_float('tau', 0.005, '')

    config_flags.DEFINE_config_file(
        'config',
        'configs/sac.py',
        'File path to the training hyperparameter configuration.',
        lock_config=False)

    def distill(teacher_agent, env, kwargs, replay_buffer):
        # Create student agent using SACLearner
        student_agent = SACLearner(
            FLAGS.seed + 1000,  # Ensure a different seed
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            **kwargs
        )
        kl_losses = []
        for step in range(FLAGS.distill_steps):
            # Sample batch from the replay buffer
            batch = replay_buffer.sample(64)  # TODO replace with batch_size
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
            
            # Compute KL divergence loss
            epsilon = 1e-8
            kl_loss = jnp.mean(teacher_log_probs * (jnp.log(teacher_log_probs + epsilon) - jnp.log(student_log_probs + epsilon)))      
            kl_losses.append(kl_loss)

            if step % 100 == 0:
                print(f"Step {step}: KL Loss = {kl_loss}")
                # wandb.log({'distill_step': step, 'kl_loss': kl_loss}) 
                wandb.log({'distill_step': step + FLAGS.reset_interval, 'kl_loss': float(kl_loss)})


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
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        wandb.login()
        # Initialize WandB
        wandb.init(
            project="M_1",
            name=f"Test_{FLAGS.env_name}_seed{FLAGS.seed}",
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
        for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                        smoothing=0.1,
                        disable=not FLAGS.tqdm):
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

            if i % FLAGS.eval_interval == 0:
                eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)
                #
                eval_return = eval_stats['return']
                timestep = info['total']['timesteps']
                #
                eval_returns.append(
                    (info['total']['timesteps'], eval_stats['return']))
                np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                        eval_returns,
                        fmt=['%d', '%.1f'])
                wandb.log({'timestep': timestep, 'eval_return': eval_return})
            if FLAGS.resets and i % FLAGS.reset_interval == 0:
                # create a completely new agent
                # agent = SACLearner(FLAGS.seed + i,
                #                 env.observation_space.sample()[np.newaxis],
                #                 env.action_space.sample()[np.newaxis], **kwargs)
                agent = distill(agent, env, kwargs, replay_buffer)
                wandb.log({"timestep": i, "reset_event": 1})

        # Finish WandB run
        wandb.finish()
        
    if __name__ == '__main__':
        app.run(main)
