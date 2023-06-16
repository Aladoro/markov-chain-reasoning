import hydra
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import time
import logz

tfl = tf.keras.layers
tfm = tf.keras.models
tfo = tf.keras.optimizers
tfi = tf.keras.initializers
tfd = tfp.distributions
tfr = tf.keras.regularizers

MBPO_TARGET_ENT = {'Hopper-v2':-1, 'HalfCheetah-v2':-3, 'Walker2d-v2':-3, 'Ant-v2':-4, 'Humanoid-v2':-2, 'InvertedPendulum-v2': -0.05}


def run_exp(cfg, agent, replay_buffer, sampler):
    logz.configure_output_dir('data')
    mean_test_returns = []
    mean_test_std = []
    steps = []

    print(agent._cri)

    step_counter = 0
    logz.log_tabular('epoch', 0)
    logz.log_tabular('number_collected_observations', step_counter)
    print('Epoch {}/{} - total steps {}'.format(0, cfg.epochs, step_counter))
    start_training_time = time.time()
    evaluation_stats = sampler.evaluate(agent, cfg.test_runs_per_epoch, log=False)
    for k, v in evaluation_stats.items():
        logz.log_tabular(k, v)
    if cfg.save_training_statistics:
        agent.save_metrics_and_reset()
        for k, v in agent._latest_log_dict.items():
            logz.log_tabular(k, v)
    mean_test_returns.append(evaluation_stats['episode_returns_mean'])
    mean_test_std.append(evaluation_stats['episode_returns_std'])
    steps.append(step_counter)
    epoch_end_eval_time = time.time()
    evaluation_time = epoch_end_eval_time - start_training_time

    logz.log_tabular('training_time', 0.0)
    logz.log_tabular('evaluation_time', np.around(evaluation_time, decimals=3))
    logz.dump_tabular()
    for e in range(cfg.epochs):
        logz.log_tabular('epoch', e + 1)
        epoch_start_time = time.time()
        while step_counter < (e + 1) * cfg.steps_per_epoch:
            traj_data = sampler.sample_steps(agent, 0.1, n_steps=1)
            replay_buffer.add(traj_data)
            step_counter += traj_data['n']
            if step_counter > cfg.start_training:
                agent.train(replay_buffer, batch_size=cfg.batch_size,
                            n_updates=cfg.updates_per_step * traj_data['n'],
                            act_delay=cfg.actor_delay,
                            tar_delay=cfg.target_delay)
            elif step_counter == cfg.start_training:
                if cfg.training_catchup:
                    n_updates = cfg.updates_per_step * cfg.start_training
                else:
                    n_updates = cfg.updates_per_step * traj_data['n']
                agent.train(replay_buffer, batch_size=cfg.batch_size,
                            n_updates=n_updates,
                            act_delay=cfg.actor_delay,
                            tar_delay=cfg.target_delay)

        logz.log_tabular('number_collected_observations', step_counter)
        epoch_end_training_time = time.time()
        training_time = epoch_end_training_time - epoch_start_time
        print('Epoch {}/{} - total steps {}'.format(e + 1, cfg.epochs, step_counter))
        evaluation_stats = sampler.evaluate(agent, cfg.test_runs_per_epoch, log=False)
        for k, v in evaluation_stats.items():
            logz.log_tabular(k, v)
        if cfg.save_training_statistics:
            agent.save_metrics_and_reset()
            for k, v in agent._latest_log_dict.items():
                logz.log_tabular(k, v)
        mean_test_returns.append(evaluation_stats['episode_returns_mean'])
        mean_test_std.append(evaluation_stats['episode_returns_std'])
        steps.append(step_counter)
        epoch_end_eval_time = time.time()
        evaluation_time = epoch_end_eval_time - epoch_end_training_time
        logz.log_tabular('training_time', np.around(training_time, decimals=3))
        logz.log_tabular('evaluation_time', np.around(evaluation_time, decimals=3))
        logz.dump_tabular()
        if cfg.save_weights_every is not None:
            if (e + 1) % cfg.save_weights_every == 0:
                logz.save_tf_weights(agent, e + 1)
    total_training_time = time.time() - start_training_time
    print('Total training time: {}'.format(total_training_time))
    plt.errorbar(steps, mean_test_returns, mean_test_std)
    plt.xlabel('steps')
    plt.ylabel('returns')
    plt.show(block=False)
    return mean_test_returns

def instatiate_models(cfg):
    sampler = hydra.utils.instantiate(cfg.sampler)
    replay_buffer = hydra.utils.instantiate(cfg.replay_buffer)
    env = sampler._env

    action_size = env.action_space.shape[0]

    cfg.action_size = action_size
    cfg.action_scale = float(env.action_space.high[0])
    cfg.agent.target_entropy = MBPO_TARGET_ENT.get(cfg.task_name, -1 * action_size)

    agent = hydra.utils.instantiate(cfg.agent)
    obs = np.expand_dims(env.reset().astype('float32'), axis=0)
    agent(obs)
    agent.summary()
    return cfg, agent, replay_buffer, sampler

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    if cfg.gpu >= 0:
        GPU_TO_USE = cfg.gpu
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[GPU_TO_USE], 'GPU')
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[GPU_TO_USE],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=cfg.memory_limit)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                print(e)
    else:
        tf.config.set_visible_devices([], 'GPU')
    cfg, agent, replay_buffer, sampler = instatiate_models(cfg)
    run_exp(cfg, agent, replay_buffer, sampler)
    return cfg, agent, replay_buffer, sampler



if __name__ == '__main__':
    main()
