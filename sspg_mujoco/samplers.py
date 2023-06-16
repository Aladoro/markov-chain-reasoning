import numpy as np
import copy
import cv2


def log_trajectory_statistics(trajectory_rewards, log=True, trajectory_evals=None):
    out = {}
    out['number_test_episodes'] = len(trajectory_rewards)
    out['episode_returns_mean'] = np.mean(trajectory_rewards)
    out['episode_returns_max'] = np.max(trajectory_rewards)
    out['episode_returns_min'] = np.min(trajectory_rewards)
    out['episode_returns_std'] = np.std(trajectory_rewards)
    if trajectory_evals is not None:
        out['episode_trajectory_evals_mean'] = np.mean(trajectory_evals)
        out['episode_trajectory_evals_max'] = np.max(trajectory_evals)
        out['episode_trajectory_evals_min'] = np.min(trajectory_evals)
        out['episode_trajectory_evals_std'] = np.std(trajectory_evals)
    if log:
        print('Number of completed trajectories - {}'.format(out['number_test_episodes']))
        print('Latest trajectories mean reward - {}'.format(out['episode_returns_mean']))
        print('Latest trajectories max reward - {}'.format(out['episode_returns_max']))
        print('Latest trajectories min reward - {}'.format(out['episode_returns_min']))
        print('Latest trajectories std reward - {}'.format(out['episode_returns_std']))
    return out


class Sampler(object):
    def __init__(self, env, eval_env=None, episode_limit=1000, init_random_samples=1000,
                 visual_env=False, stacked_frames=3, ims_channels=3):
        self._env = env
        self._eval_env = eval_env or copy.deepcopy(self._env)
        self._max_episode_steps = self._env._max_episode_steps
        self._visual_env = visual_env
        self._stacked_frames = stacked_frames
        self._ims_channels = ims_channels
        self._el = episode_limit
        self._nr = init_random_samples

        self._tc = 0
        self._ct = 0

        self._ob = None
        self._agent_feed = None
        self._reset = True

        self._curr_return = 0

        self._lat_returns = []

    def handle_ob(self, ob):
        return ob.astype('float32'), ob.astype('float32')

    def get_action(self, policy, noise_stddev, observation, ):
        agent_input = np.expand_dims(observation, axis=0)
        return np.array(policy.get_action(agent_input, noise_stddev=noise_stddev))[0]

    def get_action_prob(self, policy, observation):
        agent_input = np.expand_dims(observation, axis=0)
        action, logprobs = policy._act.get_action_probability(agent_input)
        return np.array(action)[0], np.array(logprobs)[0, 0]

    def sample_step(self, policy, noise_stddev):
        if self._reset or self._ct >= self._el:
            self._lat_returns.append(self._curr_return)
            self._curr_return = 0
            self._ct = 0
            self._reset = False
            first = True
            self._ob, self._agent_feed = self.handle_ob(self._env.reset())
        else:
            first = False
        if self._tc < self._nr:
            act = self._env.action_space.sample()
        else:
            act = self.get_action(policy=policy, noise_stddev=noise_stddev, observation=self._agent_feed, )
        ob = self._ob
        self._ob, rew, self._reset, info = self._env.step(act)
        self._curr_return += rew
        if self._visual_env:
            ims = self._ob['ims']
        self._ob, self._agent_feed = self.handle_ob(self._ob)
        nob = self._ob
        self._ct += 1
        if (self._ct == self._max_episode_steps) and self._reset:
            don = False
        else:
            don = self._reset
        self._tc += 1
        out = {'obs': ob, 'nobs': nob, 'act': act, 'rew': rew, 'don': don, 'first': first, 'n': 1}
        if self._visual_env:
            out['ims'] = ims
        return out

    def sample_steps(self, policy, noise_stddev, n_steps=1):
        obs, nobs, acts, rews, dones, firsts, visual_obs = [], [], [], [], [], [], []
        for i in range(n_steps):
            if self._reset or self._ct >= self._el:
                self._lat_returns.append(self._curr_return)
                self._curr_return = 0
                self._ct = 0
                self._reset = False
                firsts.append(True)
                self._ob, self._agent_feed = self.handle_ob(self._env.reset())
            else:
                firsts.append(False)
            if self._tc < self._nr:
                act = self._env.action_space.sample()
            else:
                act = self.get_action(policy=policy, noise_stddev=noise_stddev, observation=self._agent_feed)

            obs.append(self._ob)
            acts.append(act)
            self._ob, rew, self._reset, info = self._env.step(act)
            self._curr_return += rew
            if self._visual_env:
                visual_obs.append(self._ob['ims'])
            self._ob, self._agent_feed = self.handle_ob(self._ob)
            nobs.append(self._ob)
            rews.append(rew)
            self._ct += 1
            if (self._ct == self._max_episode_steps) and self._reset:
                dones.append(False)
            else:
                dones.append(self._reset)
            self._tc += 1
        out = {'obs': np.stack(obs), 'nobs': np.stack(nobs), 'act': np.stack(acts),
               'rew': np.array(rews), 'don': np.array(dones), 'first': np.array(firsts),
               'n': n_steps}
        if self._visual_env:
            out['ims'] = np.stack(visual_obs)
        return out

    def sample_trajectory(self, policy, noise_stddev):
        obs, nobs, acts, rews, dones, firsts, visual_obs = [], [], [], [], [], [], []
        ct = 0
        done = False
        first = True
        if self._reset:
            self._lat_returns.append(self._curr_return)
        ob, agent_feed = self.handle_ob(self._env.reset())
        self._curr_return = 0
        while not done and ct < self._el:
            if self._tc < self._nr:
                act = self._env.action_space.sample()
            else:
                act = self.get_action(policy=policy, noise_stddev=noise_stddev, observation=agent_feed)
            firsts.append(first)
            first = False
            obs.append(ob)
            acts.append(act)
            ob, rew, done, info = self._env.step(act)
            self._curr_return += rew
            if self._visual_env:
                visual_obs.append(ob['ims'])
            ob, agent_feed = self.handle_ob(ob)
            nobs.append(ob)
            rews.append(rew)
            ct += 1
            if (ct == self._max_episode_steps) and done:
                dones.append(False)
            else:
                dones.append(done)
            self._tc += 1
        self._reset = True
        out = {'obs': np.stack(obs), 'nobs': np.stack(nobs), 'act': np.stack(acts),
               'rew': np.array(rews), 'don': np.array(dones), 'first': np.array(firsts),
               'n': ct}
        if self._visual_env:
            out['ims'] = np.stack(visual_obs)
        return out

    def sample_test_trajectories(self, policy, noise_stddev, n=5, visualize=False, ):
        obs, nobs, acts, rews, dones, rets, ids, visual_obs = [], [], [], [], [], [], [], []
        for i in range(n):
            ret = 0
            ct = 0
            done = False
            ob, agent_feed = self.handle_ob(self._eval_env.reset())
            while not done and ct < self._el:
                if policy is not None:
                    act = self.get_action(policy=policy, noise_stddev=noise_stddev, observation=agent_feed, )
                else:
                    act = self._eval_env.action_space.sample()
                ob, rew, done, info = self._eval_env.step(act)
                if visualize:
                    current_im = self._eval_env.render(mode='rgb_array')
                    cv2.imshow('visualization', current_im)
                    cv2.waitKey(10)
                ob, agent_feed = self.handle_ob(ob)
                ids.append(i)
                ret += rew
                ct += 1
            rets.append(ret)
        if visualize:
            cv2.destroyAllWindows()
        return rets

    def sample_test_trajectories_rw_logprobs(self, policy, noise_stddev, compute_logprob, n=5, ):
        trajs = []
        for i in range(n):
            obs, nobs, acts, rews, visual_obs = [], [], [], [], []
            ct = 0
            if compute_logprob:
                logprobs = []
            done = False
            ob, agent_feed = self.handle_ob(self._eval_env.reset())
            while not done and ct < self._el:
                if policy is not None:
                    if not compute_logprob:
                        act = self.get_action(policy=policy, noise_stddev=noise_stddev, observation=agent_feed, )
                    else:
                        act, logprob = self.get_action_prob(policy=policy, observation=agent_feed)
                        logprobs.append(logprob)

                else:
                    act = self._eval_env.action_space.sample()
                obs.append(ob)
                acts.append(act)
                ob, rew, done, info = self._eval_env.step(act)
                if self._visual_env:
                    visual_obs.append(ob['ims'])
                ob, agent_feed = self.handle_ob(ob)
                nobs.append(ob)
                rews.append(rew)
                ct += 1
            if (ct == self._max_episode_steps):
                final_done = False
            else:
                final_done = done
            traj = {'obs': np.stack(obs), 'nobs': np.stack(nobs), 'act': np.stack(acts),
                    'rew': np.array(rews), 'final_don': final_done, 'n': ct, }
            if self._visual_env:
                traj['ims'] = np.stack(visual_obs)
            if compute_logprob:
                traj['lprobs'] = np.array(logprobs)
            trajs.append(traj)
        return trajs

    def evaluate(self, policy, n=10, log=True, ):
        print('Evaluating the agent\'s behavior')
        rets = self.sample_test_trajectories(policy, 0.0, n)
        eval_stats = log_trajectory_statistics(rets, log)
        if len(self._lat_returns) > 0:
            eval_stats['latest_train_return'] = self._lat_returns[-1]
        else:
            eval_stats['latest_train_return'] = 0.0
        return eval_stats
