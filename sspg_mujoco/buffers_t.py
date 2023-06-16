import numpy as np
import tensorflow as tf

tfd = tf.data

LEN_ACTION_SPACE = 1 # one action value
LEN_OBSV_SPACE = 3 

class ReplayBuffer(object):
    def __init__(self, buffer_size,):
        self.buffer_size = buffer_size
        self.states = np.zeros([self.buffer_size, LEN_OBSV_SPACE], dtype=np.float32)
        self.next_states = np.zeros([self.buffer_size, LEN_OBSV_SPACE], dtype=np.float32)
        self.actions = np.zeros([self.buffer_size, LEN_ACTION_SPACE], dtype=np.float32)
        self.next_actions = np.zeros([self.buffer_size, LEN_ACTION_SPACE], dtype=np.float32)
        self.rewards = np.zeros([self.buffer_size], dtype=np.float32)
        self.index = 0
        self.full = False

    def add(self, states, actions, next_states, next_actions, rewards):
        (self.states[self.index], next_states[self.index],
         self.actions[self.index], self.next_actions[self.index],
         self.rewards[self.index]) = states, actions, next_states, next_actions, rewards
        self.index += 1
        if self.index == self.buffer_size:
            self.index = 0
            self.full = True

    def get_batch(self, batch_size):
        if self.full:
            indices = np.random.randint(self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(self.index, size=batch_size)
        states, actions, next_states, next_actions, rewards = (self.states[indices], next_states[indices],
                                                               self.actions[indices], self.next_actions[indices],
                                                               self.rewards[indices])
        return states, actions, next_states, next_actions, rewards

    def gather_n_steps_indices_slow(self, indices, n):
        n_samples = indices.shape[0]
        obs = self.states[indices]
        nobses = np.empty((n_samples, n, LEN_OBSV_SPACE))
        acts = np.empty((n_samples, n, LEN_ACTION_SPACE))
        rews = np.empty((n_samples, n))
        dones = np.empty((n_samples, n))
        mask = np.empty((n_samples, n))
        if self.visual_data:
            imses = np.empty((n_samples, n, *self.ims_shape))
        for i in range(n):
            if i == 0:
                mask[:, 0] = 1 - self.first[indices]
            else:
                mask[:, i] = mask[:, i-1] * (1 - self.first[indices + i])
            nobses[:, i] = self.next_states[indices + i]
            acts[:, i] = self.actions[indices + i]
            rews[:, i] = self.rewards[indices + i]
            dones[:, i] = self.don[indices + i]
            if self.visual_data:
                imses[:, i] = self.ims[indices + i]
        imses, nimses = self.split_ims(imses)
        ims = imses[0]
        return {'obs': obs, 'nobses': nobses, 'acts': acts, 'rews': rews,
                'dones': dones, 'ims': ims, 'nimses': nimses, 'mask': mask}

    def gather_n_steps_indices(self, indices, n):
        n_samples = indices.shape[0]
        gather_ranges = np.stack([np.arange(indices[i], indices[i] + n)
                                  for i in range(n_samples)], axis=0) % self.buffer_size
        obs = self.states[indices]
        nobses = self.next_states[gather_ranges]
        acts = self.actions[gather_ranges]
        rews = self.rewards[gather_ranges]
        dones = self.don[gather_ranges]
        mask = 1 - self.first[gather_ranges]
        mask[0] = 1
        for i in range(n - 2):
            mask[:, i + 2] = mask[:, i + 1] * mask[:, i + 2]
        if self.visual_data:
            imses = self.ims[gather_ranges]
            imses, nimses = self.split_ims(imses)
            ims = imses[:, 0]
            return {'obs': obs, 'nobses': nobses, 'acts': acts, 'rews': rews,
                    'dones': dones, 'ims': ims, 'nimses': nimses, 'mask': mask}
        return {'obs': obs, 'nobses': nobses, 'acts': acts, 'rews': rews,
                'dones': dones, 'mask': mask}

    def gather_n_steps_actions(self, indices, n):
        n_samples = indices.shape[0]
        gather_ranges = np.stack([np.arange(indices[i], indices[i] + n)
                                  for i in range(n_samples)], axis=0)
        acts = self.actions[gather_ranges]
        mask = 1 - self.first[gather_ranges]
        return {'acts': acts, 'mask': mask}

    def get_n_steps_random_batch(self, batch_size, n):
        if self.full:
            indices = np.random.randint(self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(self.index, size=batch_size)
        return self.gather_n_steps_indices(indices, n)

    def get_n_steps_random_actions_batch(self, batch_size, n):
        if self.full:
            indices = np.random.randint(self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(self.index, size=batch_size)
        return self.gather_n_steps_actions(indices, n)

    def gather_n_steps_indices_for_representations(self, indices, n):
        assert self.visual_data
        n_samples = indices.shape[0]
        gather_ranges = np.stack([np.arange(indices[i], indices[i] + n)
                                  for i in range(n_samples)], axis=0)
        obs = self.states[indices]
        nobs = self.next_states[indices]
        acts = self.actions[gather_ranges]
        act = acts[:, 0]
        rew = self.rewards[indices]
        done = self.don[indices]
        mask = 1 - self.first[gather_ranges]
        for i in range(n - 1):
            mask[:, i + 1] = mask[:, i] * mask[:, i + 1]
        imses = self.ims[gather_ranges]
        imses, nimses = self.split_ims(imses)
        ims = imses[:, 0]
        nims = nimses[:, 0]
        return {'obs': obs, 'nobs': nobs, 'act': act, 'acts': acts, 'rew': rew,
                'done': done, 'ims': ims, 'nims': nims,'nimses': nimses, 'mask': mask}
