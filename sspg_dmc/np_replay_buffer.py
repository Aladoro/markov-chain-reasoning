import numpy as np
from replay_buffer import AbstractReplayBuffer


class EfficientReplayBuffer(AbstractReplayBuffer):

    def __init__(self, buffer_size, batch_size, nstep, discount, frame_stack,
                 data_specs=None):
        self.buffer_size = buffer_size
        self.data_dict = {}
        self.index = -1
        self.traj_index = 0
        self.frame_stack = frame_stack
        self._recorded_frames = frame_stack + 1
        self.batch_size = batch_size
        self.nstep = nstep
        self.discount = discount
        self.full = False
        self.discount_vec = np.power(discount, np.arange(nstep))
        self.next_dis = discount**nstep

    def _initial_setup(self, time_step,):
        self.index = 0
        self.obs_shape = list(time_step.observation.shape)
        self.ims_channels = self.obs_shape[0] // self.frame_stack
        self.act_shape = time_step.action.shape

        self.obs = np.zeros([self.buffer_size, self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8)
        self.act = np.zeros([self.buffer_size, *self.act_shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)
        self.ent = None

    def _initialize_entropy(self, ):
        self.ent = np.zeros([self.buffer_size], dtype=np.float32)

    def add_data_point(self, time_step, entropy=None):
        first = time_step.first()
        latest_obs = time_step.observation[-self.ims_channels:]
        if first:
            end_index = self.index + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.index:self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    self.full = True
                else:
                    self.obs[self.index:end_index] = latest_obs
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index:self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                self.obs[self.index:end_index] = latest_obs
                self.valid[self.index:end_invalid] = False
            self.index = end_index
            self.traj_index = 1
        else:
            np.copyto(self.obs[self.index], latest_obs)
            np.copyto(self.act[self.index], time_step.action)
            if entropy is not None:
                self.ent[self.index] = entropy
            self.rew[self.index] = time_step.reward
            self.dis[self.index] = time_step.discount
            self.valid[(self.index + self.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.nstep:
                self.valid[(self.index - self.nstep + 1) % self.buffer_size] = True
            self.index += 1
            self.traj_index += 1
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

    def add(self, time_step, entropy=None):
        if self.index == -1:
            self._initial_setup(time_step,)
        if entropy is not None and self.ent is None:
            self._initialize_entropy()
        self.add_data_point(time_step, entropy)

    def __next__(self, ):
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.gather_nstep_indices(indices)

    def gather_nstep_indices(self, indices):
        n_samples = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
                                  for i in range(n_samples)], axis=0) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack:] # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]

        all_rewards = self.rew[gather_ranges]

        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        nobs = np.reshape(self.obs[nobs_gather_ranges], [n_samples, *self.obs_shape])

        act = self.act[indices]
        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        if self.ent is not None:
            all_entropies = self.ent[gather_ranges]
            discounted_nstep_entropy_correction = np.sum(all_entropies[:, 1:] * self.discount_vec[1:], axis=1, keepdims=True)
            ret = (obs, act, rew, dis, nobs, discounted_nstep_entropy_correction)
        else:
            ret = (obs, act, rew, dis, nobs)
        return ret

    def gather_random_all_nstep_batch(self, ):
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.gather_all_nstep_indices(indices)

    def gather_all_nstep_indices(self, indices):
        n_samples = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
                                  for i in range(n_samples)], axis=0) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack:] # bs x nstep
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]

        all_rewards = self.rew[gather_ranges]

        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        all_obs = self.obs[all_gather_ranges] # bs x frame_stack + n_step x *obs_dims

        strides_list = list(all_obs.strides)

        new_shape = [n_samples, self.nstep+1, self.frame_stack, self.ims_channels, *self.obs_shape[1:]]
        new_strides = [*strides_list[:2], *strides_list[1:]]
        all_obs = np.lib.stride_tricks.as_strided(all_obs, shape=new_shape, strides=new_strides)
        all_obs = np.reshape(all_obs, [n_samples, self.nstep+1, *self.obs_shape])


        obs = all_obs[:, 0]
        nobs = all_obs[:, 1:]

        flat_nobs = np.reshape(nobs, [n_samples*self.nstep, *self.obs_shape])

        act = self.act[indices]
        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        ret = obs, act, rew, dis, flat_nobs
        return ret


    def update_entropy(self, indices, entropies):
        raise NotImplementedError

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index

    def get_train_and_val_indices(self, validation_percentage):
        all_indices = self.valid.nonzero()[0]
        num_indices = all_indices.shape[0]
        num_val = int(num_indices * validation_percentage)
        np.random.shuffle(all_indices)
        val_indices, train_indices = np.split(all_indices,
                                              [num_val])
        return train_indices, val_indices

    def get_obs_act_batch(self, indices):
        n_samples = indices.shape[0]
        obs_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i])
                                      for i in range(n_samples)], axis=0) % self.buffer_size
        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        act = self.act[indices]
        return obs, act



