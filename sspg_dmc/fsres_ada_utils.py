import torch as th


class StepsUpdater:
    def __init__(self, init, decay=0.99, device='cuda'):
        assert (decay < 1.0) and (decay > 0.0)
        self.moving_average = th.tensor(init, device=device)
        self.moving_deviation = th.tensor(0.0, device=device)
        self.decay = decay
        self.res = 1 - self.decay

    def update(self, sample):
        self.moving_average = (self.moving_average*self.decay + self.res*sample)
        deviation = th.abs(self.moving_average - sample)
        self.moving_deviation = self.moving_deviation*self.decay + self.res*deviation

    def get(self,):
        return {'updater_average': self.moving_average,
                'updater_deviation': self.moving_deviation}

    def get_steps_estimate(self, steps_estimate='mean_ceil'):
        if steps_estimate == 'mean_ceil':
            return th.ceil(self.moving_average).to(dtype=th.int32)
        if steps_estimate == 'mean_floor':
            return self.moving_average.to(th.int32)
        else:
            raise NotImplementedError

