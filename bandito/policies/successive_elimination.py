import numpy as np
from copy import deepcopy

from bandito.policies import Policy
from bandito.entities import PolicyPayload


class SuccessiveElimination(Policy):
    """ An adaptive algorithm alternate all active arms until a lower confidence bound of one arm excides upper
    confidence bounds of others.

    Confidence bounds are defined as follows:
     - LCB_i(t) := mu_i(t) - r_t(i)
     - UCB_i(t) := mu_i(t) + r_t(i)

     where mu_i(t) is average of arm i (so far) at time t, and radius at each time step t
     r_i(t) = sqrt(2 * log(t_max) / n_t(i)))$

    Attributes:
        t_max: time horizon / total number of rounds

    Algorithm:
        initiate: all arms are active
        for each round t = 1, 2,..., t_max:
            for each active arm:
                alternate all active arms
                deactivate all arms such that exists arm i' which UCB_t(i) < LCB_t(i')
    """

    def __init__(self, t_max: int) -> None:
        super().__init__(t_max)

    def __call__(self) -> PolicyPayload:
        active_arms = set(range(len(self.arms)))
        max_lcb = (0, 0)
        t = 0
        while t < self.t_max:
            active_arms_tmp = deepcopy(active_arms)
            for i in active_arms_tmp:
                self.a[t] = i
                self.arms[i].x[t] = self.arms[i].x_temp[t]
                self.reward[t] = self.arms[i].x_temp[t]
                self.arms[i].n += 1

                mu = self.arms[i].get_x_avg(t)
                ucb = mu + np.sqrt(2 * np.log(self.t_max) / self.arms[i].n)
                lcb = mu - np.sqrt(2 * np.log(self.t_max) / self.arms[i].n)
                max_lcb = (i, lcb) if max_lcb[0] == i or max_lcb[1] < lcb else max_lcb

                if ucb < max_lcb[1] and max_lcb[0] != i:
                    active_arms.remove(i)

                t += 1

        mean_reward = np.array([self.arms[i].mu for i in self.a])
        t = np.arange(1, self.t_max + 1)

        return PolicyPayload(
            arms=self.a,
            reward=self.reward,
            regred=np.cumsum(self.get_best_arm.mu - mean_reward),
            realized_regred=np.cumsum(self.get_best_arm.mu - self.reward),
            mean_reward=mean_reward,
            expected_regred=np.sqrt(len(self.arms) * t * np.log(self.t_max)),
        )
