import numpy as np
from copy import deepcopy

from bandito.policies import Policy
from bandito.entities import PolicyPayload


class SuccessiveElimination(Policy):
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
                # print(max_lcb, i, self.arms[i].n, mu)
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
