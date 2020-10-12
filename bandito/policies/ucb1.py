import numpy as np

from bandito.policies import Policy
from bandito.entities import PolicyPayload


class UCB1(Policy):
    def __init__(self, t_max: int) -> None:
        super().__init__(t_max)

    def __call__(self) -> PolicyPayload:
        max_ucb = (0, 0)

        for i in range(len(self.arms)):
            t = i
            self.a[i] = t
            self.arms[i].x[t] = self.arms[i].x_temp[t]
            self.reward[t] = self.arms[i].x_temp[t]
            self.arms[i].n += 1

            mu = self.arms[i].get_x_avg(t)

            ucb = mu + np.sqrt(2 * np.log(self.t_max))
            max_ucb = (i, ucb) if max_ucb[1] < ucb else max_ucb
        print(t)

        while t < self.t_max:
            i = max_ucb[0]
            self.a[t] = i
            self.arms[i].x[t] = self.arms[i].x_temp[t]
            self.reward[t] = self.arms[i].x_temp[t]
            self.arms[i].n += 1

            mu = self.arms[i].get_x_avg(t)
            ucb = mu + np.sqrt(2 * np.log(self.t_max) / self.arms[i].n)
            max_ucb = (i, ucb)

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
