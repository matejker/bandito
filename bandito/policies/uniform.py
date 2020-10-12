import numpy as np
import math
from typing import Optional

import bandito.policies.exceptions as ex
from bandito.policies import Policy
from bandito.entities import PolicyPayload


class Uniform(Policy):
    def __init__(self, t_max: int, q: Optional[float] = None) -> None:
        self.q = q or t_max ** (2 / 3) * (4 * np.log(t_max)) ** (1 / 3) / (t_max * 4)

        if self.q > 1 or self.q < 0:
            raise ex.UniformPolicy(f"Cut parameter q={q} cannot be q < 0 or q > 1!")
        super().__init__(t_max)

    def __call__(self) -> PolicyPayload:
        m = len(self.arms)
        n = math.floor(self.q * self.t_max / m)
        reminder = math.ceil(self.t_max - n * m)
        a_best = (0, 0)

        # Exploration phase
        for i, a in enumerate(self.arms):
            # fmt: off
            self.a[i * n: i * (n + 1)] = [i] * n
            self.reward[i * n: i * (n + 1)] = a.x_temp[i * n: i * (n + 1)]
            a.x[i * n: i * (n + 1)] = a.x_temp[i * n: i * (n + 1)]
            s = np.sum(a.x_temp[i * n: i * (n + 1)])
            # fmt: on
            a_best = (i, s) if s > a_best[1] else a_best

        # Exploitation phase
        # fmt: off
        self.a[self.t_max - reminder:] = [a_best[0]] * reminder
        self.reward[self.t_max - reminder:] = self.arms[a_best[0]].x_temp[self.t_max - reminder:]
        self.arms[a_best[0]].x[self.t_max - reminder:] = self.arms[a_best[0]].x_temp[self.t_max - reminder:]
        # fmt: on

        mean_reward = np.array([self.arms[i].mu for i in self.a])
        t = np.arange(1, self.t_max + 1)

        return PolicyPayload(
            arms=self.a,
            reward=self.reward,
            regred=np.cumsum(self.get_best_arm.mu - mean_reward),
            realized_regred=np.cumsum(self.get_best_arm.mu - self.reward),
            mean_reward=mean_reward,
            expected_regred=t ** (2 / 3) * (len(self.arms) * np.log(t)) ** (1 / 3),
        )
