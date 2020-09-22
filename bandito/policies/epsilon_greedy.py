import numpy as np

import bandito.policies.exceptions as ex
from bandito.policies import Policy
import bandito.types as typ


class EpsilonGreedy(Policy):
    def __init__(self, epsilon: typ.Optional[typ.Union[np.array, float]] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if epsilon and (epsilon > 1 or epsilon < 0):
            raise ex.EpsilonGreedyPolicy(f"Parameter epsilon={epsilon} cannot be epsilon < 0 or epsilon > 1!")

        self.epsilon = epsilon

    def __call__(self) -> dict:

        if isinstance(self.epsilon, float):
            self.epsilon = np.full(self.t_max, self.epsilon)
        elif not self.epsilon:
            t = np.arange(1, self.t_max + 1)
            self.epsilon = (t ** (-1 / 3)) * (len(self.arms) * np.log(t)) ** (1 / 3)

        rnd_arms = np.random.randint(low=0, high=len(self.arms), size=self.t_max)
        rnd_toss = np.random.uniform(size=self.t_max)
        a_best = (0, 0)

        for t in range(self.t_max):
            if rnd_toss[t] < self.epsilon[t]:
                i = rnd_arms[t]
            else:
                i = a_best[0]

            self.a[t] = i
            self.arms[i].x[t] = self.arms[i].x_temp[t]
            self.reward[t] = self.arms[i].x_temp[t]
            avg = self.arms[i].get_x_avg(t)

            a_best = (i, avg) if avg > a_best[1] else a_best

        return {"arms": self.a, "reward": self.reward}
