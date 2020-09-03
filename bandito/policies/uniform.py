import numpy as np
import math

import bandito.policies.exceptions as ex
from bandito.policies import Policy


class Uniform(Policy):
    def __init__(self, q: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        if q > 1 or q < 0:
            raise ex.UniformPolicy(f"Cut parameter q={q} cannot be q < 0 or q > 1!")
        self.q = q

    def __call__(self) -> dict:
        m = len(self.arms)
        n = math.floor(self.q * self.t_max / m)
        reminder = math.ceil(self.t_max - n * m)
        a_best = (0, 0)

        # Exploration phase
        for i, a in enumerate(self.arms):
            self.a[i * n: i * (n + 1)] = np.fill(n, i)  # fmt: no
            self.reward[i * n: i * (n + 1)] = a.x_temp[i * n: i * (n + 1)]  # fmt: no
            a.x[i * n: i * (n + 1)] = a.x_temp[i * n: i * (n + 1)]  # fmt: no
            s = np.sum(a.x_temp[i * n: i * (n + 1)])  # fmt: no
            a_best = (i, s) if s > a_best[1] else a_best

        # Exploitation phase
        self.a[self.t_max - reminder:] = np.fill(reminder, a_best[0])  # fmt: no
        self.reward[self.t_max - reminder:] = self.arms[a_best[0]].x_temp[self.t_max - reminder:]  # fmt: no
        self.arms[a_best[0]].x[self.t_max - reminder:] = self.arms[a_best[0]].x_temp[self.t_max - reminder:]  # fmt: no

        return {"arms": self.a, "reward": self.reward}
