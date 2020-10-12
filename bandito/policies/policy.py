import numpy as np
from typing import List

import bandito.policies.exceptions as ex
from bandito.arms import Arm
from bandito.entities import PolicyPayload


class Policy:
    """ Base policy class, a policy or algorithm tries to pick up most performing arm among all, by picking at each
    round a single arm based on well defined logic.

    Attributes:
        t_max: time horizon / total number of rounds
        t: time step, t = 1, 2, ..., t_max
        a: played arm at each time step t
        reward: reward at each time step t
        arms: list of arms

    Raises:
        TimeCanNotBeNegative: if t_max is negative
    """

    def __init__(self, t_max: int) -> None:
        if t_max < 0:
            raise ex.TimeCanNotBeNegative(f"Time t_max={t_max} cannot be negative!")

        self.t_max: int = t_max
        self.t: int = 0
        self.a: np.array = [0] * t_max
        self.reward: np.array = np.zeros(t_max)
        self.arms: List[Arm] = []

    def __repr__(self) -> str:
        name = self.__class__.__name__
        allowed_atr = {"a", "reward"}.intersection(set(self.__dict__.keys()))
        all_atr = self.__dict__
        var_n_val = ", ".join([f"{var}(t)={all_atr[var][self.t]}" for var in allowed_atr])
        var_n_val += f", t={self.t}"
        return f"{name}({var_n_val})"

    def __str__(self) -> str:
        return self.__class__.__name__

    def __call__(self) -> PolicyPayload:
        return PolicyPayload(
            arms=self.a,
            reward=self.reward,
            regred=np.zeros(self.t_max),
            realized_regred=np.zeros(self.t_max),
            mean_reward=np.zeros(self.t_max),
            expected_regred=np.zeros(self.t_max),
        )

    @property
    def get_best_arm(self):
        """ Select arm with the best (theoretical) mean.

        Returns:
            arms: best arm
        """
        mus = [arm.mu for arm in self.arms]
        max_index = max(range(len(self.arms)), key=mus.__getitem__)

        return self.arms[max_index]
