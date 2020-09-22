import numpy as np

import bandito.policies.exceptions as ex
from bandito.arms import Arm
import bandito.types as typ


class Policy:
    def __init__(self, t_max: int) -> None:
        if t_max < 0:
            raise ex.TimeCanNotBeNegative(f"Time t_max={t_max} cannot be negative!")

        self.t_max: int = t_max
        self.t: int = 0
        self.a: np.array = [0] * t_max
        self.reward: np.array = np.zeros(t_max)
        self.arms: typ.List[Arm] = []

    def __repr__(self) -> str:
        name = self.__class__.__name__
        allowed_atr = {"a", "reward"}.intersection(set(self.__dict__.keys()))
        all_atr = self.__dict__
        var_n_val = ", ".join([f"{var}(t)={all_atr[var][self.t]}" for var in allowed_atr])
        var_n_val += f", t={self.t}"
        return f"{name}({var_n_val})"

    def __str__(self) -> str:
        return self.__class__.__name__

    def __call__(self):
        return {"arms": self.a, "reward": self.reward}
