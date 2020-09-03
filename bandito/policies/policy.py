import numpy as np

import bandito.policies.exceptions as ex
from bandito import Arm
import bandito.types as typ


class Policy:
    def __init__(self, t_max: int) -> None:
        if t_max < 0:
            raise ex.TimeCanNotBeNegative(f"Time t_max={t_max} cannot be negative!")

        self.t_max: int = t_max
        self.t: int = 0
        self.a: np.array = np.zeros(t_max, dtype=int)
        self.reward: np.array = np.zeros(t_max)
        self.arms: typ.List[Arm] = []

    def __repr__(self) -> str:
        name = self.__class__.__name__
        allowed_atr = {"x", "x_avg", "s"}.intersection(set(self.__dict__.keys()))
        all_atr = self.__dict__
        var_n_val = ", ".join([f"{var}(t)={all_atr[var][self.t]}" for var in allowed_atr])
        var_n_val = var_n_val + f'mu={all_atr["mu"]}' if "mu" in self.__dict__.keys() else var_n_val
        return f"{name}({var_n_val})"

    def __str__(self) -> str:
        return self.__class__.__name__
