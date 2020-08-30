import numpy as np
import bandito.types as typ
from bandito.arms.exceptions import TimeCanBeNegative


class Arm:
    def __init__(self, t_max: int, t: typ.Optional[int]) -> None:
        if t_max < 0:
            raise TimeCanBeNegative(f"Time (t_max={t_max}) can not be negative!")
        if t < 0:
            raise TimeCanBeNegative(f"Time (t_max={t}) can not be negative!")

        self.t_max = t_max

        self.x = np.full(t_max, np.nan)
        self.x_avg = np.zeros(t_max)
        self.s = np.zeros(t_max)

        self.t = t

    def get_x_avg(self, t: typ.Optional[int]) -> np.array:
        t = t or self.t
        if t < 0:
            raise TimeCanBeNegative(f"Time (t_max={t}) can not be negative!")

        self.x_avg = np.nanmean(self.x[:t])
        return self.x_avg

    def get_s(self, t: typ.Optional[int] = None) -> np.array:
        t = t or self.t
        if t < 0:
            raise TimeCanBeNegative(f"Time (t_max={t}) can not be negative!")

        self.s[:t] = np.nancumsum(self.x[:t])
        return self.s


class UnknownArm(Arm):
    def __init__(self, **kwargs) -> None:
        super().__int__(**kwargs)  # type: ignore

    def get_distribution(self, t: typ.Optional[int]):
        pass


class KnownArm(Arm):
    def __init__(self, mu: float, **kwargs) -> None:
        super().__int__(**kwargs)  # type: ignore
        self.mu = mu
