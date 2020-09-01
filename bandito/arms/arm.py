import numpy as np
import bandito.types as typ
import bandito.arms.exceptions as ex


class Arm:
    def __init__(self, t_max: int, t: typ.Optional[int] = 0) -> None:
        if t_max < 0:
            raise ex.TimeCanNotBeNegative(f"Time t_max={t_max} can not be negative!")
        if t < 0:
            raise ex.TimeCanNotBeNegative(f"Time t_max={t} can not be negative!")
        if t_max < t:
            raise ex.TimeStepCanNotExceedTmax(f"Time cannot t > t_max, t={t}, t_max={t_max}")

        self.t_max = t_max

        self.x = np.full(t_max, np.nan)
        self.x_avg = np.zeros(t_max)
        self.s = np.zeros(t_max)

        self.t = t

    def __repr__(self) -> str:
        name = self.__class__.__name__
        allowed_atr = {"x", "x_avg", "s"}.intersection(set(self.__dict__.keys()))
        all_atr = self.__dict__
        var_n_val = ", ".join([f"{var}={all_atr[var][self.t]}" for var in allowed_atr])
        var_n_val = var_n_val + f'mu={all_atr["mu"]}' if "mu" in self.__dict__.keys() else var_n_val
        return f"{name}({var_n_val})"

    def get_x_avg(self, t: typ.Optional[int]) -> np.array:
        t = t or self.t
        if t < 0:
            raise ex.TimeCanNotBeNegative(f"Time t={t} can not be negative!")
        if self.t_max < t:
            raise ex.TimeStepCanNotExceedTmax(f"Time cannot t > t_max, t={self.t}, t_max={self.t_max}")

        self.x_avg = np.nanmean(self.x[:t])
        return self.x_avg

    def get_s(self, t: typ.Optional[int]) -> np.array:
        t = t or self.t
        if t < 0:
            raise ex.TimeCanNotBeNegative(f"Time t={t} can not be negative!")
        if self.t_max < t:
            raise ex.TimeStepCanNotExceedTmax(f"Time cannot t > t_max, t={self.t}, t_max={self.t_max}")

        self.s[:t] = np.nancumsum(self.x[:t])
        return self.s


class UnknownArm(Arm):
    def __init__(self, x: np.array, **kwargs) -> None:
        super().__int__(**kwargs)  # type: ignore
        if x.size() != self.x.size():
            raise ex.ObservationNumberDoesNotMatch(
                f"Input observation X_i(t) does not match with t_max={self.t_max}, "
                f"given: {x.size()}, expected: {self.x.size()})"
            )
        self.x = x

    def get_distribution(self, t: typ.Optional[int]):
        pass


class KnownArm(Arm):
    def __init__(self, mu: float, **kwargs) -> None:
        super().__int__(**kwargs)  # type: ignore
        self.mu = mu
