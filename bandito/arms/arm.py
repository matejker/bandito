import numpy as np
import bandito.types as typ
import bandito.arms.exceptions as ex


class Arm:
    def __init__(self, t_max: int, x_temp: np.array = None, t: int = 0) -> None:
        if t_max < 0:
            raise ex.TimeCanNotBeNegative(f"Time t_max={t_max} cannot be negative!")
        if t < 0:
            raise ex.TimeCanNotBeNegative(f"Time t_max={t} cannot be negative!")
        if t_max < t:
            raise ex.TimeStepCanNotExceedTmax(f"Time t={self.t} cannot be t > t_max={self.t_max}!")

        self.x_temp: np.array = x_temp or np.zeros(t_max)

        if self.x_temp.size != t_max:
            raise ex.ObservationNumberDoesNotMatch(
                f"Input observation X_temp_i(t) does not match with t_max={self.t_max}, "
                f"given: {self.x_temp.size()}, expected: {t_max})!"
            )

        self.t_max: int = t_max

        self.x: np.array = np.full(t_max, np.nan)
        self.x_avg: np.array = np.zeros(t_max)
        self.s: np.array = np.zeros(t_max)
        self.mu: float = 0

        self.t: int = t

    def __repr__(self) -> str:
        name = self.__class__.__name__
        allowed_atr = {"x", "x_avg", "s"}.intersection(set(self.__dict__.keys()))
        all_atr = self.__dict__
        var_n_val = ", ".join([f"{var}(t)={all_atr[var][self.t]}" for var in allowed_atr])
        var_n_val = var_n_val + f'mu={all_atr["mu"]}' if "mu" in self.__dict__.keys() else var_n_val
        return f"{name}({var_n_val})"

    def __str__(self) -> str:
        return self.__class__.__name__

    def get_x_avg(self, t: typ.Optional[int]) -> np.array:
        t = t or self.t
        if t < 0:
            raise ex.TimeCanNotBeNegative(f"Time t={t} cannot be negative!")
        if self.t_max < t:
            raise ex.TimeStepCanNotExceedTmax(f"Time t={self.t} cannot be t > t_max={self.t_max}!")

        self.x_avg = np.nanmean(self.x[:t])
        return self.x_avg

    def get_s(self, t: typ.Optional[int]) -> np.array:
        t = t or self.t
        if t < 0:
            raise ex.TimeCanNotBeNegative(f"Time t={t} cannot be negative!")
        if self.t_max < t:
            raise ex.TimeStepCanNotExceedTmax(f"Time t={self.t} cannot be t > t_max={self.t_max}!")

        self.s[:t] = np.nancumsum(self.x[:t])
        return self.s


class UnknownArm(Arm):
    def __init__(self, **kwargs) -> None:
        super().__int__(**kwargs)  # type: ignore
        self.mu = np.mean(self.x_temp)
