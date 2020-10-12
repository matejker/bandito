import numpy as np
from typing import Optional

import bandito.arms.exceptions as ex


class Arm:
    """ Core and base arm class with arm data and parameters. An arm or action is a single event which bandits policy
    can play at each round.

    Attributes:
        t_max: time horizon / total number of rounds
        x: reward at time t
        x_avg: average reward (so far) at each time step t
        s: total number of observation (so far) at each time step t
        mu: (theoretical) mean
        n: total number of observation
        t: time step, t = 1, 2, ..., t_max

    Raises:
        TimeCanNotBeNegative: if t_max or t is negative
        TimeStepCanNotExceedTmax: when t is greater than t_max
        ObservationNumberDoesNotMatch: when x_temp is smaller size than t_max
    """

    def __init__(self, t_max: int, x_temp: np.array = None, t: int = 0) -> None:
        if t_max < 0:
            raise ex.TimeCanNotBeNegative(f"Time t_max={t_max} cannot be negative!")
        if t < 0:
            raise ex.TimeCanNotBeNegative(f"Time t_max={t} cannot be negative!")
        if t_max < t:
            raise ex.TimeStepCanNotExceedTmax(f"Time t={self.t} cannot be t > t_max={self.t_max}!")

        if x_temp and x_temp.size < t_max:
            raise ex.ObservationNumberDoesNotMatch(
                f"Input observation X_temp_i(t) does not match with t_max={self.t_max}, "
                f"given: {x_temp.size()}, expected: {t_max})!"
            )

        self.x_temp: np.array = x_temp or np.zeros(t_max)
        self.x_temp = self.x_temp[:t_max]
        self.t_max: int = t_max

        self.x: np.array = np.full(t_max, np.nan)
        self.x_avg: np.array = np.zeros(t_max)
        self.s: np.array = np.zeros(t_max)
        self.mu: float = 0
        self.n: int = 0

        self.t: int = t

    def __repr__(self) -> str:
        name = self.__class__.__name__
        allowed_atr = {"x", "x_avg", "s"}.intersection(set(self.__dict__.keys()))
        all_atr = self.__dict__
        var_n_val = ", ".join([f"{var}(t)={all_atr[var][self.t]}" for var in allowed_atr])
        var_n_val = var_n_val + f', mu={all_atr["mu"]}' if "mu" in self.__dict__.keys() else var_n_val
        var_n_val += f", t={self.t}"
        return f"{name}({var_n_val})"

    def __str__(self) -> str:
        return self.__class__.__name__

    def get_x_avg(self, t: Optional[int]) -> np.array:
        """ Counts arm mean so far for each time step t.

        Args:
            t: time step

        Returns:
            x_avg: an arm average so far
        """
        t = t or self.t
        if t < 0:
            raise ex.TimeCanNotBeNegative(f"Time t={t} cannot be negative!")
        if self.t_max < t:
            raise ex.TimeStepCanNotExceedTmax(f"Time t={self.t} cannot be t > t_max={self.t_max}!")

        self.x_avg = np.nanmean(self.x[: t + 1])
        return self.x_avg

    def get_s(self, t: Optional[int]) -> np.array:
        """ Counts number of observation so far for each time step t.

        Args:
            t: time step

        Returns:
            s: number of observations so far
        """
        t = t or self.t
        if t < 0:
            raise ex.TimeCanNotBeNegative(f"Time t={t} cannot be negative!")
        if self.t_max < t:
            raise ex.TimeStepCanNotExceedTmax(f"Time t={self.t} cannot be t > t_max={self.t_max}!")

        self.s[: t + 1] = np.nancumsum(self.x[: t + 1])
        return self.s


class UnknownArm(Arm):
    """ Unlike for Arms with known probability distribution, sometimes we do not know the arm underlying distribution.
    For such case we have a generic class which counts mean a arm avg.

    Attributes:
        mu: average over arm observation
    """

    def __init__(self, **kwargs) -> None:
        super().__int__(**kwargs)  # type: ignore
        self.mu = np.mean(self.x_temp)
