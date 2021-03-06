import numpy as np

from bandito.arms.arm import Arm
import bandito.arms.exceptions as ex


class Exponential(Arm):
    """ An arm with Exponential distribution.

    Attributes:
        t_max: time horizon / total number of rounds
        beta: 1 / lambda
        **kwargs: kwargs

    Raises:
        ExponentialScale: if scale parameter beta is smaller that 0
    """

    def __init__(self, t_max: int, beta: float, **kwargs) -> None:
        if beta < 0:
            raise ex.ExponentialScale(f"Scale parameter beta has to be beta={beta} > 0 (beta = 1 / lambda)]")

        super().__init__(t_max, **kwargs)
        self.mu = beta
        self.x_temp = np.random.exponential(scale=beta, size=self.t_max)
