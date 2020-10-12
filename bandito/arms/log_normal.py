import numpy as np

from bandito.arms.arm import Arm


class LogNormal(Arm):
    """ An arm with Log-Normal distribution.

    Attributes:
        t_max: time horizon / total number of rounds
        mu: mean value
        sigma: variance
        **kwargs: kwargs
    """

    def __init__(self, t_max: int, mu: float = 0, sigma: float = 1, **kwargs) -> None:
        super().__init__(t_max, **kwargs)
        self.mu = mu
        self.x_temp = np.random.lognormal(mean=mu, sigma=sigma, size=self.t_max)
