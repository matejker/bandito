import numpy as np

from bandito.arms.arm import Arm
import bandito.arms.exceptions as ex


class Uniform(Arm):
    """ An arm with Uniform distribution.

    Attributes:
        t_max: time horizon / total number of rounds
        a: lower bound
        b: upper bound
        **kwargs: kwargs

    Raises:
        UnifromBounds: when upper bound is smaller then lower bound
    """

    def __init__(self, t_max: int, a: float = 0, b: float = 1, **kwargs) -> None:
        if b < a:
            raise ex.UnifromBounds(f"Bounds have to be a={a} <= b={b}]")

        super().__init__(t_max, **kwargs)
        self.mu = (a + b) / 2
        self.x_temp = np.random.uniform(size=self.t_max, low=a, high=b)
