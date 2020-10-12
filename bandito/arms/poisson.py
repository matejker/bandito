import numpy as np

from bandito.arms.arm import Arm
import bandito.arms.exceptions as ex


class Poisson(Arm):
    """ An arm with Poisson distribution.

    Attributes:
        t_max: time horizon / total number of rounds
        lam: lambda
        **kwargs: kwargs

    Raises:
        PoissonLambda: if lambda is smaller that 0
    """

    def __init__(self, t_max: int, lam: float, **kwargs) -> None:
        if lam <= 0:
            raise ex.PoissonLambda(f"Lambda has to be lambda={lam} => 0]")

        super().__init__(t_max, **kwargs)
        self.mu = lam
        self.x_temp = np.random.poisson(lam=lam, size=self.t_max)
