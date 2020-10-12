from typing import List

from bandito.arms import Arm
from bandito.policies import Policy
from bandito.exceptions import TimeCanNotBeNegative
from bandito.entities import PolicyPayload


class Bandito:
    """ Core class defining a bandit. A bandit is defined by number of rounds, arms and policy.

    Example:
        policy = Uniform(t_max, q=q)
        arms = [
            Bernoulli(t_max, p=0.2, a=0.1, b=1.1),
            Bernoulli(t_max, p=0.2, a=0.5, b=1.5),
            Bernoulli(t_max, p=0.5, a=0, b=1),
            Bernoulli(t_max, p=0.7, a=-0.2, b=1.8)
        ]
        bandito = Bandito(policy_u, arms, t_max)
        bandito_run = bandito()

    Attributes:
        arms: List of (stochastic) arms
        t_max: time horizon / total number of rounds
        policy: A policy / algorithm which bandit looks for the most profitable arm
    """

    def __init__(self, policy: Policy, arms: List[Arm], t_max: int) -> None:
        if t_max < 0:
            raise TimeCanNotBeNegative(f"Time (t_max={t_max}) can not be negative!")

        self.arms: List[Arm] = arms
        self.t_max: int = t_max
        self.policy: Policy = policy

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(Policy: {self.policy}, Arms: [{self.arms}])"

    def __call__(self) -> PolicyPayload:
        self.policy.arms = self.arms
        return self.policy()
