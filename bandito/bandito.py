import bandito.types as typ
from bandito.arms import Arm
from bandito.policies import Policy
from bandito.exceptions import TimeCanNotBeNegative


class Bandito:
    def __init__(self, policy: Policy, arms: typ.List[Arm], t_max: int) -> None:
        if t_max < 0:
            raise TimeCanNotBeNegative(f"Time (t_max={t_max}) can not be negative!")

        self.arms = arms
        self.t_max = t_max
        self.policy = policy

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(Policy: {self.policy}, Arms: [{self.arms}])"
