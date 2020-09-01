import bandito.types as typ
from bandito.arms import Arm
from bandito.exceptions import TimeCanNotBeNegative


class Bandito:
    def __init__(self, arms: typ.List[Arm], t_max: int) -> None:
        if t_max < 0:
            raise TimeCanNotBeNegative(f"Time (t_max={t_max}) can not be negative!")

        self.arms = arms
        self.t_max = t_max

    def __repr__(self) -> str:
        name = self.__class__.__name__
        var_n_val = ", ".join([f"{var}={val}" for var, val in self.__dict__.items()])
        return f"{name}({var_n_val})"
