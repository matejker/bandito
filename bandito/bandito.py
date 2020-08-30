import bandito.types as typ
from bandito.arms import Arm
from bandito.exceptions import TimeCanBeNegative


class Bandito:
    def __init__(self, arms: typ.List[Arm], t_max: int) -> None:
        if t_max < 0:
            raise TimeCanBeNegative(f"Time (t_max={t_max}) can not be negative!")

        self.arms = arms
        self.t_max = t_max
