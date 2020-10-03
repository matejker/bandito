from dataclasses import dataclass
import numpy as np


@dataclass
class PolicyPayload:
    arms: np.array
    reward: np.array
    regred: np.array
    realized_regred: np.array
    mean_reward: np.array
    expected_regred: np.array
