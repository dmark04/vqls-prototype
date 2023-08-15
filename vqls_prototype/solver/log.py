from dataclasses import dataclass
from typing import Optional, Union, List, Callable, Dict, Tuple
import numpy as np

@dataclass
class VQLSLog:
    values: List
    parameters: List

    def update(self, count, cost, parameters):
        self.values.append(cost)
        self.parameters.append(parameters)