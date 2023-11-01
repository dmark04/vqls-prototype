from dataclasses import dataclass
from typing import List


@dataclass
class VQLSLog:
    values: List
    parameters: List

    def update(self, count, cost, parameters):  # pylint: disable=unused-argument
        self.values.append(cost)
        self.parameters.append(parameters)
