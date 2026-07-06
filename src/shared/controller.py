import os
import sys
import typing
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InputData:
    x: np.ndarray
    y: np.ndarray


import abc


class InputBoundary(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def featurize(
        self, input_data: typing.Dict[str, InputData], domain_input: typing.Dict
    ):
        raise NotImplementedError()


class Controller:
    def __init__(self) -> None:
        pass

    def csv2input(self, path_list) -> typing.Dict[str, InputData]:
        input_data: typing.Dict[str, InputData] = {}
        for path in path_list:
            series = os.path.basename(path).split(".")[0].split("_")[0]
            dat = np.loadtxt(path, delimiter=",")
            x = dat[:, 0]
            y = dat[:, 1]
            sorted_index = np.argsort(x)
            x = x[sorted_index]
            y = y[sorted_index]
            input_data[series] = InputData(x=x, y=y)
        return input_data


if __name__ == "__main__":
    data_path = "data/O1s.csv"
    controller = Controller()

    output = controller.csv2input(path=data_path)
    print(output)
