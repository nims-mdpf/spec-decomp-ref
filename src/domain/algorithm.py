import abc
import typing
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution

import domain.controller as domain_controller
import domain.model as ModelClass


class AbstractOptimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def optimize(self) -> np.ndarray:
        raise NotImplementedError()

    def loss(self, parameters: np.array, *args):
        tmp_model = args[0]
        return tmp_model.evaluate(parameters=parameters)


class DifferentiaEvolution(AbstractOptimizer):
    def __init__(
        self, domain_input: domain_controller.DomainInputData, model: ModelClass.Model
    ) -> None:
        self.bounds = domain_input.bounds
        self.model = model

    def optimize(self) -> np.ndarray:
        self.result = differential_evolution(
            func=self.loss,
            bounds=self.bounds,
            seed=333,
            popsize=20,
            disp=True,
            updating="deferred",
            mutation=(0.2, 0.8),
            workers=20,
            args=(self.model,),
        )

        return self.result.x
