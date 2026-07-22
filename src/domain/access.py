import abc

import pandas as pd


class DataAccessInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def access(self) -> None:
        raise NotImplementedError()


class DataAccess(DataAccessInterface):
    def access(self):
        return pd.read_csv("db/reference.csv")
