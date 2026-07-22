import json
import typing
from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class DomainInputData:
    bounds: typing.List[typing.Tuple]
    series: typing.List[str]
    states: typing.List[str]
    common_index: typing.List[int]
    uncommon_index: typing.List[int]
    peak_type: str
    background_type: str
    noise_type: str
    regularization_type: str

    @property
    def K(self):
        return len(self.states)

    @property
    def M(self):
        return len(self.series)


class ParameterName(Enum):
    K = "number_of_peaks"
    higth = "hight"
    position = "position"
    sigma = "sigma"
    gamma = "gamma"


class Controller:
    def __init__(self, config: typing.Dict) -> None:
        self.is_common: typing.Dict = config["is_common"]
        self.bounds_dict: typing.Dict = config["bounds"]
        self.states: typing.Dict = config["states"]
        self.funcion_type: typing.Dict = config["model"]

    def config2input(self, input_data):
        series = input_data.keys()
        bounds = self.get_bounds(input_data)

        common_index = self.get_indication_common_index()
        uncommon_index = self.get_indication_uncommon_index()
        return DomainInputData(
            series=series,
            states=self.states,
            bounds=bounds,
            common_index=common_index,
            uncommon_index=uncommon_index,
            peak_type=self.funcion_type["peak"],
            background_type=self.funcion_type["background"],
            noise_type=self.funcion_type["noise"],
            regularization_type=self.funcion_type["regularization"],
        )

    def get_indication_common_index(self):
        is_global = np.array(list(self.is_common.values()))
        return np.arange(len(is_global), dtype=int)[is_global]

    def get_indication_uncommon_index(self):
        is_global = np.array(list(self.is_common.values()))
        is_local = ~is_global
        return np.arange(len(is_global), dtype=int)[is_local]

    def get_bounds(self, input_data):
        series = input_data.keys()
        states = self.states

        M = len(series)
        K = len(states)

        global_bounds = []
        for key in self.bounds_dict.keys():
            if self.is_common[key]:
                for k in range(K):
                    global_bounds.append(self.bounds_dict[key])

        range_scale = 2
        local_bounds = []
        for m, key in enumerate(series):
            y = input_data[key].y
            y_max = np.max(y)
            bg_low = y[0]
            bg_high = y[-1]
            approx_signal_intensity = y_max - (bg_high + bg_low) / 2.0

            max_intensity = approx_signal_intensity + range_scale * np.sqrt(
                approx_signal_intensity
            )
            min_intensity = approx_signal_intensity - range_scale * np.sqrt(
                approx_signal_intensity
            )
            range_intensity = (min_intensity, max_intensity)

            local_bounds.append(range_intensity)
            for key in self.bounds_dict.keys():
                if not self.is_common[key]:
                    for k in range(K):
                        local_bounds.append(self.bounds_dict[key])

            # range_background = (bg_low-range_scale*np.sqrt(bg_low), bg_low+range_scale*np.sqrt(bg_low))
            range_background = (bg_low * 0.5, bg_low * 1.5)
            # print(f"bg_low_range = {range_background}")
            local_bounds.append(range_background)
            # range_background = (bg_high-range_scale*np.sqrt(bg_high), bg_high+range_scale*np.sqrt(bg_high))
            range_background = (bg_high * 0.5, bg_high * 1.5)
            # print(f"bg_high_range = {range_background}")
            local_bounds.append(range_background)

        bounds = global_bounds + local_bounds
        return bounds


if __name__ == "__main__":
    config_path = "data/config.json"
    with open(config_path) as f:
        config = json.load(f)
    controller = Controller(config)
    l_series = ["Ca2p", "O1s"]
    l_states = ["CaO", "CaCO3"]
    domain_input = controller.config2input(
        series=l_series, states=l_states, y=np.random.normal(loc=100, size=10)
    )
    # print(domain_input)
    print(domain_input.K)
    print(domain_input.M)
    print(len(domain_input.bounds))
    print(domain_input.bounds)
