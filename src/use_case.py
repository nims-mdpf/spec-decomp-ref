import json
import os
import sys
import typing
from glob import glob

import domain.access
import domain.algorithm
import domain.controller
import domain.integrator
import domain.model
import domain.utility
import numpy as np
import pandas as pd
import shared.controller
import shared.presenter
import shared.utility
from domain.model import *


class Featurizer(shared.controller.InputBoundary):
    @shared.utility.args_type_check
    def result2output(
        self,
        series: typing.List[str],
        integrator: domain.integrator.Integrator,
        opt_parameters: np.ndarray,
    ) -> typing.List[shared.presenter.OutputData]:
        output_data = []
        for m, key in enumerate(series):
            integrator.forward(m, parameters=opt_parameters)
            output_data_m = shared.presenter.OutputData(
                name=key,
                x=integrator.models[m].x,
                y=integrator.models[m].y,
                f=integrator.models[m].f,
                b=integrator.models[m].b,
                peaks=integrator.models[m].peaks,
                parameters={
                    "peaks": integrator.models[m].output_parameters(
                        integrator.input_controller(
                            input_array=opt_parameters, m=m, K=integrator.K
                        )
                    ),
                    "background": integrator.models[
                        m
                    ].background_obj.output_parameters(),
                },
            )
            output_data.append(output_data_m)

        return output_data

    def set_model(self, input_type: str, **kwargs):
        try:
            return globals()[input_type](**kwargs)
        except KeyError as e:
            print(f"キーが見つかりません:{input_type}")
            sys.exit(0)

    @shared.utility.args_type_check
    def featurize(
        self,
        input_data: typing.Dict[str, shared.controller.InputData],
        domain_input: domain.controller.DomainInputData,
    ) -> domain.model.Model:
        # モデル系インスタンス生成
        peaks_obj = self.set_model(
            domain_input.peak_type, number_of_state=domain_input.K
        )
        background_obj = self.set_model(domain_input.background_type)
        noise_obj = self.set_model(domain_input.noise_type)
        regularizer_obj = self.set_model(domain_input.regularization_type)

        data_access = domain.access.DataAccess()
        reference_table = data_access.access()

        models = domain.utility.initialize(
            input_data,
            reference_table,
            domain_input.states,
            peaks_obj,
            background_obj,
            noise_obj,
            regularizer_obj,
        )

        # 制御系インスタンス生成
        # (インテグレータオブジェクト，コントロールオブジェクト生成)
        input_controller = domain.integrator.InputController(
            num_input_data=domain_input.M,
            global_idx=domain_input.common_index,
            local_idx=domain_input.uncommon_index,
            sg_param_size=peaks_obj.parameter_size,
            bg_param_size=background_obj.parameter_size,
        )
        integrator = domain.integrator.Integrator(
            models, input_controller, domain_input.K
        )

        # 最適化アルゴリズムのインスタンス生成
        differentia_evolution = domain.algorithm.DifferentiaEvolution(
            domain_input=domain_input, model=integrator
        )
        # 最適化の実行
        opt_parameters = differentia_evolution.optimize()

        # 出力データクラスへの変換
        series = input_data.keys()
        output_data = self.result2output(
            series=series,
            integrator=integrator,
            opt_parameters=opt_parameters,
        )

        return output_data


def decompose():
    path_list = sorted(glob("data/*.csv"))
    shared_controller = shared.controller.Controller()
    input_data = shared_controller.csv2input(path_list=path_list)

    config_path = "data/config.json"
    with open(config_path) as f:
        config = json.load(f)
    controller = domain.controller.Controller(config)
    domain_input = controller.config2input(input_data=input_data)

    featurizer = Featurizer()
    output_data = featurizer.featurize(input_data=input_data, domain_input=domain_input)
    # print("output_data = ", output_data)

    for output_data_m in output_data:
        presenter = shared.presenter.Presenter()
        presenter.output(output_data=output_data_m, states=domain_input.states)
