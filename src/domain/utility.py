import typing
import numpy as np
import pandas as pd
import json
import copy

import domain.model
from domain.adjustment import RefParamKeys


def initialize(data: typing.Dict, reference_table: pd.core.frame.DataFrame, states: typing.List,
               peaks_obj: domain.model.Signal, background_obj: domain.model.Signal,
               noise_obj, regularizer_obj):
    
    models = []
    for i, series in enumerate(data.keys()):    
        reference_array = []
        for j, state in enumerate(states):
            reference_table_subset = reference_table[
                # (reference_table[RefParamKeys.series.value]==series) & (reference_table[RefParamKeys.state.value]==state)
                (reference_table[RefParamKeys.series.value]==series.split('-')[0]) & 
                (reference_table[RefParamKeys.state.value]==state)
            ]

            if(len(reference_table_subset.index)==0):
                reference_table_subset = [[0.0, 0.0, 1.0, 1.0]] # ダミーリファレンス
            else:
                reference_table_subset = reference_table_subset.copy()
                rsum = reference_table_subset[RefParamKeys.r.value].sum()
                reference_table_subset[RefParamKeys.r.value] = reference_table_subset[RefParamKeys.r.value] / rsum
                reference_table_subset[RefParamKeys.r.value] = reference_table_subset[RefParamKeys.r.value] * reference_table_subset[RefParamKeys.A.value]
                reference_table_subset = reference_table_subset[[RefParamKeys.r.value, RefParamKeys.mu.value, RefParamKeys.w.value, RefParamKeys.u.value]]

            reference_array.append( np.array(reference_table_subset) )
        
        # オブジェクトを値渡し
        model = domain.model.Model(
            x = data[series].x,
            y = data[series].y,
            peaks_obj = copy.deepcopy(peaks_obj),
            background_obj = copy.deepcopy(background_obj),
            noise_obj = copy.deepcopy(noise_obj),
            regularizer_obj = copy.deepcopy(regularizer_obj),
            reference_array = reference_array,
        )
        models.append(model)

    return models
