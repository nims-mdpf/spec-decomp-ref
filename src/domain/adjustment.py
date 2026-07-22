from enum import Enum

import numpy as np

EPS = 1e-6  # ゼロ割の対策


class RefParamKeys(Enum):
    series = "series"
    state = "state"
    h = "h"
    A = "A"
    r = "area"
    mu = "position"
    w = "sigma"
    u = "gamma"


def preprocess_area_parameter(area: np.array):
    nonzero_area = np.maximum(0.0, area)  # ReLU関数
    area_ratio = nonzero_area / max(
        np.sum(nonzero_area), EPS
    )  # 面積cの総和を1にする処理
    return area_ratio


### 参照スペクトルの活用
# ----------------------------------------------------------------------------------------
def calc_reference(
    reference: np.ndarray, perturbation: np.ndarray, **kwargs
) -> np.ndarray:
    """参照データと調整パラメータを入力して，調整後のパラメータを出力する．
    Args:
        reference (np.ndarray): 参照データ
                    [
                        [
                            [A_11, p_11, w_11, u_11], [A_12, p_12, w_12, u_12],
                            [A_1l, p_1l, w_1l, u_1l], ..., [A_1L, p_1L, w_1L, u_1L] (k=1)
                        ],
                        ...
                        ...
                        [
                            [A_k1, p_k1, w_k1, u_k1], [A_k2, p_k2, w_k2, u_k2],
                            [A_kl, p_kl, w_1l, u_1l], ..., [A_1L, p_1L, w_1L, u_1L] (k=k)
                        ],
                        ....
                    ]
                    ## size is [K, L_k, 4]
                    ## A is "a * r"
        perturbation (np.ndarray): 最適化する調整パラメータ
                    [
                        c_1, c_2, c_k, ... , c_K,
                        m_1, m_2, m_k, ... , m_K,
                        s_1, s_2, s_k, ... , g_K,
                        g_1, g_2, g_k, ... , s_K
                    ]
                    ## size is 4*K
                    ## m -> mu, s -> sigma, g -> gamma
    Yields:
        Iterator[np.ndarray]: 調整後のパラメータ. size is ピークの個数
    """
    K = len(reference)  # 状態数（例えば，化合物種）
    perturbation[:K] = preprocess_area_parameter(perturbation[:K])
    index = np.nonzero(perturbation[:K])[0]  # nonzeroのインデックスだけでfor文

    for k in index:
        cr = (
            reference[k][:, 0] * perturbation[(0 * K) + k]
        )  # [A_k1, A_k2, ..., A_kL] * c_k
        pm = (
            reference[k][:, 1] + perturbation[(1 * K) + k]
        )  # [p_k1, p_k2, ..., p_kL] + mu_k
        sw = (
            reference[k][:, 2] * perturbation[(2 * K) + k]
        )  # [w_k1, w_k2, ..., w_kL] * sigma_k
        gu = (
            reference[k][:, 3] * perturbation[(3 * K) + k]
        )  # [u_k1, u_k2, ..., u_kL] * gamma_k

        output = np.hstack((cr, pm, sw, gu))

        yield output  # Yieldでfor分で状態ごとにピークパラメータを返す．
