import os
import sys
import typing

import numpy as np
from scipy.special import gammaln, voigt_profile

import domain.adjustment

EPS = 1e-6  # ゼロ割の対策


### モデルクラス
# ----------------------------------------------------------------------------------------
class Model:
    def __init__(
        self,
        x,
        y,
        peaks_obj,
        background_obj,
        noise_obj,
        regularizer_obj,
        reference_array,
    ) -> None:
        self.x = x
        self.y = y
        self.set_model(
            peaks_obj, background_obj, noise_obj, regularizer_obj, reference_array
        )

    def set_model(
        self, peaks_obj, background_obj, noise_obj, regularizer_obj, reference_array
    ) -> None:
        self.reference = reference_array
        self.peaks_obj = peaks_obj
        self.background_obj = background_obj
        self.noise_obj = noise_obj
        self.regularizer_obj = regularizer_obj

    def forward(self, parameters: np.ndarray) -> np.ndarray:
        perturbation_parameters = parameters[: self.peaks_obj.parameter_size]
        h = perturbation_parameters[0]
        delta = perturbation_parameters[1:]

        s = np.zeros(len(self.x))
        self.peaks = []
        # 状態kのパラメータごとにfor文を回す．パラメータはself.reference.callのyieldで返ってくる．
        for signal_parameters in domain.adjustment.calc_reference(
            self.reference, delta
        ):
            peak = self.peaks_obj.forward(self.x, signal_parameters)
            s += peak
            self.peaks.append(peak)

        self.peaks = np.array(self.peaks)
        # シグナルの最大強度をhにする
        scale = h / max(np.max(s), EPS)
        self.peaks *= scale
        s *= scale

        background_parameters = parameters[self.peaks_obj.parameter_size :]
        b = self.background_obj.forward(self.x, background_parameters, signal=s)
        self.b = b

        f = s + b
        self.f = f

        return f

    def evaluate(self, parameters: np.ndarray) -> float:
        f = self.forward(parameters)
        loss = self.noise_obj.evaluate(self.y, f)
        regularization = self.regularizer_obj.evaluate(obj=self)
        return loss + regularization

    def output_parameters(self, parameters: np.ndarray):
        perturbation_parameters = parameters[: self.peaks_obj.parameter_size]
        h = perturbation_parameters[0]
        delta = perturbation_parameters[1:]

        K = len(self.reference)
        output_param = {
            domain.adjustment.RefParamKeys.h.value: h,
            domain.adjustment.RefParamKeys.r.value: [],
            domain.adjustment.RefParamKeys.mu.value: [],
            domain.adjustment.RefParamKeys.w.value: [],
            domain.adjustment.RefParamKeys.u.value: [],
        }

        area = np.array([delta[(0 * K) + k] for k in range(K)])
        for k in range(K):
            # r = delta[(0*K)+k]
            m = delta[(1 * K) + k]
            w = delta[(2 * K) + k]
            u = delta[(3 * K) + k]

            # output_param[domain.adjustment.RefParamKeys.r.value].append(a)
            output_param[domain.adjustment.RefParamKeys.mu.value].append(m)
            output_param[domain.adjustment.RefParamKeys.w.value].append(w)
            output_param[domain.adjustment.RefParamKeys.u.value].append(u)

        r = domain.adjustment.preprocess_area_parameter(area)
        output_param[domain.adjustment.RefParamKeys.r.value] = list(r)

        return output_param


import abc


class Signal(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def output_parameters(self) -> typing.Dict:
        raise NotImplementedError()


### ピーク関数のクラス
# ----------------------------------------------------------------------------------------
class Peaks(Signal):
    PEAK_PARAM_SIZE: int = 4  # [c, mu, s, g]

    def __init__(self, number_of_state) -> None:
        self.number_of_state = number_of_state
        self.parameter_size = (
            self.PEAK_PARAM_SIZE * self.number_of_state + 1
        )  # [c, p, s, g] * K + h
        self.c = np.array([])

    @property
    def freedom_degree(self):
        # num_nonzero = np.count_nonzero(self.c)
        num_nonzero = len(self.c)
        freedom_degree = self.PEAK_PARAM_SIZE * num_nonzero + 1
        return freedom_degree

    def initialize_parameters(self, K):
        # K: ピークの個数
        self.c = np.array([np.nan] * K)
        self.p = np.array([np.nan] * K)
        self.s = np.array([np.nan] * K)
        self.g = np.array([np.nan] * K)

    def array2parameters(self, array: np.ndarray):
        # K: ピークの個数
        K = int(len(array) // self.PEAK_PARAM_SIZE)
        self.initialize_parameters(K)
        for k in range(K):
            self.c[k] = array[0 * K + k]
            self.p[k] = array[1 * K + k]
            self.s[k] = array[2 * K + k]
            self.g[k] = array[3 * K + k]

    def output_parameters(self):
        return {}

    def forward(self, x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        self.array2parameters(parameters)
        K = int(len(parameters) // self.PEAK_PARAM_SIZE)  # ピークの個数
        f = np.zeros(len(x))
        self.peaks = []
        for k in range(K):
            peak = self.h[k] * voigt_profile(x - self.p[k], self.s[k], self.g[k])
            self.peaks.append(peak)
            f += peak
        self.peaks = np.array(self.peaks)
        return f


class VoigtPeaks(Peaks):
    def forward(self, x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        self.array2parameters(parameters)
        K = int(len(parameters) // self.PEAK_PARAM_SIZE)  # ピークの個数
        f = np.zeros(len(x))
        self.peaks = []
        for k in range(K):
            peak = self.c[k] * voigt_profile(x - self.p[k], self.s[k], self.g[k])
            self.peaks.append(peak)
            f += peak
        self.peaks = np.array(self.peaks)
        return f


class GaussianPeaks(Peaks):
    def forward(self, x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        self.array2parameters(parameters)
        K = int(len(parameters) // self.PEAK_PARAM_SIZE)  # ピークの個数
        f = np.zeros(len(x))
        self.peaks = []
        for k in range(K):
            peak = self.c[k] * voigt_profile(x - self.p[k], self.s[k], 0.0)
            self.peaks.append(peak)
            f += peak
        self.peaks = np.array(self.peaks)
        return f


class LorentzianPeaks(Peaks):
    def forward(self, x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        self.array2parameters(parameters)
        K = int(len(parameters) // self.PEAK_PARAM_SIZE)  # ピークの個数
        f = np.zeros(len(x))
        self.peaks = []
        for k in range(K):
            peak = self.c[k] * voigt_profile(x - self.p[k], 0.0, self.g[k])
            self.peaks.append(peak)
            f += peak
        self.peaks = np.array(self.peaks)
        return f


class pseudoVoigtPeaks(Peaks):
    def pseudo_voigt_function(self, x, c, w, r):
        G = np.power(2.0, -np.power((x - c) / w, 2))
        L = 1.0 / (1.0 + np.power((x - c) / w, 2))
        return r * L + (1.0 - r) * G

    def forward(self, x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        self.array2parameters(parameters)
        K = int(len(parameters) // self.PEAK_PARAM_SIZE)  # ピークの個数
        f = np.zeros(len(x))
        self.peaks = []
        for k in range(K):
            peak = self.c[k] * self.pseudo_voigt_function(
                x, self.p[k], self.s[k], self.g[k]
            )
            self.peaks.append(peak)
            f += peak
        self.peaks = np.array(self.peaks)
        return f


# バックグラウンドクラス
# ----------------------------------------------------------------------------------------------
# import bgrm
# class SakuraiBackground(Signal):
#     def forward(self, x: np.ndarray, parameters: np.ndarray, **kwargs) -> np.ndarray:
#         y = kwargs['signal']
#         input = np.vstack((x, y)).T
#         return bgrm.linear(input)[:, 2]

#     def output_parameters(self):
#         return {}


class LinearBackground(Signal):
    BG_PARAM_SIZE: int = 2
    parameter_size: int

    def __init__(self) -> None:
        self.parameter_size = self.BG_PARAM_SIZE

    def initialize_parameters(self):
        self.a = None
        self.b = None

    def array2parameters(self, parameters: np.ndarray, **kwargs):
        self.initialize_parameters()
        self.a = parameters[0]
        self.b = parameters[1]

    def output_parameters(self):
        return {
            "a": self.a,
            "b": self.b,
        }

    def forward(self, x: np.ndarray, parameters: np.ndarray, **kwargs) -> np.ndarray:
        self.array2parameters(parameters)
        t = (self.a - self.b) / (x[0] - x[-1])
        bg = t * (x - x[0]) + self.a
        return bg


class ShirleyBackground:
    BG_PARAM_SIZE: int = 2
    parameter_size: int

    def __init__(self) -> None:
        self.parameter_size = self.BG_PARAM_SIZE

    def initialize_parameters(self):
        self.a = None
        self.b = None

    def array2parameters(self, parameters: np.ndarray, **kwargs):
        self.initialize_parameters()
        self.b = parameters[0]
        self.a = parameters[1]

    def output_parameters(self):
        return {
            "a": self.a,
            "b": self.b,
        }

    def forward(self, x: np.ndarray, parameters: np.ndarray, **kwargs) -> np.ndarray:
        """バックグラウンド（シャーリー法）

        Args:
            x (np.ndarray): エネルギー. ex) [529.0, 529.1, 529.2, 529.3, .....]
            parameters (np.ndarray): BGパラメータ（size is 2）

        Returns:
            np.ndarray: バックグラウンド
        """
        self.array2parameters(parameters)
        signal = kwargs["signal"]
        all_area = max(np.sum(signal), EPS)
        cumsum_area = np.cumsum(signal)

        bg = (self.a - self.b) * (cumsum_area / all_area) + self.b

        return bg


class ZeroPositionPeakBackground:
    BG_PARAM_SIZE: int = 2
    parameter_size: int

    def __init__(self) -> None:
        self.parameter_size = self.BG_PARAM_SIZE

    def initialize_parameters(self):
        self.h = None
        self.s = None

    def array2parameters(self, parameters: np.ndarray, **kwargs):
        self.initialize_parameters()
        self.h = parameters[0]
        self.s = parameters[1]

    def output_parameters(self):
        return {
            "h": self.h,
            "s": self.s,
        }

    def forward(self, x: np.ndarray, parameters: np.ndarray, **kwargs) -> np.ndarray:
        self.array2parameters(parameters)
        bg = h * np.exp(-(x**2) / (2 * s**2))
        return bg


# ノイズクラス
# ----------------------------------------------------------------------------------------------
class GaussianNoise:
    def evaluate(self, y: np.ndarray, f: np.ndarray, **kwargs):
        """ノイズモデル（ガウス）

        Args:
            y (np.ndarray): 観測強度
            f (np.ndarray): フィッティング強度

        Returns:
            float: 負の対数尤度
        """
        return np.sum((y - f) ** 2)


class PoissonNoise:
    def evaluate(self, y: np.ndarray, f: np.ndarray, **kwargs):
        """ノイズモデル（ポアソン）

        Args:
            y (np.ndarray): 観測強度
            f (np.ndarray): フィッティング強度

        Returns:
            float: 負の対数尤度
        """
        return np.sum(-y * np.log(f) + f + gammaln(y + 1))


class LaplaceNoise:
    def evaluate(self, y: np.ndarray, f: np.ndarray, **kwargs):
        return np.sum(abs(y - f))


class PoissonL0RegularizationNoise(PoissonNoise):
    def evaluate(self, y: np.ndarray, f: np.ndarray, **kwargs):
        cost = super().evaluate(y, f)
        freedom_degree = kwargs["freedom_degree"]
        N = len(y)
        return cost + 2.0 * N + freedom_degree


class GaussianL0RegularizationNoise(GaussianNoise):
    def evaluate(self, y: np.ndarray, f: np.ndarray, **kwargs):
        cost = super().evaluate(y, f)
        freedom_degree = kwargs["freedom_degree"]
        N = len(y)
        return cost + 2.0 * N + freedom_degree


# 正則化クラス
# ----------------------------------------------------------------------------------------------
class Regularizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self) -> float:
        raise NotImplementedError()


class BayesianInformationCriterion(Regularizer):
    def evaluate(self, obj: Model):
        N = len(obj.x)
        freedom_degree = obj.peaks_obj.freedom_degree
        return self.calcBIC(N=N, freedom_degree=freedom_degree)

    def calcBIC(self, N: int, freedom_degree: int, **kwargs):
        return 2.0 * N + freedom_degree


class NotRegularization(Regularizer):
    def evaluate(self, **kwargs):
        return 0.0
