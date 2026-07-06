import abc
import json
import os
import sys
import typing
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.ticker import ScalarFormatter
from scipy import integrate
from scipy.special import gammaln

plt.rcParams["font.size"] = 16
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["axes.grid"] = False
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 1.0
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.top"] = True


def calc_poisson_error(ycalc, yexp):
    """
    Calculate the Poisson error between calculated values (ycalc) and experimental values (yexp).

    Parameters:
    - ycalc (numpy.ndarray): Array of calculated values.
    - yexp (numpy.ndarray): Array of experimental values.

    Returns:
    Tuple[float, numpy.ndarray]: A tuple containing the total error (err) and the array of individual errors (errdata).
    """
    errdata = ycalc - yexp * np.log(ycalc) + gammaln(yexp + 1)
    err = errdata.sum()
    return err, errdata


@dataclass(frozen=True)
class OutputData:
    name: str
    x: np.ndarray
    y: np.ndarray
    f: np.ndarray
    b: np.ndarray
    peaks: np.ndarray
    parameters: typing.Dict


class OutputBoundary(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def output(self, output_data: OutputData) -> None:
        raise NotImplementedError()


class Presenter(OutputBoundary):
    def __init__(self) -> None:
        os.makedirs("out/", exist_ok=True)

    def output(self, output_data: OutputData, states=None):
        self.save_fitting(output_data=output_data)
        self.save_fitting_image(output_data=output_data, states=states)
        self.save_parameters(output_data=output_data)

    def save_fitting(self, output_data: OutputData):
        import pandas as pd

        fitting = pd.DataFrame()
        fitting["x"] = output_data.x
        fitting["y"] = output_data.y
        fitting["f"] = output_data.f
        fitting["b"] = output_data.b
        K = output_data.peaks.shape[0]
        for k in range(K):
            fitting[f"k{k}"] = output_data.peaks[k, :]
        fitting.to_csv(f"out/fitting_{output_data.name}.csv")

    def save_parameters(self, output_data: OutputData):
        with open(f"out/parameters_{output_data.name}.json", "w") as f:
            json.dump(output_data.parameters, f, indent=4)

    def save_fitting_image_color(self, output_data: OutputData, states=None):
        r_list = output_data.parameters["peaks"]["area"]
        num_candidate = len(r_list)
        # 色のリスト，物質名のラベルリストを作成．
        # r_listに基づき有限の面積強度を持つものを抽出．
        colors = [f"C{k}" for k in range(num_candidate)]
        if states is None:
            peakslabels = [f"material_{id + 1}" for id in range(num_candidate)]
        else:
            peakslabels = states
        bool_list = [(r > 0.0) for r in r_list]
        colors = np.array(colors)[bool_list]
        peakslabels = np.array(peakslabels)[bool_list]

        print(colors, peakslabels)
        num_pickup = len(peakslabels)
        # フィッティングデータの要素数との整合性チェック
        if output_data.peaks.shape[0] != num_pickup:
            print(
                "Error: length mismatch", output_data.peaks.shape[0], len(peakslabels)
            )

        # ax はフィッティングの図および残差，ax3 はポアソンコスト
        gskw = {"height_ratios": [6, 1], "hspace": 0.05}
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw=gskw)
        ax, ax3 = axes

        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        x = output_data.x
        ax.scatter(
            x,
            output_data.y,
            color="white",
            edgecolor="k",
            s=15,
            label="spectrum",
            zorder=1,
        )
        ax.plot(x, output_data.b, color="gray", zorder=1)
        ax.plot(x, output_data.f, color="r", ls="--", label="fitted", zorder=3)

        for k in range(num_pickup):
            y = output_data.peaks[k, :] + output_data.b
            ax.plot(x, y, color=colors[k], lw=2.5, label=peakslabels[k], zorder=2)

        # Add residue with its baseline
        res_baseline = (
            output_data.f.min() - (output_data.f.max() - output_data.f.min()) * 0.1
        )
        res_baseline = max(0, res_baseline)
        ax.axhline(res_baseline, ls=":", color="gray")
        y = output_data.y - output_data.f + res_baseline
        ax.plot(x, y, color="black", ls="-", label="residue", zorder=3)

        ax.legend(loc="upper right", fontsize=14)

        str_area_list = [f"{v * 100:.1f}" for v in r_list]
        title = "Area ratio = " + ", ".join(str_area_list)
        print(title)
        ax.set_title(title, fontsize=16)
        ax.set_ylabel("Intensity")

        cost_python, poisson_cost = calc_poisson_error(output_data.f, output_data.y)
        # ax3.plot(x, poisson_cost, "-", c="C4", ms=2, lw=1, label="cost")
        barwidth = np.abs((x[1] - x[0]) * 1.0)
        ax3.bar(x, poisson_cost, width=barwidth, color="C4", label="cost")
        # ax3.set_yscale("log")
        # ax3.set_ylim(1, 100)
        # ax3.set_yticks([1, 10])
        ax3.set_ylim(0, 50)
        ax3.set_ylabel("Cost", fontsize=16)

        xlim = (x.max(), x.min())
        ax3.set_xlim(xlim)
        ax3.set_xlabel("Binding energy (eV)")

        # fig.tight_layout()
        figfile = f"out/fitting_{output_data.name}.png"
        fig.savefig(figfile, bbox_inches="tight")
        plt.close("all")

    def save_fitting_image(self, output_data: OutputData, states=None):
        r_list = output_data.parameters["peaks"]["area"]
        num_candidate = len(r_list)
        # 色のリスト，物質名のラベルリストを作成．
        # r_listに基づき有限の面積強度を持つものを抽出．
        if states is None:
            peakslabels = [f"material_{id + 1}" for id in range(num_candidate)]
        else:
            peakslabels = states
        bool_list = [(r > 0.0) for r in r_list]
        peakslabels = np.array(peakslabels)[bool_list]

        colors_array = ["C0", "C1", "C2", "C3"]
        linestyle_array = [":", "--", "-.", "-"]
        colors_repeated = (colors_array * ((num_candidate // len(colors_array)) + 1))[
            :num_candidate
        ]
        linestyle_repeated = (
            linestyle_array * ((num_candidate // len(linestyle_array)) + 1)
        )[:num_candidate]
        colors = np.array(colors_repeated)[bool_list]
        linestyles = np.array(linestyle_repeated)[bool_list]

        # print(colors, peakslabels)

        num_pickup = len(peakslabels)
        # フィッティングデータの要素数との整合性チェック
        if output_data.peaks.shape[0] != num_pickup:
            print(
                "Error: length mismatch", output_data.peaks.shape[0], len(peakslabels)
            )

        # ax はフィッティングの図および残差，ax3 はポアソンコスト
        gskw = {"height_ratios": [6, 1], "hspace": 0.05}
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw=gskw)
        ax, ax3 = axes

        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        x = output_data.x
        ax.plot(x, output_data.y, "ko", mfc="white", ms=4, label="spectrum", zorder=1)

        ax.plot(x, output_data.b, color="gray", zorder=1)
        ax.plot(
            x, output_data.f, "-", color="C8", lw=3, label="fitted", alpha=0.8, zorder=3
        )

        for k in range(num_pickup):
            y = output_data.peaks[k, :] + output_data.b
            ax.plot(
                x,
                y,
                linestyles[k],
                color=colors[k],
                lw=1.5,
                label=peakslabels[k],
                zorder=2,
            )

        # Add residue with its baseline
        res_baseline = (
            output_data.f.min() - (output_data.f.max() - output_data.f.min()) * 0.1
        )
        res_baseline = max(0, res_baseline)
        ax.axhline(res_baseline, ls="-", lw=0.5, color="gray")
        y = output_data.y - output_data.f + res_baseline
        ax.plot(x, y, "k-", lw=0.8, label="residue", zorder=3)

        ax.legend(loc="upper right", fontsize=14)

        str_area_list = [f"{v * 100:.1f}" for v in r_list]
        title = "Area ratio = " + ", ".join(str_area_list)
        print(title)
        ax.set_title(title, fontsize=16)
        ax.set_ylabel("Intensity")

        cost_python, poisson_cost = calc_poisson_error(output_data.f, output_data.y)
        barwidth = np.abs((x[1] - x[0]) * 1.0)
        ax3.bar(x, poisson_cost, width=barwidth, color="gray", label="cost")
        ax3.set_ylim(0, 50)
        ax3.set_ylabel("Cost", fontsize=16)

        xlim = (x.max(), x.min())
        ax3.set_xlim(xlim)
        ax3.set_xlabel("Binding energy (eV)")

        # fig.tight_layout()
        figfile = f"out/fitting_{output_data.name}.png"
        fig.savefig(figfile, bbox_inches="tight")
        plt.close("all")


if __name__ == "__main__":
    K = 3
    size = 10
    output_data = OutputData(
        name="hogehoge",
        x=np.random.normal(size=size),
        y=np.random.normal(size=size),
        f=np.random.normal(size=size),
        b=np.random.normal(size=size),
        peaks=np.random.normal(size=(K, size)),
        parameters={
            "hoge": [1, 2, 3],
            "fuga": 456,
        },
    )
    print(output_data)
    print(output_data.peaks.shape)

    presenter = Presenter()
    presenter.save_fitting(output_data=output_data)
    presenter.save_fitting_image(output_data=output_data)
    presenter.save_parameters(output_data=output_data)
