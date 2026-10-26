"""Check plot/report wording where it affects the statistical interpretation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.export import interpretations
from analysis.regression import fit_model
from analysis.visualization import distribution_plot, forest_plot


def test_one_predictor_model_is_not_described_as_adjusted():
    x = np.arange(15, dtype=float)
    frame = pd.DataFrame({"x":x, "y":2*x + np.array([1,-1,0,2,1,-2,1,0,-1,2,-2,0,1,-1,1])})
    model = fit_model(frame, "y", ["x"], {"x":"continuous", "y":"continuous"})
    assert model["success"]
    text = interpretations(model)[0]
    assert "unadjusted" in text
    assert "conditional on the other" not in text
    fig = forest_plot(model)
    assert "Unadjusted" in fig.axes[0].get_title(loc="left")
    plt.close(fig)


def test_continuous_plot_count_uses_only_finite_plotted_values():
    frame = pd.DataFrame({"x":[1,2,3,float("inf"),np.nan,"bad"]})
    fig = distribution_plot(frame, "x", {"x":"continuous"})
    assert "3 observed measurements" in fig.axes[0].get_title(loc="left")
    plt.close(fig)
