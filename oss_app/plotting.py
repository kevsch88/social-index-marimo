import matplotlib.pyplot as plt
import altair as alt
from typing import Union, Tuple
from scipy.stats import norm as normf
from sklearn import mixture
from oss_app.dataset import Dataset
from scipy.stats import ks_2samp
from sklearn import decomposition
from pandas.api.types import is_numeric_dtype
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
import pandas as pd
import marimo as mo
import numpy as np


def map_colors(self, data_array: np.ndarray = [], color_set=["viridis_r", "plasma_r"], cmap_rng=None):
    if not data_array:
        data_array = self.si_scores

    self.cset = color_set
    num_sex = len(np.unique(self.df["Sex"].values))

    if not cmap_rng:
        cmap_rng = (0.4, 1)

    match num_sex:
        case 2:  # 2 sexes present
            # default colors for male and female animals
            plot_colors = []
            self.c_debug(f"using {self.cset} as color sets for 2 groups (default:sexes)")
            for i, cset in enumerate(self.cset):
                plot_colors.append(CmapSet(cset, data_array, cmap_range=cmap_rng))
        # only males present
        case 1 if "Male".casefold() in self.df["Sex"].values:
            plot_colors = CmapSet(self.cset[0], data_array, cmap_range=cmap_rng)
            self.c_debug(f"using {self.cset[0]} as color set for males")
        # only females present
        case 1 if "Female".casefold() in self.df["Sex"].values:
            plot_colors = CmapSet(self.cset[1], data_array, cmap_range=cmap_rng)
            self.c_debug(f"using {self.cset[1]} as color set for females")
        case _:  # default to plasma cmap in any other case
            plot_colors = CmapSet(self.cset[0], data_array, cmap_range=cmap_rng)
            self.c_debug(f"using {self.cset[1]} as default color set")

    self.plot_colors = plot_colors


def blend_cmap(self, name, split=True, cmap_rng: tuple[float, float] = None, col_choice=None):
    # for corr heatmaps (pca), makes it one directional - might change this
    if col_choice is None:
        col_choice = self.color_choice
    match col_choice:
        case "viridis_r":
            tophalf, bothalf = "BrBG_r", "BrBG"
        case "plasma_r":
            tophalf, bothalf = "RdPu_r", "RdPu"
        case _:
            if "_r" in col_choice:
                tophalf, bothalf = f"{col_choice[:-2]}", f"{col_choice}"
            else:
                tophalf, bothalf = f"{col_choice}_r", f"{col_choice}"

    top = cm.get_cmap(tophalf, 1024)
    bottom = cm.get_cmap(bothalf, 1024)

    if not cmap_rng:
        cmap_rng = (0, 1)
    if split:
        newcolors = np.vstack((top(np.linspace(0, 0.5, 1024)), bottom(np.linspace(0.5, 1, 1024))))
    else:
        newcolors = np.vstack(
            (
                top(np.linspace(1 - cmap_rng[1], 1 - cmap_rng[0], 1024)),
                bottom(np.linspace(cmap_rng[0], cmap_rng[1], 1024)),
            )
        )
    if not name:
        name = f"blended_{self.color_choice}"
    blended_cmap = ListedColormap(newcolors, name=name)
    self.blended_cmap = blended_cmap


def show_colormap(cmap: Union[str, ListedColormap], label: str = "", figsize: Tuple[float, float] = (1.25, 0.25)):
    """Displays a visual representation of a Matplotlib colormap within a Marimo cell.
    This function generates a horizontal color bar for the given colormap and
    embeds it as an HTML object in a Marimo markdown cell, making it easy to
    visualize colormaps directly in a notebook.
        cmap: The Matplotlib colormap to display. This can be the string name
            of a registered colormap (e.g., "viridis") or a `matplotlib.colors.Colormap`
            object.
        label: An optional text label to display next to the colormap visualization.
        figsize: A tuple specifying the (width, height) in inches for the
            generated color bar figure.
    Returns:
        A `marimo.md` object containing the HTML representation of the colormap
        visualization, ready to be displayed in a Marimo cell.
    """
    fig, ax = plt.subplots(figsize=figsize, frameon=False)
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect="auto", cmap=cmap)
    ax.set_axis_off()

    fig_md = mo.as_html(fig)
    md_content = f"{label}{fig_md}" if label else f"{fig_md}"
    plt.close(fig)  # Prevent double-plotting in some environments
    return mo.md(md_content)


def show_color(hex_code: str, label: str = ""):
    """Generates a Marimo markdown object to display a color swatch.

    This function creates a small, colored square using HTML and CSS,
    wrapped in a Marimo markdown object, making it easy to visualize
    colors directly within a Marimo notebook.

    Args:
        hex_code (str): The hexadecimal color code to display (e.g., "#FF0000").
        label (str, optional): A text label to show next to the color swatch.
            Defaults to an empty string.

    Returns:
        marimo.md: A Marimo markdown object that renders as a label
            followed by a colored square.
    """
    "example hex_code=#FF0000"
    md_content = mo.md(
        f"""{label}: &nbsp;<span style="background-color: {hex_code}; display: inline-block; width: 20px; height: 20px; border: 1px solid black;vertical-align:middle"></span>"""
    )
    return md_content
