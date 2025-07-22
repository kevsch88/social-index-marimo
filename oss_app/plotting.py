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

from oss_app.utils import mo_print


# def map_colors(self, data_array: np.ndarray = [], color_set=["viridis_r", "plasma_r"], cmap_rng=None):
#     if not data_array:
#         data_array = self.si_scores

#     self.cset = color_set
#     num_sex = len(np.unique(self.df["Sex"].values))

#     if not cmap_rng:
#         cmap_rng = (0.4, 1)

#     match num_sex:
#         case 2:  # 2 sexes present
#             # default colors for male and female animals
#             plot_colors = []
#             self.c_debug(
#                 f"using {self.cset} as color sets for 2 groups (default:sexes)")
#             for i, cset in enumerate(self.cset):
#                 plot_colors.append(
#                     CmapSet(cset, data_array, cmap_range=cmap_rng))
#         # only males present
#         case 1 if "Male".casefold() in self.df["Sex"].values:
#             plot_colors = CmapSet(
#                 self.cset[0], data_array, cmap_range=cmap_rng)
#             self.c_debug(f"using {self.cset[0]} as color set for males")
#         # only females present
#         case 1 if "Female".casefold() in self.df["Sex"].values:
#             plot_colors = CmapSet(
#                 self.cset[1], data_array, cmap_range=cmap_rng)
#             self.c_debug(f"using {self.cset[1]} as color set for females")
#         case _:  # default to plasma cmap in any other case
#             plot_colors = CmapSet(
#                 self.cset[0], data_array, cmap_range=cmap_rng)
#             self.c_debug(f"using {self.cset[1]} as default color set")

class ColorSet:
    """A class to handle color mapping for datasets."""

    def __init__(self,
                 color_name: str, metric_name: str,
                 grouping_variable: str, group_name: str, data: pd.DataFrame | pd.Series | np.ndarray | None = None,
                 cmap_range: tuple[float, float] = (0.4, 1), n_divisions: int = 151
                 ):
        self.name = color_name
        self.metric_name = metric_name
        self.grouping_variable = grouping_variable
        if not isinstance(group_name, str):
            raise ValueError("group_name must be a string.")
        self.group_name = group_name
        self.data = self._filter_data(data.copy())
        self.cmap_range = cmap_range if cmap_range else (0, 1)
        # get colormap based on name and range given
        self.cmap = cm.get_cmap(color_name)
        self.divcmap = ListedColormap(self.cmap(np.linspace(
            self.cmap_range[0], self.cmap_range[1], n_divisions)), name=f"{color_name}_divided")
        self.colors = self._generate_colors()

    def _filter_data(self, data: pd.DataFrame | pd.Series | np.ndarray | None = None):
        """Filters the data to the relevant group."""
        if not isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
            return data
        data = data[data[self.grouping_variable] == self.group_name][self.metric_name] if isinstance(
            data, pd.DataFrame) else data[data[self.grouping_variable] == self.group_name]
        # mo_print(
        #     f'{self.grouping_variable} == {self.group_name} -> {len(data)} rows')
        return data

    def _generate_colors(self):
        """Generates colors based on the colormap and data."""
        if isinstance(self.data, pd.DataFrame):
            values = self.data.select_dtypes(
                include=[np.number]).values.flatten()
        elif isinstance(self.data, pd.Series):
            values = self.data.values.flatten()
        elif isinstance(self.data, np.ndarray):
            values = self.data.flatten()
        else:
            raise ValueError("Data must be a DataFrame or NumPy array.")

        normed_values = (values - np.min(values)) / \
            (np.max(values) - np.min(values))
        return [colors.rgb2hex(self.cmap(v)) for v in normed_values]

    def _map_colors(self, data: pd.DataFrame | np.ndarray | None = None):
        """Maps colors to groups in a DataFrame or NumPy array."""
        if data is None:
            data = self.data
        if isinstance(data, pd.DataFrame):
            return data.apply(lambda x: self.colors[x.name], axis=1)
        elif isinstance(data, np.ndarray):
            return np.array([self.colors[i] for i in range(len(data))])
        else:
            raise ValueError("Data must be a DataFrame or NumPy array.")

    def _blend_colors(self, other_color_set: 'ColorSet', split: bool = True):
        """Blends this color set with another ColorSet."""
        if not isinstance(other_color_set, ColorSet):
            raise ValueError(
                "other_color_set must be an instance of ColorSet.")

        top_half = cm.get_cmap(self.name, 1024)
        bottom_half = cm.get_cmap(other_color_set.name, 1024)

        if split:
            new_colors = np.vstack(
                (top_half(np.linspace(0, 0.5, 1024)), bottom_half(np.linspace(0.5, 1, 1024))))
        else:
            new_colors = np.vstack(
                (
                    top_half(np.linspace(
                        1 - self.cmap_range[1], 1 - self.cmap_range[0], 1024)),
                    bottom_half(np.linspace(
                        self.cmap_range[0], self.cmap_range[1], 1024)),
                )
            )
        return ListedColormap(new_colors, name=f"blended_{self.name}_{other_color_set.name}")

    def __repr__(self):
        return f"ColorSet(name={self.name}, group_name={self.group_name}, cmap_range={self.cmap_range})"


def show_colormap(cmap: Union[str, ListedColormap], label: str = "", figsize: Tuple[float, float] = (1.25, 0.25), font_family: str = "arial"):
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

    fig_html = mo.as_html(fig)
    # md_content = f"{label}{fig_md}" if label else f"{fig_md}"
    plt.close(fig)  # Prevent double-plotting in some environments
    if label:
        # Create a marimo markdown object for the label
        label_md = mo.md(
            f"<span style='font-family:{font_family}'>{label}</span>")
        # Use hstack to place them side by side
        return mo.hstack([label_md, fig_html])
    else:
        return fig_html


def show_color(color: str, label: str = "", font_family: str = "arial") -> mo.md:
    """Generates a Marimo markdown object to display a color swatch.

    This function creates a small, colored square using HTML and CSS,
    wrapped in a Marimo markdown object, making it easy to visualize
    colors directly within a Marimo notebook.

    Args:
        color (str): The name or hexadecimal color code to display (e.g., "#FF0000").
        label (str, optional): A text label to show next to the color swatch.
            Defaults to an empty string.

    Returns:
        marimo.md: A Marimo markdown object that renders as a label
            followed by a colored square.
    """
    "example hex_code=#FF0000"
    md_content = mo.md(
        f"""<span style='font-family:{font_family}'>{label}</span>&nbsp;<span style="background-color: {color}; display: inline-block; width: 20px; height: 20px; border: 1px solid black;vertical-align:middle"></span>"""
    )
    return md_content.style({'font-family': 'sans-serif'})


def colormap_to_hex(cmap: Union[str, ListedColormap], num_colors: int = 10) -> list[str]:
    """Converts a Matplotlib colormap to a list of hexadecimal color codes.

    Args:
        cmap (Union[str, ListedColormap]): The colormap to convert. This can be
            the name of a registered colormap (e.g., "viridis") or a
            `matplotlib.colors.Colormap` object.
        num_colors (int): The number of colors to extract from the colormap.
            Defaults to 10.

    Returns:
        list[str]: A list of hexadecimal color codes representing the
            specified number of colors from the colormap.
    """
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    return [colors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, num_colors)]
