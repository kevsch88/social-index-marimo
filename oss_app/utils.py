import marimo as mo
import pandas as pd
import altair as alt
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Union, Tuple
from scipy.stats import norm as normf
from sklearn import mixture
from oss_app.dataset import Dataset
from scipy.stats import ks_2samp
from sklearn import decomposition
from matplotlib.lines import Line2D

# %% Helper functions for oss_app.py notebook


def mo_print(*args, **kwargs):
    """
    Prints messages to a marimo output cell using mo.redirect_stdout().
    Accepts the same arguments as the built-in print() function.
    """
    with mo.redirect_stdout():
        print(*args, **kwargs)


def today_date(time_included: bool = False) -> str:
    """
    Returns today's date in the format YYYY-MM-DD.
    """
    if time_included:
        return datetime.now().strftime("%y%m%d_%Hh%Mm")
    return datetime.now().strftime("%Y%m%d")


def save_parameters(
    params_output: dict, location: str | Path = None, date_string: str = None, overwrite: bool = False
) -> Path:
    if location is None:
        location = Path("oss_app/params")
    elif not isinstance(location, Path):
        location = Path(location)

    assert location.exists(), f"Destination folder not found:  {location}"
    assert location.is_dir(), f"Destination is not a folder: {location}"

    if not date_string:
        date_string = today_date(time_included=False)

    file_out_name = f"params_{date_string}.json"
    file_out_path = location / file_out_name
    if not overwrite:
        i = 0
        while file_out_path.exists():  # if file already exist, append numbers
            file_out_path = location / f"params_{date_string}_{i}.json"
            i += 1

    mo_print(f"Saving parameters to `{file_out_path}`...")
    with open(file_out_path, "w") as file:
        json.dump(params_output, file, indent=4)
    mo_print("...file saved.")
    return file_out_path


# %% Plotting helper functions


def _split_data_by_group(
    df_to_use: pd.DataFrame, compvar: str, groupvar: str, filters: dict
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Splits filtered dataframe into two arrays based on a grouping variable.
    """

    assert isinstance(df_to_use, pd.DataFrame), "df_to_use must be a pandas DataFrame"
    if filters:
        query_str = " & ".join(
            [f'`{var}`=="{val}"' if isinstance(val, str) else f"`{var}`=={val}" for var, val in filters.items()]
        )
        df_to_use = df_to_use.query(query_str)

    comp_grps = df_to_use.groupby(by=groupvar)

    if len(comp_grps) != 2:
        raise ValueError(
            f"Expected 2 groups after filtering, but found {len(comp_grps)}. Please filter data to two groups."
        )

    comp_arr_labels = list(comp_grps.groups.keys())
    comp_arrays = comp_grps[compvar].apply(lambda x: x.to_numpy() if isinstance(x, pd.Series) else np.array(x)).values

    return comp_arrays[0], comp_arrays[1], comp_arr_labels


def _calculate_distribution(data_array: np.ndarray, f_axis: np.ndarray) -> tuple[np.ndarray, dict]:
    # Fit a Gaussian Mixture Model (MLE for single Gaussian)
    f = np.ravel(data_array).astype(np.float64).reshape(-1, 1)
    g = mixture.GaussianMixture(n_components=1, covariance_type="full", random_state=0)
    g.fit(f)
    weights, means, covars = g.weights_, g.means_, g.covariances_

    mean = means[0, 0]
    std_dev = np.sqrt(covars[0, 0, 0])

    pdf = normf.pdf(f_axis, loc=mean, scale=std_dev).ravel()

    # This will make the curve integrate to 100 over its range
    dist_freq = pdf * 100  # scale for percentage density

    gmm_params = {"mean": mean, "std": std_dev}
    return dist_freq, gmm_params


def _calculate_intersection_and_overlap(gmm_params1: dict, gmm_params2: dict, x_range: list | np.ndarray):
    """
    Calculate overlap coefficient (using the GMM-derived parameters)
    Uses the raw norm.pdf for the coefficient, then multiplies by 100 for percentage
    """
    # using estimated loc and scale from GMM parameters
    mean1_gmm_est, std1_gmm_est = gmm_params1["mean"], gmm_params1["std"]
    mean2_gmm_est, std2_gmm_est = gmm_params2["mean"], gmm_params2["std"]

    pdf_group1_normalized_for_overlap = normf.pdf(x_range, loc=mean1_gmm_est, scale=std1_gmm_est)
    pdf_group2_normalized_for_overlap = normf.pdf(x_range, loc=mean2_gmm_est, scale=std2_gmm_est)

    min_of_normalized_pdfs = np.minimum(pdf_group1_normalized_for_overlap, pdf_group2_normalized_for_overlap)
    dx_plot = x_range[1] - x_range[0]
    # This is the [0,1] coefficient
    overlap_coefficient = np.sum(min_of_normalized_pdfs) * dx_plot
    overlap_percentage_value = overlap_coefficient * 100

    print(f"\nCalculated Overlap Coefficient: {overlap_coefficient:.4f}")
    print(f"Calculated Overlap Percentage: {overlap_percentage_value:.2f}%")
    return overlap_coefficient, overlap_percentage_value


def _perform_ks_test(data_array1: np.ndarray, data_array2: np.ndarray) -> tuple[float, float]:
    """
    Performs and reports the Kolmogorov-Smirnov test between two groups.
    """
    ks_stat, ks_pval = ks_2samp(data_array1.reshape(-1), data_array2.reshape(-1))
    print(f"\t--Kolmogorov-Smirnov test between groups: {ks_stat:.2f},  p-val: {ks_pval:.5f}")
    return ks_stat, ks_pval


def compare_dists_altair(
    df: pd.DataFrame,
    compare_metric="",
    group_variable="",
    filters={},
    max_y=40,
    rangex: list | None = [-2.5, 2.5],
    bin_width=1,
    dot=False,
    user_colors=None,
    user_labels: list[str, str] = None,
    set_size_params=None,
    legend=False,
    hide_text=False,
    alpha1=0.7,
    alpha2=0.4,
):
    """
    Compares distribution between two groups for a given variable using Altair.
    This function is a wrapper that utilizes helper methods to perform its core tasks:
    1. _split_data_by_group: Splits data into two arrays for comparison.
    2. _calculate_distribution: Calculates the GMM distribution for each group.
    3. _calculate_intersection_and_overlap: Finds the intersection and overlap coefficient.
    4. _perform_ks_test: Runs a Kolmogorov-Smirnov test.
    The final output is an Altair chart object.

    :param compvar: Comparison variable name.
    :type compvar: str
    :param groupvar: Grouping variable name.
    :type groupvar: str
    :param filters: Dictionary of filters to apply to the dataframe.
    :type filters: dict
    :param max_y: Maximum y-axis value.
    :type max_y: int
    :param rangex: Range for the x-axis, e.g., [-2.5, 2.5].
    :type rangex: list
    :param binw: Bin width for density calculation scaling.
    :type binw: int
    :param dot: Whether to plot the intersection point.
    :type dot: bool
    :param user_colors: Tuple of two colors for the groups.
    :type user_colors: tuple[str, str], optional
    :param user_labels: List of two labels for the groups.
    :type user_labels: list[str, str], optional
    :param set_size_params: Tuple (width, height) in inches for the plot size.
    :type set_size_params: tuple[float, float], optional
    :param legend: Whether to display the legend.
    :type legend: bool
    :param hide_text: Whether to hide all text (titles, labels, ticks).
    :type hide_text: bool
    :param alpha1: Opacity for the first group's area.
    :type alpha1: float
    :param alpha2: Opacity for the second group's area.
    :type alpha2: float
    :return: An Altair chart object.
    :rtype: alt.Chart
    """
    assert compare_metric != "", 'no comparison variable for "compare_metric" chosen'
    assert group_variable != "", 'no grouping variable for "group_variable" chosen'

    # 1. Split data into two groups
    array1, array2, comp_arr_labels = _split_data_by_group(df, compare_metric, group_variable, filters)

    plot_title = f"{compare_metric=},  {group_variable=}"

    # 2. Calculate distributions for each group
    concat_min = np.concatenate((array1, array2)).min()
    concat_max = np.concatenate((array1, array2)).max()
    if not rangex:  # If rangex is not provided, set it based on data but wider
        rangex = [round(concat_min - 5), round(concat_max + 5)]
    range_min, range_max = rangex
    smooth_factor = 500
    x_plot_range = np.linspace(
        # Increase smooth_factor for smoother curves
        range_min,
        range_max,
        smooth_factor,
    )

    # Calculate distributions and get GMM parameters
    dist_freq_group1, gmm_params_group1 = _calculate_distribution(
        data_array=array1,
        f_axis=x_plot_range,
    )
    mean1_gmm_est = gmm_params_group1["mean"]
    std1_gmm_est = gmm_params_group1["std"]

    dist_freq_group2, gmm_params_group2 = _calculate_distribution(
        data_array=array2,
        f_axis=x_plot_range,
    )
    mean2_gmm_est = gmm_params_group2["mean"]
    std2_gmm_est = gmm_params_group2["std"]

    print(f"GMM Group 1 Estimates: Mean={mean1_gmm_est:.2f}, Std={std1_gmm_est:.2f}")
    print(f"GMM Group 2 Estimates: Mean={mean2_gmm_est:.2f}, Std={std2_gmm_est:.2f}")

    # 3. Calculate intersection and overlap coefficient
    overlap_coeff, overlap_pct = _calculate_intersection_and_overlap(gmm_params_group1, gmm_params_group2, x_plot_range)

    # 4. Perform and report Kolmogorov-Smirnov test
    ks_stat, ks_pval = _perform_ks_test(array1, array2)

    # --- Altair Charting ---
    # Prepare data for Altair DataFrame
    _user_colors = user_colors if user_colors else ("#e45756", "#4c78a8")
    _lgnd_labels = user_labels if user_labels is not None else [str(lbl) for lbl in comp_arr_labels]

    data_dist_list = []
    for val_x, val_y in zip(x_plot_range, dist_freq_group1):
        data_dist_list.append({"f_axis": val_x, "dist_val": val_y, "group": _lgnd_labels[0], "alpha_val": alpha1})
    for val_x, val_y in zip(x_plot_range, dist_freq_group2):
        data_dist_list.append({"f_axis": val_x, "dist_val": val_y, "group": _lgnd_labels[1], "alpha_val": alpha2})
    source_dist = pd.DataFrame(data_dist_list)

    # Base chart properties
    plot_width = set_size_params[0] * 96 if set_size_params else 192
    plot_height = set_size_params[1] * 96 if set_size_params else 192
    if not max_y:
        concat_max = max(dist_freq_group1.max(), dist_freq_group2.max())
        max_y = round(concat_max + 10)
    # Create Altair chart
    area_chart = (
        alt.Chart(source_dist)
        .mark_area(
            line={"color": "black", "strokeWidth": 1.5, "strokeOpacity": 1},
            strokeWidth=0,
        )
        .encode(
            x=alt.X("f_axis:Q", title=f"Var: {compare_metric}", scale=alt.Scale(domain=rangex, nice=False, zero=False)),
            y=alt.Y(
                "dist_val:Q",
                title="% Subjects",
                stack=False,
                scale=alt.Scale(domain=[0, max_y], nice=False, zero=True),
                axis=alt.Axis(format=".0f"),
            ),
            fill=alt.Fill(
                "group:N",
                scale=alt.Scale(domain=_lgnd_labels, range=_user_colors),
                legend=alt.Legend(title=group_variable) if legend else None,
            ),
            fillOpacity=alt.FillOpacity("alpha_val:Q", legend=None),
        )
    )

    chart = area_chart

    # Titles and styling
    chart = chart.properties(
        width=plot_width,
        height=plot_height,
        title=alt.TitleParams(
            text=f"{plot_title}",
            subtitle=f"Overlap coeff: {overlap_coeff:.3f},  pct: {overlap_pct:.2f}%  |  K-S test: {ks_stat:.2f}, p-val: {ks_pval:.3f}",
            fontSize=12,
            subtitleFontSize=10,
            subtitleFontStyle="italic",
            anchor="middle",
            offset=10,
        ),
    )

    if hide_text:
        chart = chart.configure_axis(labels=False, title=None, ticks=False, grid=False).properties(
            title=alt.TitleParams(text="", subtitle="")
        )
    else:
        chart = chart

    return chart


# %% Data processing helper functions


def fix_name(*names: str) -> Union[str, Tuple[str, ...]]:
    """
    Fixes one or more names by removing leading/trailing whitespace and converting to lowercase.
    If one name is passed, returns a string.
    If multiple names are passed, returns a tuple of strings.
    """
    fixed_names = [name.strip().lower() for name in names]
    if len(fixed_names) == 1:
        return fixed_names[0]
    return tuple(fixed_names)


def fix_column_names(df: pd.DataFrame, *additional_df: Tuple[pd.DataFrame, ...]):
    """
    Fix column names by removing leading/trailing whitespace and converting to lowercase.
    """
    df.columns = [fix_name(col) for col in df.columns]
    for additional in additional_df:
        additional.columns = [fix_name(col) for col in additional.columns]


def make_categorical(series: pd.Series) -> dict:
    """
    Creates a dictionary with codes and labels for a categorical series.
    Leveraged in plotting functions

    Args:
        series: The pandas Series to convert.

    Returns:
        A dictionary containing the categorical codes and labels (categories).
    """
    categorical_series = pd.Categorical(series)
    return {"codes": categorical_series.codes, "labels": categorical_series.categories}


def do_pca(data: pd.DataFrame, n_comp=3):
    """Performs Principal Component Analysis (PCA) on the provided dataset.

    This function wraps `sklearn.decomposition.PCA` to simplify the process of
    dimensionality reduction. It fits the PCA model to the data, transforms the
    data into the principal component space, and returns the fitted model along
    with the results.

    Args:
        data (pd.DataFrame): A DataFrame containing the scaled numerical data
            (features) to be analyzed. The shape should be (n_samples, n_features).
        n_comp (int, float, optional): The number of principal components to keep.
            - If an integer, it's the absolute number of components.
            - If a float between 0.0 and 1.0 (e.g., 0.95), it's the amount of
              variance that should be explained by the selected components.
            Defaults to 3.

    Returns:
        tuple: A tuple containing the following three elements:
            - pca (sklearn.decomposition.PCA): The fitted PCA object from scikit-learn.
              This object can be used for further analysis or inverse transformations.
            - principal_components (np.ndarray): An array of shape
              (n_samples, n_components) representing the data transformed into the
              principal component space.
            - pc_evr (np.ndarray): An array containing the percentage of variance
              explained by each of the selected components.
    """
    pca = decomposition.PCA(n_components=n_comp)
    principal_components = pca.fit_transform(data)
    pc_evr = pca.explained_variance_ratio_
    return pca, principal_components, pc_evr


def pca_biplot(
    dataset_obj: Dataset,
    df_to_use=pd.DataFrame() | str | None,
    grouping_variable="",
    metrics_inluded: list[str] | None = None,
    labels="",
    n_comp=3,
    mapping=0,
    pcs=None,
    colmap=None,
    hide_text=False,
    **scatter_kwargs,
):
    assert isinstance(dataset_obj, Dataset), "dataset_obj must be an instance of Dataset class."
    df: pd.DataFrame = pd.DataFrame()

    # Data prep
    assert not df_to_use.empty, "Dataset object's input DataFrame `df_to_use` cannot be empty."
    if isinstance(df_to_use, str):
        assert getattr(dataset_obj, df_to_use, None) is not None, f"Dataset object has no dataset `{df_to_use}`."
        df = getattr(dataset_obj, df_to_use)
    elif df_to_use is None:
        assert getattr(dataset_obj, "scaled_df", None) is not None, "Default dataset `scaled_df` not found."
        df = getattr(dataset_obj, "scaled_df")
    else:
        df = df_to_use
    if not metrics_inluded:
        # metrics_inluded = [col for col in dataset_obj.metric_variables if col != "si_score"]
        metrics_inluded = dataset_obj.metric_variables
    metric_labels = metrics_inluded

    assert not (df.empty or df is None), "Dataset object's input DataFrame `df` cannot be None."
    df = df[metrics_inluded].copy()  # type: ignore # copy to avoid modifying original DataFrame

    if not grouping_variable:
        grouping_variable = dataset_obj.grouping_variable
    group_categories = make_categorical(df[grouping_variable])

    # PCA
    pca, princomps, pc_evr = do_pca(df, n_comp)
    if not getattr(dataset_obj, "pca", None):
        dataset_obj.pca = do_pca(df, n_comp)
    if pcs is None:
        pc_range = range(n_comp)
        pcset_to_plot = [[x, y] for x, y in zip(pc_range[0:-1], pc_range[1:])]
    else:
        pcset_to_plot = [pcs]

    # Plotting setup
    x_index = group_categories["codes"]
    stylemap = {0: "v", 1: "o"}  # marker style for groups
    sizemap = {0: 70, 1: 60}  # marker size for groups
    conds_stylemap = [*map(stylemap.get, x_index)]
    conds_sizemap = [*map(sizemap.get, x_index)]
    if colmap is None:
        match not dataset_obj.plot_colors:
            case False:
                colmap = dataset_obj.plot_colors[0].new_cmap
            case _:
                dataset_obj.map_colors(dataset_obj.si_scores)
                colmap = dataset_obj.plot_colors[0].new_cmap
    mapping = dataset_obj.plot_colors[0].face_colors

    kwargs = dict(  # scatter parameters
        edgecolors="face",
        lw=0.5,
        zorder=2,
    )
    kwargs.update(scatter_kwargs)  # update kwargs with user parameters

    # set up legend entries
    leg_markers = [*map(stylemap.get, np.unique(group_categories["codes"]))]
    legend_handles = [
        Line2D([], [], marker=m, markeredgecolor="k", markeredgewidth=1.5, markerfacecolor="w", linewidth=0)
        for m in leg_markers
    ]

    # Plotting, per set of PCs given, only one if `pcs` not None
    for pcset in pcset_to_plot:
        coeff = np.transpose(pca.components_[pcset[0] : pcset[1] + 1])
        xs = princomps[:, pcset[0]]
        ys = princomps[:, pcset[1]]
        n = coeff.shape[0]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        # pdb.set_trace()
        ncmap = colmap

        plt.figure(figsize=(3, 3))
        # plt.scatter(xs * scalex, ys * scaley, c=mapping, cmap=ncmap, s=25, edgecolors='face',
        #             marker=conds_stylemap, size=conds_sizemap, zorder=2)
        for _st, _si, _c, _x, _y in zip(conds_stylemap, conds_sizemap, mapping, xs, ys):
            fs = plt.scatter(_x * scalex, _y * scaley, color=_c, cmap=ncmap, s=_si, marker=_st, **kwargs)
        for i in range(n):
            if not hide_text:
                match labels:
                    case "labeled":
                        # plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, f'{var_labels[i]}',
                        plt.text(
                            coeff[i, 0] + 0.1,
                            coeff[i, 1],
                            f"{metric_labels[i]}",
                            color="darkgreen",
                            ha="left",
                            va="center",
                        )
                    case "ordered":
                        plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.0, str(i), color="k", ha="center", va="center")
                    case "":
                        pass
            # arc_direction_list = iter(["-", "", "-", "-", "-"])
            plt.annotate(
                "",
                xy=(coeff[i, 0], coeff[i, 1]),
                xytext=(0, 0),  # color='crimson'
                arrowprops=dict(
                    arrowstyle="-|>",
                    shrinkA=0,
                    fill=True,
                    color="xkcd:coral",
                    mutation_scale=12,
                    lw=1.5,
                    #  connectionstyle= f'arc3,rad={next(arc_direction_list)}0.1',
                ),
                zorder=3,
            )

        plt.grid(zorder=1)
        plt.xlim(-1.2, 1.2)
        plt.xticks([-1, 0, 1])
        plt.ylim(-1.2, 1.2)
        plt.yticks([-1, 0, 1])
        # plt.xlim(lims:=(-.9, .9))
        # plt.xlim(lims:=(-.92, .92))
        # plt.xticks(ticklims:=[-0.5, 0, 0.5])
        # plt.ylim(lims)
        # plt.yticks(ticklims)
        plt.xlabel(f"PC{pcset[0] + 1}")
        plt.ylabel(f"PC{pcset[1] + 1}")
        plt.title(
            f"Explained variance ratio: \nPC{pcset[0] + 1}: {pc_evr[pcset[0]]:.2f},  PC{pcset[1] + 1}: {pc_evr[pcset[1]]:.2f} \nCombined:{sum(pc_evr[pcset]):.2f}"
        )
        if hide_text:
            # Hide X and Y axes label marks
            ax = plt.gca()
            plt.xlim(lims := (-0.92, 0.92))
            plt.ylim(lims)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            plt.xlabel(None)
            plt.ylabel(None)
            plt.title(None)
            plt.grid(visible=False)
            # single grid line instead
            ax.axhline(0, linestyle=":", color="xkcd:gray", zorder=1)  # horizontal lines
            ax.axvline(0, linestyle=":", color="xkcd:gray", zorder=1)  # vertical lines
        else:
            labels = list(group_categories["labels"])
            handles = legend_handles
            ax = plt.gca()
            ax.legend(
                handles,
                labels,
                loc="upper right",
                fontsize=8,
                bbox_to_anchor=(1.5, 1),
                frameon=False,
                markerscale=1.5,
            )


# TODO: Logging helper functions
