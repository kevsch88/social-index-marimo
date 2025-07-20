import marimo as mo
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

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


# %% Data processing helper functions


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


# TODO: Logging helper functions
