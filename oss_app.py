# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():

    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from oss_app import oss_func_simple as sf
    # import pandera.pandas as pa
    from pathlib import Path
    import altair as alt
    import asyncio
    import json
    from os import mkdir
    from datetime import datetime
    return Path, alt, asyncio, datetime, json, mo, np, pd, sf


@app.cell
def _(mo):
    mo.md(r"""# Setup""")
    return


@app.cell
def _(mo):
    mo.md(r"""## file selection""")
    return


@app.cell
def _(Path, mo):
    # set starting path and dynamic state to retrieve, for returning to default
    # change if need to look for file elsewhere
    starting_path = Path(r'./raw_csv').resolve()
    reset_button = mo.ui.run_button(label='Return to initial path')

    checkbox_style = "<span style='display:inline;vertical-align:baseline;color:coral; font-size:14px'>"
    #     'justify-self':'left','padding':'1px 1px 1px 5px'>

    overwrite_cbox = mo.ui.checkbox(
        label=checkbox_style+"Overwrite outputs</span>", value=False)
    return checkbox_style, overwrite_cbox, reset_button, starting_path


@app.cell
def _(mo, reset_button, starting_path):
    # file selection
    browser = mo.ui.file_browser(
        initial_path=starting_path,
        filetypes=['.csv'],
        selection_mode='file',
        multiple=False,
        restrict_navigation=False,  # prevent from moving out of input csv folder
        label='''###Select CSV file to import.<br /></h3><font color="khaki">If file not visible below, make sure it is inside the "raw_csv" folder.''',
    )

    # file selector
    file = mo.md('''
        {browser}

        <span style='display:inline;padding:0px 0px 0px 5px;float:left'>{reset_button}</span> 
        ''').batch(browser=browser,
                   reset_button=reset_button
                   ).form(
        bordered=True, loading=False,
        submit_button_label="Confirm Selection"
    )
    # <span style='display:inline; float:right; box-sizing:border-box; border:1px solid coral; padding-left:5px'>{overwrite}<br>{append}</span>
    # selections for overwriting or appending outputs
    return (file,)


@app.cell
def _(checkbox_style, mo, overwrite_cbox):
    append_cbox = mo.ui.checkbox(label=checkbox_style+"Append outputs</span>",
                                 value=False, disabled=True if overwrite_cbox.value else False)
    return (append_cbox,)


@app.cell
async def _(asyncio, file, file_ui, mo, reset_button):

    # reset gui
    mo.output.append(file)
    if reset_button.value:
        mo.output.clear()
        with mo.status.spinner(title="Reloading GUI...") as _spinner:
            # task=asyncio.create_task(mo.output.clear())

            # mo.output.append(_spinner)
            _spinner.update(subtitle="Clearing...")
            await asyncio.sleep(1)

            _spinner.update(subtitle="Reloading...")
            await asyncio.sleep(1.5)
            # await asyncio.run(mo.output.clear())
            # await task
            _spinner.update(title="Done...")
            mo.output.replace(file_ui)

    return


@app.cell
def _(append_cbox, mo, overwrite_cbox):
    overwrite_choice = mo.md(f"""
            <span style='display:inline; float:left; width:150px; box-sizing:border-box; border:1px solid coral; padding-left:5px'>{overwrite_cbox}</span>
            """)

    append_choice = mo.md(f"""
            <span style='display:inline; float:left; width:150px; box-sizing:border-box; border:1px solid coral; padding-left:5px'>{append_cbox}</span>
            """)
    return append_choice, overwrite_choice


@app.cell
def _(append_choice, mo, overwrite_choice):

    mo.output.append(
        mo.vstack([
            mo.md("<br><h3> Overwrite or append selections</h3>"),
            mo.hstack([overwrite_choice, mo.md("Replace output folders and files if they exist.")],
                      justify='start', align='center', widths=[0.15, 0.85]),
            mo.hstack([append_choice, mo.md("Disabled if overwriting. Appends numbers to output folders and files if they exist.")],
                      justify='start', align='center', widths=[0.15, 0.85]),
        ])
    )
    return


@app.cell
def _():
    # form = (
    #     mo.md(
    #         '''
    #     **Enter your prompt.**

    #     {prompt}

    #     **Choose a random seed.**

    #     {seed}
    #     '''
    #     )
    #     .batch(
    #         prompt=mo.ui.text_area(),
    #         seed=mo.ui.number(),
    #     )
    #     .form()
    # )
    # form
    return


@app.cell
def _(Path, file, mo, overwrite_cbox):
    # Prevent progression if no file selected, mo.stop prevents code below from running.
    file_not_chosen = file.value is None
    mo.stop(file_not_chosen, mo.md(
        "###**Confirm file selection to continue...**"))

    file_choice = file.value["browser"][0]
    if file.value["browser"] == None:
        filepath = r"raw_csv/example_SI_data.csv"
        filename = Path(filepath).name
    else:
        filepath = file_choice.path
        filename = file_choice.name
        filepath_parent = Path(filepath).parent
    overwrite = overwrite_cbox.value
    if overwrite:
        _notice = mo.md(f"""
        /// warning | Overwrite set to True, outputs will be replaced.
        """)
    else:
        _notice = mo.md(f"""
        /// tip | Overwrite set to False, new outputs will be created, with appended numbers if already present.
        """)
    mo.output.append(_notice)
    return filename, filepath, filepath_parent, overwrite


@app.cell
def _(Path, datetime, filename, filepath_parent, mo, overwrite):
    create_btn = mo.ui.run_button(label='Create new output folders')

    # output folder will be in input file folder
    # today_dt = datetime.today().strftime('%y%m%d_%Hh%Mm')  # date and time stamp
    today_dt = datetime.today().strftime('%y%m%d')  # date stamp only
    save_path = filepath_parent / f'{Path(filename).stem}_{today_dt}'

    # check for previously saved params file
    params_found = (params_file := list(
        filepath_parent.rglob('params*.json'))) != []
    load_btn = mo.ui.run_button(
        label='Load params file', disabled=not params_found)  # load params button
    loaded_params, set_loaded_params = mo.state(False)

    with mo.redirect_stdout():
        if not overwrite:
            if params_found:
                params_file = params_file[0]
                print(f'Found params file at: {params_file}.')
                print(f'Load previous parameters or create new output folder?')
            else:
                print(f'No previous params file found.')
                print(f'Click button to create new output folder.')

    # if not overwriting and previous params found; load or create new?
    mo.output.append(mo.vstack([
        create_btn,
        load_btn
    ]))
    return (
        create_btn,
        loaded_params,
        params_file,
        save_path,
        set_loaded_params,
        today_dt,
    )


@app.cell
def _(mo):
    def display_selections_markdown(params_output: dict, title: str = "Selections"):
        if not params_output:
            return mo.md("---\n\n## No selections have been confirmed yet.")

        md_content = f"---\n\n## <span style='color:lightcoral'>{title}\n\n"

        # Helper to format list or single item
        def format_value(key, value):
            if value is None or (isinstance(value, list) and not value):
                return f"* {key.replace('_', ' ').title()}: _Not selected_"
            elif isinstance(value, list):
                return f"* {key.replace('_', ' ').title()}: &nbsp; <span style='color:coral; font-family:monospace'>{', '.join(value)}</span>"
            else:
                return f"* {key.replace('_', ' ').title()}: &nbsp;  `{value}`"

        filename = params_output["filename"]
        filters = params_output["filters"]
        fformat = "<span style='color:coral; font-family: monospace'>"
        md_content += "#### Input file:\n&nbsp; " + f'"`{filename}`"'
        md_content += "\n#### Variables:"
        md_content += "\n" + \
            format_value("subject_id_variable",
                         params_output.get("subject_id_variable"))
        md_content += "\n" + \
            format_value("sex_variable",
                         params_output.get("sex_variable"))
        md_content += "\n" + \
            format_value("grouping_variable",
                         params_output.get("grouping_variable"))
        md_content += "\n" + \
            format_value("index_variables",
                         params_output.get("index_variables"))
        md_content += "\n" + \
            format_value("metric_variables",
                         params_output.get("metric_variables"))
        md_content += "\n#### Filters:"
        for fkey, fval in filters.items():
            md_content += "\n* " + \
                f'{fformat}{fkey}</span>" &nbsp;== &nbsp; "{fformat}{fval}</span>"'

        return mo.md(md_content)
    return (display_selections_markdown,)


@app.cell
def _(display_selections_markdown, json, mo, params_file, set_loaded_params):
    if True:  # load_btn.value:  # load params pressed
        with open(params_file, 'r') as _file:
            previous_params = json.load(_file)
        set_loaded_params(True)

    mo.ui.tabs({
        'Parameters loaded': display_selections_markdown(previous_params, 'Loaded Parameters'),
        'JSON file': mo.json(previous_params, label='Loaded PARAMS.JSON')
    })
    return (previous_params,)


@app.cell
def _(loaded_params):
    loaded_params()
    return


@app.cell
def _(
    Path,
    create_btn,
    filename,
    filepath_parent,
    loaded_params,
    mo,
    overwrite,
    save_path,
    today_dt,
):
    mo.stop(create_btn.value == False or loaded_params())
    _save_path = save_path  # make local copy of variable
    match _save_path.exists():
        case x if x and overwrite:  # overwrite == True
            with mo.redirect_stdout():
                print(
                    f'Output folder exists already in: {_save_path}, output folder will be replaced.')
        case y if y and (not overwrite):  # overwrite == False
            with mo.redirect_stdout():
                print(
                    f'Output folder exists already in: {_save_path}, a new output folder will be made.')
            _i = 0
            while _save_path.exists():  # if file already exist, append numbers
                _save_path = filepath_parent / \
                    f'{Path(filename).stem}_{today_dt}_{_i}'
                _i += 1
    # create output folder
    # parents False as csv file should be present in existing path
    _save_path.mkdir(exist_ok=True, parents=False)
    with mo.redirect_stdout():
        print(f'Created output folder at: {_save_path}')

    return


@app.cell
def _(filepath, pd):
    df = pd.read_csv(filepath)
    return (df,)


@app.cell
def _(df, mo):

    choice = mo.ui.switch(False, label="### Advanced")

    mo.vstack([
        mo.md('<br>'),
        mo.md("### CSV file loaded. Use the GUI below to explore the dataset."),
        mo.accordion(
            {'Click here to reveal data explorer': mo.ui.data_explorer(df)}),
    ])
    # use below if you want to save a transformation
    # transformed_df = mo.ui.dataframe(df)
    # transformed_df
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    <br>

    ---
    ## <span style='color:lightcoral'> Variables and metrics selection
    ...
    """
    )
    return


@app.cell
def _(df, loaded_params, mo, previous_params):
    # Create dropdowns for selecting columns, or fill in with loaded parameters
    subject_id_selector = mo.ui.multiselect(
        options=df.columns.tolist(), label="ID Variable", max_selections=1,
        value=[] if not loaded_params() else previous_params["subject_id_variable"]
    )
    sex_variable_selector = mo.ui.multiselect(
        options=df.columns.tolist(), label="Sex Variable", max_selections=1,
        value=[] if not loaded_params() else previous_params["sex_variable"]
    )
    grouping_variable_selector = mo.ui.multiselect(
        options=df.columns.tolist(), label="Grouping Variable", max_selections=1,
        value=[] if not loaded_params() else previous_params["grouping_variable"]
    )

    return (
        grouping_variable_selector,
        sex_variable_selector,
        subject_id_selector,
    )


@app.cell
def _(
    df,
    grouping_variable_selector,
    loaded_params,
    mo,
    previous_params,
    sex_variable_selector,
    subject_id_selector,
):
    # Register selections or set defaults
    id_variable = subject_id_selector.value
    sex_variable = sex_variable_selector.value
    group_variable = grouping_variable_selector.value

    # index selector (single index variable choices removed)
    selected_idx_options = set(id_variable+sex_variable+group_variable)
    filtered_idx_options = [option for i, option in enumerate(
        df.columns.tolist()) if (option not in selected_idx_options)]

    index_column_selector = mo.ui.multiselect(
        options=filtered_idx_options, label="Other Indices",
        value=[] if not loaded_params() else previous_params["index_variables"]
    )
    return group_variable, id_variable, index_column_selector, sex_variable


@app.cell
def _(
    df,
    group_variable,
    id_variable,
    index_column_selector,
    loaded_params,
    mo,
    previous_params,
    sex_variable,
):
    indices = index_column_selector.value

    # metric selector (index variable choices removed)
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    selected_options = set(indices+id_variable+sex_variable+group_variable)
    filtered_options = [option for i, option in enumerate(df.columns.tolist()) if (
        option not in selected_options) or (option in numeric_columns)]
    metric_columns_selector = mo.ui.multiselect(
        options=filtered_options, label="Metrics",
        value=[] if not loaded_params() else previous_params["metric_variables"]
    )

    # button to confirm choices
    submit_choices = mo.ui.run_button(label="Confirm Selections", kind='warn')
    var_choices = {}
    return metric_columns_selector, submit_choices, var_choices


@app.cell(hide_code=True)
def _(
    grouping_variable_selector,
    index_column_selector,
    metric_columns_selector,
    mo,
    sex_variable_selector,
    subject_id_selector,
    submit_choices,
    var_choices,
):
    # Column / Variable selection
    def create_input_row(selector_element, value_display_func, description_text="", description_color='#AAAAAA'):
        # Apply style to the selector to control its width and alignment
        # Ensure all selectors have the same width. '100%' makes it fill its container.
        styled_selector = selector_element.style(
            # box-sizing helps with consistent sizing
            {"text-align": "end", "box-sizing": "border-box", }
        )

        # The left column will contain the selector and its value display
        left_column_content = mo.vstack([
            styled_selector,
            mo.md(
                f"{value_display_func(selector_element.value)}"
                if selector_element.value
                else "&nbsp;"
            ).style({"color": "lightlavender"})
        ], justify='end', )

        # The right column will contain the description, with a smaller font size
        right_column_content = mo.md(
            # Use <small> tag for smaller text
            f"<medium>{description_text}</medium>"
        ).style({"color": description_color})

        return mo.hstack(
            [
                left_column_content,
                right_column_content
            ],
            widths=[0.3, 0.7],  # 30% for the left column, 70% for the right
            # Align the content of the hstack to the end (right)
            justify="end",
            align="start",     # Align content of this hstack to the top
            wrap=True,
        )

    def add_variable_input_row(choice_list: list, description: str, ):
        input_row = create_input_row(
            choice_list,
            lambda val: ',<br>'.join(sorted(val)),
            description)
        return input_row

    def add_metric_input_row(choice_list: list, description: str, ):
        input_row = create_input_row(
            choice_list,
            lambda val: f"""
            <span style='display: block; font-size:0.9em; line-height:1.5em'>
            {',  <br>'.join(sorted(val))}
            </span>
            """,
            # lambda val: ',  <br>'.join(sorted(val)) if len(val) < 5 else f"<span style='display: block; font-size:0.9em; line-height:1.5em'>{',  <br>'.join(sorted(val))}</span>",
            description)
        return input_row

    selectors = mo.vstack([
        mo.hstack([
            mo.md(f"### Select Columns for Analysis").style(
                {"flex-grow": "1"}),
            mo.md("### Descriptions").style({"flex-grow": "1"})
        ]),
        add_variable_input_row(
            subject_id_selector,
            "Select the column that uniquely identifies each subject in the dataset."
        ), mo.md('---'),
        add_variable_input_row(
            sex_variable_selector,
            "Select the column that defines the sex of each subject in the dataset."
        ), mo.md('---'),
        add_variable_input_row(
            grouping_variable_selector,
            """Select the column that defines the grouping or category for comparison within your data (e.g., condition [control vs treatment], delay[short vs long]).  
            <span style='font-size:0.8em; color:#888888'>Only one can be chosen to compare at a time for now, use filters below to restrict comparisons if many grouping variables are present.  \nFor example, if grouping by condition but don't want to collapse across sex, filter for male or female.</span>"""
        ), mo.md('---'),
        add_variable_input_row(
            index_column_selector,
            """Select columns that serve as unique identifiers for your data entries (e.g., patient ID, sample number).  
            These columns will not be included in the analysis as metrics.""",
        ), mo.md('---'),
        add_metric_input_row(
            metric_columns_selector,
            """Select the columns containing numerical data that you want to analyze as metrics.  
            These will be used for statistical computations."""
        ),
        submit_choices.style(
            {'width': 'fit-content', 'padding': '20px 20px'}).right()
    ])

    if submit_choices.value:
        var_choices.update({
            "subject_id_variable": subject_id_selector.value,
            "grouping_variable": grouping_variable_selector.value,
            "sex_variable": sex_variable_selector.value,
            "index_variables": index_column_selector.value,
            "metric_variables": metric_columns_selector.value,
        })
        all_choices = [column for sublist in var_choices.values()
                       for column in sublist]

    selectors
    return (all_choices,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## <span style='color:lightcoral'>Filters
    ... Desc for grouping
    """
    )
    return


@app.cell
def _(loaded_params, mo):
    def format_filters(filters: dict):

        where_clause = [
            {"column_id": column, "operator": "equals", "value": value}
            for column, value in filters.items()
        ]
        filter_transforms = []
        if where_clause:  # if filters present
            filter_transforms.append({
                "type": "filter_rows",
                "operation": "keep_rows",
                "where": where_clause
            })
        return filter_transforms

    # load filters button
    load_filters_btn = mo.ui.run_button(
        label='Load saved filters', disabled=not loaded_params())

    return format_filters, load_filters_btn


@app.cell
def _():
    # mo.ui.dataframe(prefiltered_df)
    return


@app.cell
def _(
    all_choices,
    df,
    format_filters,
    load_filters_btn,
    loaded_params,
    mo,
    previous_params,
    var_choices,
):
    def load_filters(df, filter_params):
        import operator
        from functools import reduce
        df_next = df.copy()
        # combine column==value paired filters
        filters = reduce(operator.and_, (df_next[column].eq(
            value) for column, value in filter_params.items()))
        # return filtered df
        return df_next[filters]

    # apply pre-filtering from selections above
    columns_included = df.columns if var_choices == {} else all_choices
    prefiltered_df = df[columns_included]

    if loaded_params():
        loaded_transforms = format_filters(previous_params['filters'])

    filter_ui = mo.ui.dataframe(prefiltered_df)
    # mo.output.append(filter_ui)
    if load_filters_btn.value:
        filter_ui._Initialized = False
        preloaded_args = filter_ui._args
        preloaded_args.initial_value["transforms"] = loaded_transforms
        filter_ui._initialize(preloaded_args)
        filter_ui._Initialized = True

    # if load_filters_btn.value:
    #     set_filters_loaded(True)

    # if load_filters_btn.value:  # if loading saved filters
    #     with mo.redirect_stdout():
    #         print('loaded filters')
    #     preloaded_args = mo.ui.dataframe(prefiltered_df)._args
    #     preloaded_args.initial_value["transforms"] = format_filters(previous_params['filters'])
        # filter_ui._initialize(preloaded_args)  # initialize with preset arguments
    # filter_ui._initialized = True
    return (filter_ui,)


@app.cell
def _(filter_ui, load_filters_btn, mo):

    mo.output.append(mo.md(f'''### Click to transform dataset 
             For example, to select only female subjects or a specific experimental group, or both:<br>
             &emsp; click on transform and choose "Filter Rows", then click on "+ Add".'''))

    mo.output.append(filter_ui)
    mo.output.append(load_filters_btn)
    mo.output.append(mo.md('<br>'))

    if load_filters_btn.value:  # if loading saved filters
        with mo.redirect_stdout():
            print('loaded filters')

    return


@app.cell
def _(filter_ui):
    # retrieve filters applied
    filter_type = filter_ui._last_transforms.transforms[0].type.value
    filter_operator = filter_ui._last_transforms.transforms[0].operation
    final_transforms = {}

    transforms = filter_ui._last_transforms.transforms
    if transforms != []:
        for transform in transforms:
            if transform.type.value == 'filter_rows':
                for filter in transform.where:
                    col = filter.column_id
                    val = filter.value
                    final_transforms.update({col: val})
    return (final_transforms,)


@app.cell
def _(
    Path,
    filename,
    filepath,
    filter_ui,
    final_transforms,
    json,
    mo,
    overwrite,
    save_path,
    today_dt,
    var_choices,
):
    filtered_df = filter_ui.value

    # update parameters output with filename and filters
    params_output = dict(
        filename=filename,
        filepath=filepath,
        **var_choices,
        filters=final_transforms,
    )

    def save_parameters(params_output: dict, location: str | Path = None):
        global filepath
        global save_path
        global overwrite
        global today_dt
        if location is None:
            # by default will save where the file was found
            location = save_path
        assert location.exists(), f'Destination folder not found:  {location}'
        file_out_name = f'params_{today_dt}.json'
        file_out_path = location / file_out_name
        if not overwrite:
            i = 0
            while file_out_path.exists():  # if file already exist, append numbers
                file_out_path = location / f'params_{today_dt}_{i}.json'
                i += 1

        with mo.redirect_stdout():
            print(f'Saving parameters to `{file_out_path}`...')
        with open(file_out_path, 'w') as file:
            json.dump(params_output, file, indent=4)
        with mo.redirect_stdout():
            print('...file saved.')
        return file_out_path

    # button to save parameters
    save_params = mo.ui.run_button(label="Save parameters", kind='warn')

    return filtered_df, params_output, save_parameters, save_params


@app.cell
def _(
    display_selections_markdown,
    mo,
    params_output,
    save_parameters,
    save_params,
):
    # display final selections
    mo.output.append(mo.vstack([
        display_selections_markdown(params_output),
        save_params.right()
    ]))

    if save_params.value:
        _file_out_path = save_parameters(params_output)

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    <br>
    #  <span style='color:coral'> Analysis outputs
    * ...
    """
    )
    return


@app.cell
def _(mo, var_choices):
    # Prevent progression if choices not made yet for columns and filtering.
    mo.stop(var_choices == {} or var_choices is None, mo.md(
        "### Confirm variables, metrics and filtering to continue..."))

    return


@app.cell
def _(filtered_df, sf, var_choices):
    # set up raw_data import
    var_properties = var_choices
    raw_data = sf.DataDF(filtered_df, var_properties)
    # raw_data.get_rawdf(str(filepath))
    raw_data.fix_columns()
    return (raw_data,)


@app.cell
def _(raw_data):

    pldf = raw_data.get_filt_df({}, [])
    pldf.scale_data(scaletype=2)
    pldf.get_si(exclude=[])
    pldf.map_colors(
        color_set=['viridis_r', 'magma_r'],
        cmap_rng=(.2, 0.8)
    )
    return (pldf,)


@app.cell
def _(mo, pldf, raw_data):
    mo.vstack([
        mo.md("### Preview datasets"),
        mo.ui.tabs({
            "Raw Data Preview": mo.plain(raw_data.raw_df.head(3)),
            "Scaled Data Preview": mo.plain(pldf.df_scaled.head(3))
        }),
        mo.md("<br>")
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## <span style='color:lightcoral'> Distribution plots
    Desc...

    Click on each  <span style='border:2px solid coral;padding-right: 5px'>:arrow_down_small: plot title</span> to reveal plot.
    """
    )
    return


@app.cell
def _(mo):
    # --- Marimo States ---
    # Initial range for the slider. This will be updated by the dropdown.
    get_range, set_range = mo.state([-12, 13])

    # State for the selected comparison metric
    get_compare_metric, set_compare_metric = mo.state('SI_scores')
    return get_range, set_compare_metric, set_range


@app.cell
def _(get_range, mo, np, pldf, set_compare_metric, set_range, sf, var_choices):

    # Define UI for rangex
    rangex_slider = mo.ui.range_slider(
        label="X-axis range",
        start=-20, stop=20, step=0.5,
        show_value=False, debounce=True,
        value=get_range(),    # The current value of the slider comes from the state
        on_change=set_range,  # When the slider is manually moved, update the state
        full_width=False
    )

    rangey_slider = mo.ui.slider(
        # label=f"Y-axis Range:",
        start=0, stop=100, step=5,
        orientation='vertical',
        value=30, debounce=True, full_width=True
        # on_change=set_yrange
    )

    # Define function for updating range
    def on_metric_change(new_metric_value):
        # Update the selected metric state
        set_compare_metric(new_metric_value)

        data = pldf.df_scaled.copy()
        # --- Logic to automatically set slider range based on new_metric_value ---
        if new_metric_value in data.columns:
            # Get min/max from the actual data for the selected column
            min_val = data[new_metric_value].min()
            max_val = data[new_metric_value].max()

            # Add a little padding for better visualization
            padding = (max_val - min_val) * 0.3
            padding = round(padding)
            # Ensure padding doesn't result in inverted range or too small
            if min_val == max_val:
                min_val -= 1
                max_val += 1

            # Update the range state, which will automatically update rangex_slider.value
            set_range([np.floor(min_val, dtype=float) - padding,
                      np.ceil(max_val, dtype=float) + padding])
        elif new_metric_value == 'SI_scores':
            # Specific range for 'SI_scores' if it's not a direct column
            set_range([-12, 13])  # Or calculate if SI_scores is derived
        else:
            # Fallback default
            # A reasonable default if no data-driven range is found
            set_range([-15, 15])

    # Define UI for compare_metric
    # Combine SI_scores with other metrics_included
    metrics_included = var_choices['metric_variables']
    cleaned_metrics = sf.fix_names(metrics_included)  # apply column name fix
    available_compare_metrics = ['SI_scores'] + cleaned_metrics

    compare_metric_selector = mo.ui.dropdown(
        options=available_compare_metrics,
        value='SI_scores',
        label="Select Metric for Comparison",
        on_change=on_metric_change  # Call custom handler
    )

    save_distplot_button = mo.ui.run_button(label='Save plot', kind='warn')
    save_distplots_button = mo.ui.run_button(label='Save plot', kind='warn')
    save_scatter_button = mo.ui.run_button(label='Save plot', kind='warn')
    save_correlations_button = mo.ui.run_button(label='Save plot', kind='warn')
    save_pca_biplot_button = mo.ui.run_button(label='Save plot', kind='warn')
    save_pca_matrix_button = mo.ui.run_button(label='Save plot', kind='warn')

    return (
        compare_metric_selector,
        rangex_slider,
        rangey_slider,
        save_distplot_button,
        save_distplots_button,
        save_scatter_button,
    )


@app.cell
def _(
    compare_metric_selector,
    get_range,
    mo,
    pldf,
    rangex_slider,
    rangey_slider,
    save_distplot_button,
    save_path,
):
    SI_plot_params = dict(
        set_size_params=(3.5, 3.5),
        hide_text=False,
    )

    def plot_distribution(compare_metric: str, max_y: int = None, rangex: list = []):

        if max_y is None:
            max_y = rangey_slider.value
        if rangex == []:
            rangex = get_range()

        # Generate the Altair plot using the selected UI values
        altplot_interactive = pldf.compare_dists_altair(
            # compare_metric=compare_metric_selector.value,
            compare_metric=compare_metric,
            filters={},
            max_y=max_y,          # y axis max
            rangex=rangex,  # x axis range
            user_labels=None,  # or eg, ['delayed', 'immediate']
            legend=True,       # add legends
            set_size_params=(3.5, 3.5),  # size of plot,
            hide_text=False,    # hide all text in plot
            alpha1=0.5,
            alpha2=0.5
        ).properties(width=350, height=350).interactive()
        return altplot_interactive

    def dist_plot_layout(compare_metric: str = ''):
        if compare_metric == '':
            compare_metric = compare_metric_selector.value

        # Define and display the UI elements and interactive Altair chart
        right_content = mo.vstack([
            mo.md(
                f'Y-axis max: <br>{rangey_slider.value}').style({'text-align': 'center'}),
            rangey_slider.center()
        ],  align='center', gap=0.5).style({'width': '80px', 'align-content': 'center'})

        xlim = [float(x) for x in get_range()]  # retrieve x range values
        dist_layout = mo.vstack([
            compare_metric_selector,
            mo.hstack([
                rangex_slider,
                mo.md(f'{xlim[0]} :::::&nbsp;{xlim[1]}')
            ], justify='start', align='end',  gap=1).center(),
            mo.hstack([
                # mo.ui.altair_chart(altplot_interactive),
                distribution_plot := plot_distribution(compare_metric),
                right_content,
                mo.md("")
            ], justify='start', align='stretch', gap=0)
        ], align='stretch', gap=1)
        return dist_layout, distribution_plot

    with mo.capture_stdout() as _buffer:
        dist_layout, dist_plot = dist_plot_layout()

    def save_plot(altair_plot, plot_filepath, format: str = 'png'):
        altair_plot.save(plot_filepath, format=format)
        print(f'\nSaved plot to "{plot_filepath}".')

    # Layout
    mo.output.append(mo.accordion({
        "### <span style='border: 2px solid coral;padding:4px 5px;display:inline-block'>:arrow_down_small:Single metric distribution comparison": mo.vstack([
            mo.md(f"""
            Desc...  <br>
            <br>
            """).style(color="white"),
            dist_layout,
            save_distplot_button.right(),
        ])}, lazy=True))

    _console_logs = ['<br>'+line for line in _buffer.getvalue().split('\n')]
    # [mo.output.append(line) for line in [mo.md(
    #     f"<span style='display:block;font-size:0.7em;line-height:0.3em;font-family:monospace'>{log}") for log in _console_logs]]
    _formatted_logs = [
        f"<span style='display:block;font-size:13px;line-height:8px;font-family:monospace'>{log}" for log in _console_logs]

    mo.output.append(
        mo.md(
            f"""/// details | Console outputs
                type: info 
            {mo.as_html(mo.md('<br>'.join(_formatted_logs)))}
            ///"""
        ))

    if save_distplot_button.value:
        plot_filepath = save_path / \
            f'distplot.png'  # TODO: change label
        with mo.redirect_stdout():
            save_plot(dist_plot, plot_filepath)

    return plot_distribution, save_plot


@app.cell
def _(
    alt,
    mo,
    pldf,
    plot_distribution,
    save_distplots_button,
    save_path,
    save_plot,
    sf,
    var_choices,
):
    metric_names = sf.fix_names(
        var_choices['metric_variables'])  # whitespaces and other

    dist_plots = []
    with mo.capture_stdout() as _buffer:
        for metric in metric_names:
            plot_data = pldf.df_scaled[metric]
            xmin = plot_data.min().round()-2
            xmax = plot_data.max().round()+2

            mean = plot_data.mean()
            std_dev = plot_data.std()
            max_y = 100
            dist_plots.append(plot_distribution(
                metric, max_y=max_y, rangex=[xmin, xmax]))

    # concatenate the list of charts
    final_chart = alt.hconcat(*dist_plots).resolve_scale(y='independent')

    # Layout
    mo.output.append(mo.accordion({
        "### <span style='border: 2px solid coral;padding:4px 5px;display:inline-block'>:arrow_down_small:Distribution comparisons for chosen metrics": mo.vstack([
            mo.md(f"""
            Desc...  <br>
            <br>
            """).style(color="white"),
            final_chart,
            save_distplots_button.right()
        ])}, lazy=True))

    _console_logs = [line for line in _buffer.getvalue().split('\n')[1:]]
    # test=[mo.output.append(line) for line in [mo.md(
    #     f"<span style='display:block;font-size:0.7em;line-height:0.3em;font-family:monospace'>{log}") for log in _console_logs]]
    _formatted_logs = [
        f"<span style='display:block;font-size:13px;line-height:1.25em;font-family:monospace'>{log}" for log in _console_logs]

    mo.output.append(
        mo.md(
            f"""/// details | Console outputs
                type: info 
            {mo.as_html(mo.md('<br>'.join(_formatted_logs)))}
            ///"""
        ))

    if save_distplots_button.value:
        dist_plots_filepath = save_path / \
            'distplots.png'  # TODO: change label
        with mo.redirect_stdout():
            save_plot(final_chart, dist_plots_filepath)
    return


@app.cell
def _(mo, save_scatter_button):

    # Layout
    mo.output.append(mo.accordion({
        "### <span style='border: 2px solid coral;padding:4px 5px;display:inline-block'>:arrow_down_small:Scatterplot metrics": mo.vstack([
            mo.md(f"""
            Desc...  <br>
            <br>
            """),
            save_scatter_button.right()
        ])}))
    return


@app.cell
def _(mo):

    # Layout
    mo.output.append(mo.accordion({
        "### <span style='border: 2px solid coral;padding:4px 5px;display:inline-block'>:arrow_down_small:Correlation scatter metrics": mo.vstack([
            mo.md(f"""
            Desc...  <br>
            <br>
            """).style(color="white"),
        ])}))
    return


@app.cell
def _(mo):

    # Layout
    mo.output.append(mo.accordion({
        "### <span style='border: 2px solid coral;padding:4px 5px;display:inline-block'>:arrow_down_small:PCA analysis": mo.vstack([
            mo.md(f"""
            Desc...  <br>
            <br>
            """).style(color="white"),
        ])}))
    return


@app.cell
def _(mo):
    mo.md(r"""# To do""")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.stop(True)
    mo.md(
        r"""
    <!-- ## sequence
    - change nomenclature and organization
        - index and group columns unclear
        - maybe ID column, independent var columns, experimental grouping columns and dependent (metric) columns
    - build DataDF
    - get raw --> make part of Class init
    - some preprocessing
        - fix columns, ie whitespaces and stuff
        - define_groups - option to make a composite group
    - set up filter if wanted
        - can use the dataframe mode for this or maybe something else and then submit transformed df
    - gets scaled data
    - gets social index
    - map colors (NEEDS REWORK)
    - sub filters for comparison?
        - change to be able to make multiple comparisons
        - multiple choice of index columns for grouping
    - 1st plots - altair/plotly? may make adding filtering redundant
        - compare_dists for SI scores - fixed, but maybe filtering? radial for sex?
        - dists by groupvar for other metrics - multiselect
    - GMM grid search and scoring, scatterplot
        - `gmm_bic_score`, `gmm_grid_search`, `gmm_cluster_plot`
    - PCA steps
        - `make_cat` make categoricals? -- add to class maybe
        - `do_pca` returns pca, principal_components, pc_evr (`pca.explained_variance_ratio_`)
        - pca_biplot
        - `get_pca_corrs` with `plot_xcorrs`
    - scatter SI plots + correl plots (needs tweaking for axis / fitting)
        - `si_plots`
        - `si_corrs` can these two be merged more or simplified?
        - uses `plot_dicts` with axis limits for given metrics
            - selections and sliders? -->
    """
    )
    return


@app.cell
def _(mo):
    callout = mo.callout(mo.md("### This is a callout"), kind="info")
    mo.accordion({"Columns selected as indices": callout}, multiple=True)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    <!-- ## elements to add:

    - [array](https://docs.marimo.io/api/inputs/array/) for multiple text or other inputs
    - text area for notes / metadata
    - [batch](https://docs.marimo.io/api/inputs/batch/) ui to preformat the layout for other ui elements
    - [code_editor](https://docs.marimo.io/api/inputs/code_editor/) for adding code snippets
    - [data explorer](https://docs.marimo.io/api/inputs/data_explorer/) for exploring dataframes interactively
        - suggests plots and lets you add/remove encodings (variables) to explore analyses
    - [dictionary](https://docs.marimo.io/api/inputs/dictionary/) to show nested such as selections for parameters etc
    - [file_browser](https://docs.marimo.io/api/inputs/file_browser/) for selecting files
    - [image](https://docs.marimo.io/api/inputs/image/) for displaying images
    - [radio]](https://docs.marimo.io/api/inputs/radio/) for selecting single options from a list of a few choices
    - [slider](https://docs.marimo.io/api/inputs/slider/) for selecting numeric values from a range
        - [range_slider](https://docs.marimo.io/api/inputs/range_slider/) for selecting a range within a range
    - [switch](https://docs.marimo.io/api/inputs/switch/) for toggling boolean values
    - [mermaid](https://docs.marimo.io/api/inputs/mermaid/) for creating diagrams and flowcharts
    - [media](https://docs.marimo.io/api/inputs/media/) for displaying images, videos, and other media files
    - [plotting](https://docs.marimo.io/api/inputs/plotting/) for creating interactive plots and visualizations

    also figure out interplay with [SQL db and models](https://docs.marimo.io/guides/working_with_data/sql/#connecting-to-a-custom-database), and loguru -->
    """
    )
    return


if __name__ == "__main__":
    app.run()
