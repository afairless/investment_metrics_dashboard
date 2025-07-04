
import base64
import datetime
import io

from typing import Union, Any
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, fields
from math import log, ceil, floor
from datetime import date

import pandera.pandas as pdr
from pandera.typing import DataFrame as pdr_DataFrame

import pyarrow.parquet as pq

import plotly.graph_objects as go
from plotly import subplots 
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, callback_context


from ..app.sidebar import (
    sidebar, 
    buy_sell_date_choice, 
    convert_hour_to_string, 
    slider_style01)

from ..data_pipeline.transform_filled_orders import (
    list_of_strings_without_nans,
    convert_to_filled_orders_by_position_change,
    convert_to_filled_orders_by_ongoing_position,
    convert_to_filled_orders_by_symbol_day)

from ..schemas.data_pipeline_schemas import (
    FilledOrdersForDashboard,
    FilledOrdersByPositionChange,
    FilledOrdersByOngoingPosition,
    FilledOrdersBySymbolDay)

from ..schemas.app_schemas import (
    FilledOrdersByPositionChangeDatetime,
    FilledOrdersByOngoingPositionDatetime,
    FilledOrdersBySymbolDayDatetime)



date_colnames01 = [
    e 
    for e in FilledOrdersByPositionChange.__schema__.dtypes 
    if str(FilledOrdersByPositionChange.__schema__.dtypes[e]) == 'datetime64[ns]']
date_colnames02 = [
    e 
    for e in FilledOrdersByOngoingPosition.__schema__.dtypes 
    if str(FilledOrdersByOngoingPosition.__schema__.dtypes[e]) == 'datetime64[ns]']
date_colnames03 = [
    e 
    for e in FilledOrdersBySymbolDay.__schema__.dtypes 
    if str(FilledOrdersBySymbolDay.__schema__.dtypes[e]) == 'datetime64[ns]']
date_colnames04 = ['min_date', 'max_date']

date_colnames = (
    date_colnames01 + date_colnames02 + date_colnames03 + date_colnames04)



##################################################
# DASHBOARD LAYOUT
##################################################

# link fontawesome to get the chevron icons
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    # these meta_tags ensure content is scaled correctly on different devices
    # see: https://www.w3schools.com/css/css_rwd_viewport.asp for more
    meta_tags=[
        {'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])

smaller_plot_style = {'width': '120vh', 'height': '60vh'}
plot_style = {'width': '120vh', 'height': '80vh'}
taller_plot_style = {'width': '120vh', 'height': '95vh'}
table_style01 = {'width': '120vh', 'height': '35vh'}
table_style02 = {'width': '120vh', 'height': '80vh'}

profit_table_table = html.Div( 
    children=dash.dcc.Graph(id='profit-table', style=table_style01), 
    className='page-content')
orders_table_table = html.Div( 
    children=dash.dcc.Graph(id='orders-table', style=table_style02), 
    className='page-content')
number_of_positions_by_date_plot_graph = html.Div( 
    children=dash.dcc.Graph(
        id='number-of-positions-by-date-plot', style=smaller_plot_style), 
    className='page-content')
number_of_positions_by_gain_loss_plot_graph = html.Div( 
    children=dash.dcc.Graph(
        id='number-of-positions-by-gain-loss-plot', style=plot_style), 
    className='page-content')
balance_change_by_position_chronologically_graph = html.Div( 
    children=dash.dcc.Graph(
        id='balance-change-by-position-chronologically', 
        style=taller_plot_style), 
    className='page-content')
cumulative_balance_change_by_position_chronologically_graph = html.Div( 
    children=dash.dcc.Graph(
        id='cumulative-balance-change-by-position-chronologically', 
        style=plot_style), 
    className='page-content')
cumulative_price_change_per_share_by_position_chronologically_graph = html.Div( 
    children=dash.dcc.Graph(
        id='cumulative-price-change-per-share-by-position-chronologically',
        style=plot_style), 
    className='page-content')
price_change_per_share_by_position_chronologically_graph = html.Div( 
    children=dash.dcc.Graph(
        id='price-change-per-share-by-position-chronologically', 
        style=taller_plot_style), 
    className='page-content')
price_percentage_change_by_position_chronologically_graph = html.Div( 
    children=dash.dcc.Graph(
        id='price-percentage-change-by-position-chronologically', 
        style=taller_plot_style), 
    className='page-content')
position_hold_times_graph = html.Div( 
    children=dash.dcc.Graph(id='position-hold-times', style=plot_style), 
    className='page-content')
position_volumes_graph = html.Div( 
    children=dash.dcc.Graph(id='position-volumes', style=plot_style), 
    className='page-content')
position_commissions_graph = html.Div( 
    children=dash.dcc.Graph(id='position-commissions', style=plot_style), 
    className='page-content')
spent_outflow_by_date_plot_graph = html.Div( 
    children=dash.dcc.Graph(id='spent-outflow-by-date-plot', style=smaller_plot_style), 
    className='page-content')
balance_change_by_day_graph = html.Div( 
    children=dash.dcc.Graph(
        id='balance-change-by-day', style=taller_plot_style), 
    className='page-content')
price_change_per_share_by_day_graph = html.Div( 
    children=dash.dcc.Graph(
        id='price-change-per-share-by-day', style=taller_plot_style), 
    className='page-content')


app.layout = html.Div([
    dcc.Location(id='url'), 
    sidebar, 
    dcc.Store(id='table-mask'),
    dcc.Store(id='orders'),
    dcc.Store(id='orders-position-change-datetime'),
    dcc.Store(id='orders-ongoing-position-datetime'),
    dcc.Store(id='orders-symbol-day-datetime'),
    dcc.Store(id='selected-orders'),
    html.Div(
        children=html.H1('Investment/Trading Metrics'), 
        className='page-content'),
    profit_table_table, 
    orders_table_table,
    number_of_positions_by_gain_loss_plot_graph,
    number_of_positions_by_date_plot_graph,
    cumulative_balance_change_by_position_chronologically_graph,
    cumulative_price_change_per_share_by_position_chronologically_graph,
    balance_change_by_position_chronologically_graph,
    price_change_per_share_by_position_chronologically_graph,
    price_percentage_change_by_position_chronologically_graph,
    position_hold_times_graph,
    position_volumes_graph,
    spent_outflow_by_date_plot_graph,
    position_commissions_graph,
    balance_change_by_day_graph,
    price_change_per_share_by_day_graph,
    ])



##################################################
# BASIC CONVERSION UTILITIES
##################################################


def convert_columns_to_date_type(df: pd.DataFrame) -> pd.DataFrame:

    # in converting JSON to Pandas DataFrame, Pandera coerces date columns to 
    #   'int64' instead of 'datetime64[ns]', so convert them
    for e in date_colnames:
        if e in df.columns:
            df[e] = pd.to_datetime(df[e], unit='ms')

    return df


def convert_orders_json_to_df(
    df_json: str, convert_dates: bool=True) -> pd.DataFrame:
    """
    Convert orders tables from JSON to DataFrame

    Pandas 'read_json' parameter 'convert_dates' is set to 'False' to guarantee
        clear date-converting behavior: 
            if this function's 'convert_dates' parameter is 'False', dates will 
                not be converted
            if this function's 'convert_dates' parameter is 'True', only columns
                with names in 'date_colnames' will be converted, and not any
                other columns that the Pandas API may specify
    """

    empty_df_json = '{"columns":[],"index":[],"data":[]}'

    if df_json == empty_df_json:
        df = pd.DataFrame()
    else:
        df = pd.read_json(
            io.StringIO(df_json), orient='split', date_unit='ms', 
            convert_dates=False)
    assert isinstance(df, pd.DataFrame)

    if convert_dates:
        df = convert_columns_to_date_type(df)

    return df



##################################################
# PARAMETERS FOR SIDEBAR USER SELECTIONS
##################################################
# Sidebar menus allow the user to select/filter the view of the data in 
#   displayed plots and tables
# Set parameters for these menus based on the selected table
#   e.g., from what range of dates can the user choose?
#   e.g., from what range of equity prices can the user choose?
##################################################


empty_df_json = '{"columns":[],"index":[],"data":[]}'

@app.callback(
    Output('sidebar', 'className'),
    Input('sidebar-toggle', 'n_clicks'),
    State('sidebar', 'className'))
def toggle_classname(n, classname):
    if n and classname == '':
        return 'collapsed'
    return ''


@app.callback(
    Output('selected-orders', 'data'), 
    [Input('url', 'pathname'),
     Input('orders-position-change-datetime', 'data'), 
     Input('orders-ongoing-position-datetime', 'data'), 
     Input('orders-symbol-day-datetime', 'data')])
def select_orders_table(
    pathname: str, orders_position_change: str, orders_ongoing_position: str,
    orders_symbol_day: str) -> str:
    """
    User may choose to see plots and statistics from 3 tables:
        1) Orders Position Change
        2) Orders Ongoing Position
        3) Orders Symbol Day
    """

    if pathname == '/ongoing-position':
        selected_json = orders_ongoing_position
    elif pathname == '/symbol-day':
        selected_json = orders_symbol_day
    else:
        selected_json = orders_position_change

    assert isinstance(selected_json, str)

    return selected_json


@app.callback(
    Output('group-description-01-text', 'is_open'), 
    Input('group-description-01-collapse', 'n_clicks'),
    State('group-description-01-text', 'is_open'))
def toggle_grouping_description_01(n: int, is_open: bool) -> bool:
    """
    Hide or reveal description of grouping transactions into positions
    """

    if n:
        return not is_open
    else:
        return is_open


@app.callback(
    Output('group-description-02-text', 'is_open'), 
    Input('group-description-02-collapse', 'n_clicks'),
    State('group-description-02-text', 'is_open'))
def toggle_grouping_description_02(n: int, is_open: bool) -> bool:
    """
    Hide or reveal description of grouping transactions into positions
    """

    if n:
        return not is_open
    else:
        return is_open


@app.callback(
    Output('group-description-03-text', 'is_open'), 
    Input('group-description-03-collapse', 'n_clicks'),
    State('group-description-03-text', 'is_open'))
def toggle_grouping_description_03(n: int, is_open: bool) -> bool:
    """
    Hide or reveal description of grouping transactions into positions
    """

    if n:
        return not is_open
    else:
        return is_open


def round_power_of_ten(value: float, high_order_magnitude=True) -> float:

    if high_order_magnitude:
        return ceil(log(value, 10) * 100) / 100
    else:
        return floor(log(value, 10) * 100) / 100


@app.callback(
    [Output('date-range', 'min_date_allowed'), 
     Output('date-range', 'max_date_allowed'), 
     Output('date-range', 'initial_visible_month'), 
     Output('date-range', 'start_date'), 
     Output('date-range', 'end_date')], 
     Input('selected-orders', 'data')) 
def date_range_picker_parameters(
    df_json: str) -> tuple[datetime.date, datetime.date, datetime.date, 
    datetime.date, datetime.date]:

    empty_df_json = '{"columns":[],"index":[],"data":[]}'

    if df_json == empty_df_json:
        earliest_date = date.today()
        latest_date = date.today()
        default_start_date = date.today()
    else:
        df = convert_orders_json_to_df(df_json)
        date_colnames_local = [e for e in date_colnames if e in df.columns]

        earliest_date = df[date_colnames_local].min().min().date()

        today = datetime.datetime.now().date()
        interval_before_today = datetime.timedelta(days=180)
        latest_date = df[date_colnames_local].max().max().date()
        interval_before_today_date = today - interval_before_today 

        if latest_date > interval_before_today_date:
            default_start_date = today - interval_before_today 
        else:
            # use 180 days ago as default start date
            default_start_date = earliest_date

    return earliest_date, latest_date, latest_date, default_start_date, latest_date


@app.callback(
    [Output('hour-of-day-min', 'children'), 
     Output('hour-of-day-max', 'children')], 
     Input('hour-of-day', 'value'))
def hour_of_day_slider_values(hour_of_day: list[float]) -> tuple[str, str]:

    return (
        convert_hour_to_string(hour_of_day[0]), 
        convert_hour_to_string(hour_of_day[1]))


@app.callback(
    [Output('fill-price-buy-min', 'children'), 
     Output('fill-price-buy-max', 'children')], 
     Input('fill-price-buy', 'value'))
def fill_price_buy_slider_values(
    fill_price_buy: list[float]) -> tuple[float, float]:

    return (
        round(10 ** fill_price_buy[0], 2),
        round(10 ** fill_price_buy[1], 2))


@app.callback(
    [Output('fill-price-buy', 'min'), 
     Output('fill-price-buy', 'max'), 
     Output('fill-price-buy', 'value')], 
     Input('selected-orders', 'data')) 
def fill_price_buy_slider_range(
    df_json: str) -> tuple[float, float, tuple[float, float]]:

    df = convert_orders_json_to_df(df_json, False)

    range_min = -2
    if 'fill_price_buy' in df.columns:
        range_max = 1.02 * round_power_of_ten(df['fill_price_buy'].max())
    elif 'buy_mean_fill_price' in df.columns:
        range_max = 1.02 * round_power_of_ten(df['buy_mean_fill_price'].max())
    else:
        # TODO TEST/CHECK:  this 'else' might no longer be necessary
        # renders slider inoperable when needed column is absent
        range_max = range_min

    return range_min, range_max, (range_min, range_max)


@app.callback(
    [Output('fill-price-sell-min', 'children'), 
     Output('fill-price-sell-max', 'children')], 
     Input('fill-price-sell', 'value'))
def fill_price_sell_slider_values(
    fill_price_sell: list[float]) -> tuple[float, float]:

    return (
        round(10 ** fill_price_sell[0], 2), 
        round(10 ** fill_price_sell[1], 2))


@app.callback(
    [Output('fill-price-sell', 'min'), 
     Output('fill-price-sell', 'max'), 
     Output('fill-price-sell', 'value')], 
     Input('selected-orders', 'data')) 
def fill_price_sell_slider_range(
    df_json: str) -> tuple[float, float, tuple[float, float]]:

    if df_json == empty_df_json:
        return 0, 0, (0, 0)

    df = convert_orders_json_to_df(df_json, False)

    range_min = -2
    if 'fill_price_sell' in df.columns:
        range_max = 1.02 * round_power_of_ten(df['fill_price_sell'].max())
    elif 'sell_mean_fill_price' in df.columns:
        range_max = 1.02 * round_power_of_ten(df['sell_mean_fill_price'].max())
    else:
        # renders slider inoperable when needed column is absent
        range_max = range_min

    return range_min, range_max, (range_min, range_max)


@app.callback(
    [Output('commission-buy-sell-min', 'children'), 
     Output('commission-buy-sell-max', 'children')], 
     Input('commission-buy-sell', 'value'))
def commission_buy_sell_slider_values(
    commission_buy_sell: list[float]) -> tuple[float, float]:

    return (
        round(10 ** commission_buy_sell[0], 2), 
        round(10 ** commission_buy_sell[1], 2))


@app.callback(
    [Output('commission-buy-sell', 'min'), 
     Output('commission-buy-sell', 'max'), 
     Output('commission-buy-sell', 'value')], 
     Input('selected-orders', 'data')) 
def commission_buy_sell_slider_range(
    df_json: str) -> tuple[float, float, tuple[float, float]]:

    if df_json == empty_df_json:
        return 0, 0, (0, 0)

    df = convert_orders_json_to_df(df_json, False)

    range_min = -2

    if 'commission_cost' in df.columns:
        range_max = 1.02 * round_power_of_ten(df['commission_cost'].max())
    else:
        commission_colnames = ['commission_cost_buy', 'commission_cost_sell']
        commission_cost_max = df[commission_colnames].sum(axis=1).max()
        range_max = 1.02 * round_power_of_ten(commission_cost_max)

    return range_min, range_max, (range_min, range_max)


@app.callback(
    [Output('balance-change-min', 'children'), 
     Output('balance-change-max', 'children')], 
     Input('balance-change', 'value'))
def balance_change_slider_values(
    balance_change: list[float]) -> tuple[float, float]:

    return (
        round(balance_change[0], 2),
        round(balance_change[1], 2))


@app.callback(
    [Output('balance-change', 'min'), 
     Output('balance-change', 'max'), 
     Output('balance-change', 'value')], 
     Input('selected-orders', 'data')) 
def balance_change_slider_range(
    df_json: str) -> tuple[float, float, tuple[float, float]]:

    if df_json == empty_df_json:
        return 0, 0, (0, 0)

    df = convert_orders_json_to_df(df_json, False)

    range_min = df['balance_change'].min()
    range_max = df['balance_change'].max()

    return range_min, range_max, (range_min, range_max)


@app.callback(
    [Output('balance-change-commission-min', 'children'), 
     Output('balance-change-commission-max', 'children')], 
     Input('balance-change-commission', 'value'))
def balance_change_commission_slider_values(
    balance_change_commission: list[float]) -> tuple[float, float]:

    return (
        round(balance_change_commission[0], 2), 
        round(balance_change_commission[1], 2))


@app.callback(
    [Output('balance-change-commission', 'min'), 
     Output('balance-change-commission', 'max'), 
     Output('balance-change-commission', 'value')], 
     # returning marks to RangeSlider produces error
     #Output('balance-change-commission', 'marks'), 
     Input('selected-orders', 'data')) 
def balance_change_commission_slider_range(
    df_json: str) -> tuple[float, float, tuple[float, float]]:

    if df_json == empty_df_json:
        return 0, 0, (0, 0)

    df = convert_orders_json_to_df(df_json, False)

    range_min = df['balance_change_commission'].min()
    range_max = df['balance_change_commission'].max()

    #marks = {
    #    i: str(i) for i in range( 
    #        int(df['balance_change_commission'].min()), 
    #        int(df['balance_change_commission'].max()),
    #        # show about 9 marks on scale
    #        ((int(df['balance_change_commission'].max()) - 
    #          int(df['balance_change_commission'].min())) // 8))},

    return range_min, range_max, (range_min, range_max)#, marks


@app.callback(
    [Output('shares-num-fill-min', 'children'), 
     Output('shares-num-fill-max', 'children')], 
     Input('shares-num-fill', 'value'))
def shares_num_fill_slider_values(
    shares_num_fill: list[float]) -> tuple[float, float]:

    return (
        round(10 ** shares_num_fill[0], 2), 
        round(10 ** shares_num_fill[1], 2))


@app.callback(
    [Output('shares-num-fill', 'min'), 
     Output('shares-num-fill', 'max'), 
     Output('shares-num-fill', 'value')], 
     Input('selected-orders', 'data')) 
def shares_num_fill_slider_range(
    df_json: str) -> tuple[float, float, tuple[float, float]]:

    if df_json == empty_df_json:
        return 0, 0, (0, 0)

    df = convert_orders_json_to_df(df_json, False)

    if 'shares_num_fill' in df.columns:
        shares_colname = 'shares_num_fill'
    else:
        shares_colname = 'match_shares_num_fill'

    range_min = round_power_of_ten(df[shares_colname].min(), False) 
    range_max = 1.02 * round_power_of_ten(df[shares_colname].max())

    return range_min, range_max, (range_min, range_max)


@app.callback(
    [Output('market-time-of-day', 'options'), 
     Output('market-time-of-day', 'value')], 
     Input('selected-orders', 'data')) 
def market_time_of_day_categories(
    df_json: str) -> tuple[list[dict[str, str]], pd.Index]:

    if df_json == empty_df_json:
        return [{'': ''}], pd.Index([])

    df = convert_orders_json_to_df(df_json, False)
    
    # 'try... except' is inelegant; might should replace with a type check
    try:
        value = df['max_date_market_time'].cat.categories
        options = [{'label': e, 'value': e.lower()} for e in value.str.title()]
    except:
        value = df['max_date_market_time'].unique()
        options = [{'label': e.title(), 'value': e} for e in value]

    return options, value


@app.callback(
    [Output('all-stock-symbol', 'value'), 
     Output('stock-symbol', 'options'), 
     Output('stock-symbol', 'value')], 
    [Input('selected-orders', 'data'),
     Input('all-stock-symbol', 'value'), 
     Input('stock-symbol', 'value')])
def stock_symbol_categories(
    df_json: str, all_selected: list[str], singly_selected: list[str],
    ) -> tuple[list[str], list[dict[str, str]], list[str]]:

    if df_json == empty_df_json:
        return [], [{'': ''}], []

    df = convert_orders_json_to_df(df_json, False)
    df_options = [{'label': e.upper(), 'value': e} for e in df['symbol'].unique()]
    df_value = df['symbol'].unique()

    # synchronized checklists:
    #   https://dash.plotly.com/advanced-callbacks
    #   selecting 'All' will alternately select all or none of the other options
    input_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if input_id == 'selected-orders':
        singly_selected = df_value.tolist()
    elif input_id == 'stock-symbol':
        all_selected = (
            ['All'] 
            if set(singly_selected) == set(df_value.tolist()) 
            else [])
    else:
        singly_selected = df_value if all_selected else []

    return all_selected, df_options, singly_selected 


@app.callback(
    [Output('all-tags', 'value'), 
     Output('tags', 'options'), 
     Output('tags', 'value')], 
    [Input('selected-orders', 'data'),
     Input('all-tags', 'value'), 
     Input('tags', 'value')])
def tags_categories(
    df_json: str, all_selected: list[str], singly_selected: list[str],
    ) -> tuple[list[str], list[dict[str, str]], list[str]]:

    if df_json == empty_df_json:
        return [], [{'': ''}], []

    df = convert_orders_json_to_df(df_json, False)
    
    if 'tags' in df.columns:
        tags_colname = 'tags'
    else:
        # 'tags_sell' column should probably be included
        tags_colname = 'tags_buy'

    df_values = [
        e
        for e in (df[tags_colname].str.split(', ')).explode().unique() 
        if e != '' and e != None]
    # alphabetize tags
    df_values.sort()
    df_options = [{'label': e, 'value': e} for e in df_values]
    #df_options = [
    #    {'label': e, 'value': e} 
    #    for e in (df[tags_colname].str.split(', ')).explode().unique() 
    #    if e != '' and e != None]

    # synchronized checklists:
    #   https://dash.plotly.com/advanced-callbacks
    #   selecting 'All' will alternately select all or none of the other options
    input_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    df_value = [e['value'] for e in df_options]
    if input_id == 'selected-orders':
        singly_selected = []
    elif input_id == 'tags':
        all_selected = (
            ['All'] 
            if set(singly_selected) == set(df_value) 
            else [])
    else:
        singly_selected = df_value if all_selected else []

    return all_selected, df_options, singly_selected 



##################################################
# DATA PIPELINE FOR (PROVIDED) TABLE
##################################################
# Given table of filled orders, produces and stores 3 derived tables:
#   1) Orders Position Change
#   2) Orders Ongoing Position
#   3) Orders Symbol Day
# Schemas for each table are validated/enforced by Pandera models
##################################################


def itemize_datetime_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract date and time information (e.g., month, hour of day) from timestamps
        and add it as separate columns in table for convenient filtering of the 
        data
    """

    df = convert_columns_to_date_type(df)

    # column 'submit_date' is in the table/dataframe 'Orders Symbol Day', but it
    #   does not include time-of-day information, only the date, and that 
    #   produces erroneous time-of-day information for the '*_hour_of_day' and
    #   '*_time_of_day' columns
    date_colnames_local = [
        e 
        for e in date_colnames 
        if e in df.columns and e != 'submit_date']

    # time of day bins:
    #   570 minutes = 9:30 AM, market opening time in United States Eastern Time
    #   960 minutes = 4:00 PM, market closing time in United States Eastern Time
    midnight01 = 0
    market_open = 570
    market_close = 960
    midnight02 = 1440


    # earliest time among the buy and sell times of the transaction
    ################################################## 

    df['min_date'] = df[date_colnames_local].min(axis=1)

    df['min_date_quarter'] = df['min_date'].dt.quarter
    df['min_date_month'] = df['min_date'].dt.month
    df['min_date_week_of_year'] = df['min_date'].dt.isocalendar().week
    df['min_date_day_of_week'] = df['min_date'].dt.day_of_week
    df['min_date_day_name'] = df['min_date'].dt.day_name()
    df['min_date_hour_of_day'] = df['min_date'].dt.hour
    df['min_date_time_of_day'] = df['min_date'].dt.time

    minute_of_day = (df['min_date'].dt.hour * 60) + df['min_date'].dt.minute
    df['min_date_market_time'] = pd.cut(
        minute_of_day, 
        bins=(midnight01, market_open, market_close, midnight02), 
        right=False, 
        labels=('premarket', 'market', 'postmarket'))


    # latest time among the buy and sell times of the transaction
    ################################################## 

    df['max_date'] = df[date_colnames_local].max(axis=1)

    df['max_date_quarter'] = df['max_date'].dt.quarter
    df['max_date_month'] = df['max_date'].dt.month
    df['max_date_week_of_year'] = df['max_date'].dt.isocalendar().week
    df['max_date_day_of_week'] = df['max_date'].dt.day_of_week
    df['max_date_day_name'] = df['max_date'].dt.day_name()
    df['max_date_hour_of_day'] = df['max_date'].dt.hour
    df['max_date_time_of_day'] = df['max_date'].dt.time

    minute_of_day = (df['max_date'].dt.hour * 60) + df['max_date'].dt.minute
    df['max_date_market_time'] = pd.cut(
        minute_of_day, 
        bins=(midnight01, market_open, market_close, midnight02), 
        right=False, 
        labels=('premarket', 'market', 'postmarket'))

    return df


@pdr.check_output(FilledOrdersForDashboard.to_schema(), lazy=True)
def convert_filled_orders_json_to_df(
    df_json: str) -> pdr_DataFrame[FilledOrdersForDashboard]:
    """
    Enforce defined schema when converting from JSON to Pandas DataFrame
    """

    df = pd.read_json(io.StringIO(df_json), orient='split', date_unit='ms')
    assert isinstance(df, pd.DataFrame)

    return df


@pdr.check_output(FilledOrdersByPositionChange.to_schema(), lazy=True)
def convert_orders_position_change_json_to_df(
    df_json: str) -> pdr_DataFrame[FilledOrdersByPositionChange]:
    """
    Enforce defined schema when converting from JSON to Pandas DataFrame
    """

    df = pd.read_json(io.StringIO(df_json), orient='split', date_unit='ms')
    assert isinstance(df, pd.DataFrame)

    return df


@pdr.check_output(FilledOrdersByOngoingPosition.to_schema(), lazy=True)
def convert_orders_ongoing_position_json_to_df(
    df_json: str) -> pdr_DataFrame[FilledOrdersByOngoingPosition]:
    """
    Enforce defined schema when converting from JSON to Pandas DataFrame
    """

    df = pd.read_json(io.StringIO(df_json), orient='split', date_unit='ms')
    assert isinstance(df, pd.DataFrame)

    return df


@pdr.check_output(FilledOrdersBySymbolDay.to_schema(), lazy=True)
def convert_orders_symbol_day_json_to_df(
    df_json: str) -> pdr_DataFrame[FilledOrdersBySymbolDay]:
    """
    Enforce defined schema when converting from JSON to Pandas DataFrame
    """

    df = pd.read_json(io.StringIO(df_json), orient='split', date_unit='ms')
    assert isinstance(df, pd.DataFrame)

    return df


@pdr.check_output(FilledOrdersByPositionChangeDatetime.to_schema(), lazy=True)
def itemize_datetime_data_orders_position_change(
    df: pd.DataFrame) -> pdr_DataFrame[FilledOrdersByPositionChangeDatetime]:
    """
    Enforce defined schema when adding date and time columns
    """

    output_df = itemize_datetime_data(df.copy())

    return output_df


@pdr.check_output(FilledOrdersByOngoingPositionDatetime.to_schema(), lazy=True)
def itemize_datetime_data_orders_ongoing_position(
    df: pd.DataFrame) -> pdr_DataFrame[FilledOrdersByOngoingPositionDatetime]:
    """
    Enforce defined schema when adding date and time columns
    """

    output_df = itemize_datetime_data(df.copy())

    return output_df


@pdr.check_output(FilledOrdersBySymbolDayDatetime.to_schema(), lazy=True)
def itemize_datetime_data_orders_symbol_day(
    df: pd.DataFrame) -> pdr_DataFrame[FilledOrdersBySymbolDayDatetime]:
    """
    Enforce defined schema when adding date and time columns
    """

    output_df = itemize_datetime_data(df.copy())

    return output_df


def parse_table_file_upload(file_content: str) -> pd.DataFrame:
    """
    Convert upload of table file to Pandas DataFrame

    NOTE:  The field separator in the uploaded file is assumed to be '|'

    Adapted from:
        https://dash.plotly.com/dash-core-components/upload
    """

    try:
        content_type, content_string = file_content.split(',')
        decoded = base64.b64decode(content_string)

        # assume that the user uploaded a CSV file
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='|')
        return df

    except Exception as e:
        #print(e)
        return pd.DataFrame()


@app.callback(
    Output('orders', 'data'), 
    Input('orders-upload', 'contents'), 
    State('orders-upload', 'filename'), 
    State('orders-upload', 'last_modified'))
def upload_orders(file_content: str, filename: str, file_date: str) -> str:
    """
    Obtain user-provided table of filled orders

    If user has not provided a table, use default or example data

    Adapted from:
        https://dash.plotly.com/dash-core-components/upload
        https://dash.plotly.com/sharing-data-between-callbacks
    """

    default_input_filepath = (
        Path(__file__).parent.parent.parent.parent / 'trades_plots_data' / 
        'filled_orders' / 'filled_orders.parquet')

    example_input_filepath = (
        Path(__file__).parent.parent.parent / 'example_data' / 'output_data' / 
        'filled_orders.parquet')

    # if user has uploaded a table, use it as input
    if file_content is not None:
        df = parse_table_file_upload(file_content)

    # if user has not uploaded a table, check hard-coded paths for a table
    elif default_input_filepath.exists():
        df = pq.read_table(default_input_filepath).to_pandas()
    elif example_input_filepath.exists():
        df = pq.read_table(example_input_filepath).to_pandas()

    else:
        df = pd.DataFrame()

    df_json = df.to_json(orient='split', date_unit='ms')
    assert isinstance(df_json, str)

    #date = datetime.datetime.fromtimestamp(file_date)

    #div = html.Div([
    #    html.H6(filename),
    #    html.H6(date),
    #    dash_table.DataTable(
    #        df.to_dict('records'),
    #        [{'name': e, 'id': e} for e in df.columns])])

    #return div
    return df_json 


@app.callback(
    [Output('orders-position-change-datetime', 'data'), 
     Output('orders-ongoing-position-datetime', 'data'), 
     Output('orders-symbol-day-datetime', 'data')],
     Input('orders', 'data')) 
def derive_and_store_orders_dfs(orders_json: str) -> tuple[str, str, str]:
    """
    From table of filled orders, calculate 3 derived tables:

        1) Orders Position Change - each row of the table is demarcated by a
            change in the position, i.e., a change in the number of shares the
            trader/investor is holding

            For example, if the investor starts with zero shares, buys 100 
                shares then buys another 100 shares, then sells all 200 shares,
                the transactions span 2 rows in this table:  the first row when
                the investor initially held 100 shares, and the second row when
                the investor held 200 shares

        2) Orders Ongoing Position - each row of the table is demarcated by when
            an investor's number of held shares is zero

            The investor in the example above starts and ends with zero shares,
                so that single 'ongoing position' appears on 1 row of this 
                table; if the investor then bought 500 shares and then sold 
                those 500 shares, that would be a separate ongoing position and
                would appear on a second row of this table

        3) Orders Symbol Day - each row of the table is demarcated by the
            combination of each equity/stock and each day, i.e., all orders that
            an investor filled on a single stock on a single day are grouped
            together

            For example, if an investor made 4 transactions of stock 'AAA' and
                6 transactions on stock 'BBB' on Day 1, and then made 12
                transactions on stock 'AAA' and 14 transactions on stock 'BBB'
                on Day 2, those transactions would appear on 4 rows of this 
                table:  'AAA' on Day 1, 'AAA' on Day 2, 'BBB' on Day 1, and 
                'BBB' on Day 2
    """

    empty_df_json = '{"columns":[],"index":[],"data":[]}'

    if orders_json == empty_df_json:

        position_change_date_df = pd.DataFrame()
        ongoing_position_date_df = pd.DataFrame()
        symbol_day_date_df = pd.DataFrame()

    else:

        orders_df = convert_filled_orders_json_to_df(orders_json)

        position_change_df = convert_to_filled_orders_by_position_change(
            orders_df)
        ongoing_position_df = convert_to_filled_orders_by_ongoing_position(
            orders_df)
        symbol_day_df = convert_to_filled_orders_by_symbol_day(
            ongoing_position_df)

        position_change_date_df = itemize_datetime_data_orders_position_change(
            position_change_df.copy())
        ongoing_position_date_df = itemize_datetime_data_orders_ongoing_position(
            ongoing_position_df.copy())
        symbol_day_date_df = itemize_datetime_data_orders_symbol_day(
            symbol_day_df.copy())

    position_change_date_json = position_change_date_df.to_json(
        orient='split', date_unit='ms')
    assert isinstance(position_change_date_json, str)

    ongoing_position_date_json = ongoing_position_date_df.to_json(
        orient='split', date_unit='ms')
    assert isinstance(ongoing_position_date_json, str)

    symbol_day_date_json = symbol_day_date_df.to_json(
        orient='split', date_unit='ms')
    assert isinstance(symbol_day_date_json, str)

    return (
        position_change_date_json, 
        ongoing_position_date_json, 
        symbol_day_date_json)



##################################################
# FILTER DATA BASED ON USER SELECTIONS
##################################################
# The user may select from among many ways to filter the data
#   e.g., to see transactions that occurred only in a selected range of dates
#   e.g., to see transactions that lost a selected amount of money
# This section aggregates all the user filter selections for subsequent display
#   and plotting
# User filter selections are made in the sidebar
##################################################


def select_real_simulator_both_trades(
    toggle: list[str], series: pd.Series) -> pd.Series:

    if toggle == ['Simulator']:
        return series == 'simulator'
    elif toggle == ['Real']:
        return series == 'real'
    elif ('Simulator' in toggle) and ('Real' in toggle):
        return series.isin(['real', 'simulator'])
    else:
        return pd.Series([False] * len(series))


def select_stocks_options_both_trades(
    toggle: list[str], series: pd.Series) -> pd.Series:

    if toggle == ['Stocks/Equities']:
        return series == 'stock'
    elif toggle == ['Options']:
        return series == 'option'
    elif ('Stocks/Equities' in toggle) and ('Options' in toggle):
        return pd.Series([True] * len(series))
    else:
        return pd.Series([False] * len(series))


def select_long_short_both_trades(
    toggle: list[str], series: pd.Series) -> pd.Series:

    if toggle == ['Long']:
        return series == 'long'
    elif toggle == ['Short']:
        return series == 'short'
    elif ('Long' in toggle) and ('Short' in toggle):
        return series.isin(['long', 'short'])
    else:
        return pd.Series([False] * len(series))


def select_bull_bear_both_trades(
    toggle: list[str], series: pd.Series) -> pd.Series:

    if toggle == ['Bull']:
        return series == 'bullish'
    elif toggle == ['Bear']:
        return series == 'bearish'
    elif ('Bull' in toggle) and ('Bear' in toggle):
        return series.isin(['bullish', 'bearish'])
    else:
        return pd.Series([False] * len(series))


def filter_dates_to_masks(
    df: pd.DataFrame, buy_sell_date_chosen: bool, min_colname: str, 
    max_colname: str, selection: Union[list[int], list[str]]) -> pd.Series:
    """
    If the user selects 'buy and sell dates', then both buy and sell dates of a
        transaction should be in the date interval; otherwise, only the sell 
        date is required to be in the date interval
    """

    # if select 'buy and sell dates' 
    if buy_sell_date_chosen:
        mask = (
            df[min_colname].isin(selection) & 
            df[max_colname].isin(selection))
    # elif select 'sell dates'
    else:
        mask = df[max_colname].isin(selection)

    return mask


def combine_filter_masks(masks: list[pd.Series]) -> pd.Series:
    """
    A transaction is included in the final mask only if it is in every 
        individual mask, i.e., individual masks are combined by conjunctions
    """

    mask = pd.concat(masks, axis=1).prod(axis=1).astype(bool)

    return mask


def filter_trades_to_mask(
    df: pd.DataFrame, real_simulated: list[str], stocks_options: list[str], 
    long_short: list[str], bull_bear: list[str], buy_sell_date: str, 
    start_date: str, end_date: str, quarter_of_year: list[str], 
    month_of_year: list[str], day_of_week: list[str], hour_of_day: list[int], 
    market_time_of_day: list[str], stock_symbol: list[str],
    fill_price_buy: list[float], fill_price_sell: list[float], 
    commission_buy_sell: list[float], balance_change: list[float], 
    balance_change_commission: list[float], shares_num_fill: list[float],
    tags: list[str]) -> pd.Series:

    if df.empty:
        return pd.Series([])


    # TRADE/POSITION/TRANSACTION TYPE FILTERS
    ##################################################

    real_simulated_mask = select_real_simulator_both_trades(
        real_simulated, df['simulator_or_real'])

    stocks_options_mask = select_stocks_options_both_trades(
        stocks_options, df['stock_or_option'])

    long_short_mask = select_long_short_both_trades(
        long_short, df['long_or_short'])

    bull_bear_mask = select_bull_bear_both_trades(
        bull_bear, df['bull_or_bear'])


    # DATE/TIME FILTERS
    ##################################################

    quarter_of_year_int = [int(i) for i in quarter_of_year]
    month_of_year_int = [int(i) for i in month_of_year]
    day_of_week_int = [int(i) for i in day_of_week]

    buy_sell_date_chosen = buy_sell_date == buy_sell_date_choice

    # these variables all have the same filtering logic, so they are factored 
    #   out together to make them unit-testable, at a small cost of running the 
    #   if...else conditional multiple times
    quarter_mask = filter_dates_to_masks(
        df, buy_sell_date_chosen, 'min_date_quarter', 'max_date_quarter', 
        quarter_of_year_int)
    month_mask = filter_dates_to_masks(
        df, buy_sell_date_chosen, 'min_date_month', 'max_date_month', 
        month_of_year_int)
    day_mask = filter_dates_to_masks(
        df, buy_sell_date_chosen, 'min_date_day_of_week', 'max_date_day_of_week', 
        day_of_week_int)
    market_time_mask = filter_dates_to_masks(
        df, buy_sell_date_chosen, 'min_date_market_time', 'max_date_market_time', 
        market_time_of_day)
    hour_mask = (
        (df['min_date_hour_of_day'] >= hour_of_day[0]) & 
        (df['max_date_hour_of_day'] <= hour_of_day[1]))

    # if select 'buy and sell dates' 
    if buy_sell_date_chosen:
        min_date_mask = pd.to_datetime(start_date) < df['min_date']
    # elif select 'sell dates'
    else:
        min_date_mask = pd.to_datetime(start_date) < df['max_date']

    # timestamp is at midnight that begins specified date, so add 1 day to 
    #   include all of the specified date
    max_date_mask = (
        pd.to_datetime(end_date) + pd.Timedelta('1 day') > df['max_date'])


    # STOCK/EQUITY SYMBOL FILTER
    ##################################################

    stock_mask = df['symbol'].isin(stock_symbol)


    # PRICE/COST FILTERS
    ##################################################

    # the columns 'fill_price_buy' and 'fill_price_sell' are only in the Orders
    #   Position Change table/dataframe; if a user can chooses any other table/
    #   dataframe, the prices are averaged in 'buy_mean_fill_price' and 
    #   'sell_mean_fill_price'

    try:
        buy_fill_price_mask = (
            (df['fill_price_buy'] >= 10 ** fill_price_buy[0]) & 
            (df['fill_price_buy'] <= 10 ** fill_price_buy[1]))
    except:
        buy_fill_price_mask = (
            (df['buy_mean_fill_price'] >= 10 ** fill_price_buy[0]) & 
            (df['buy_mean_fill_price'] <= 10 ** fill_price_buy[1]))

    try:
        sell_fill_price_mask = (
            (df['fill_price_sell'] >= 10 ** fill_price_sell[0]) & 
            (df['fill_price_sell'] <= 10 ** fill_price_sell[1]))
    except:
        sell_fill_price_mask = (
            (df['sell_mean_fill_price'] >= 10 ** fill_price_sell[0]) & 
            (df['sell_mean_fill_price'] <= 10 ** fill_price_sell[1]))

    # the column 'commission_cost' is not in the table 'orders_position_change'
    # if calling the column produces an error (because it's not in the 
    #   table), create mask from columns 'commission_cost_buy' and 
    #   'commission_cost_sell', which are in 'orders_position_change'

    try:
        # commission cost can have a large range, so it really should be on a 
        #   log scale, but that makes it inconvenient to include zero, when 
        #   there's no commission; so, " - 0.01" is a hacky way to include zero
        commission_cost_mask = (
            (df['commission_cost'] >= 10 ** commission_buy_sell[0] - 0.011) & 
            (df['commission_cost'] <= 10 ** commission_buy_sell[1]))
    except:
        commission_colnames = ['commission_cost_buy', 'commission_cost_sell']
        commission_cost = df[commission_colnames].sum(axis=1)
        commission_cost_mask = (
            (commission_cost >= 10 ** commission_buy_sell[0] - 0.011) & 
            (commission_cost <= 10 ** commission_buy_sell[1]))
        #commission_cost_mask = df.iloc[:, 0] == df.iloc[:, 0]

    balance_change_mask = (
        (df['balance_change'] >= balance_change[0]) & 
        (df['balance_change'] <= balance_change[1]))

    balance_change_commission_mask = (
        (df['balance_change_commission'] >= balance_change_commission[0]) & 
        (df['balance_change_commission'] <= balance_change_commission[1]))


    # VOLUME FILTERS
    ##################################################

    # volume information in 'orders_position_change' table/dataframe is in 
    #   column 'match_shares_num_fill' 
    if 'match_shares_num_fill' in df.columns:
        shares_num_fill_colname = 'match_shares_num_fill'
    # volume information in 'orders_ongoing_position' and 
    #   'orders_ongoing_position' tables/dataframes is in column 
    #   'shares_num_fill' 
    #elif 'shares_num_fill' in df.columns:
    else:
        shares_num_fill_colname = 'shares_num_fill'

    volume_mask = (
        (df[shares_num_fill_colname] >= 10 ** shares_num_fill[0]) & 
        (df[shares_num_fill_colname] <= 10 ** shares_num_fill[1]))


    # TAG FILTERS
    ##################################################

    # cells in columns 'tags_buy', 'tags_sell', or 'tags' may each contain 
    #   multiple tags separated by a comma and space (', ')
    # to use 'isin' function, these tags need to be separated into their own 
    #   rows, matched, then aggregated again by row index to produce a correct
    #   mask with the correct length
    try:

        # handles table 'orders_position_change'
        if tags:

            tags_buy_mask = (
                df['tags_buy'].str.split(', ').explode()
                .isin(tags)
                .reset_index().groupby('index')
                .sum() > 0)
            tags_buy_mask.columns = ['tags']

            tags_sell_mask = (
                df['tags_sell'].str.split(', ').explode()
                .isin(tags)
                .reset_index().groupby('index')
                .sum() > 0)
            tags_sell_mask.columns = ['tags']

            tag_mask = tags_buy_mask | tags_sell_mask

        # if 'tags' selection list is empty, select all rows
        else:
            tag_mask = df.iloc[:, 0] == df.iloc[:, 0]

    except:
        # handles tables 'orders_ongoing_position' and 'orders_ongoing_position'
        if tags:
            tag_mask = (
                df['tags'].str.split(', ').explode()
                .isin(tags)
                .reset_index().groupby('index')
                .sum() > 0)
        # if 'tags' selection list is empty, select all rows
        else:
            tag_mask = df.iloc[:, 0] == df.iloc[:, 0]


    # COMBINE FILTERS
    ##################################################

    masks = [
        real_simulated_mask, stocks_options_mask, long_short_mask, 
        bull_bear_mask, min_date_mask, max_date_mask, quarter_mask, month_mask, 
        day_mask, hour_mask, market_time_mask, stock_mask, buy_fill_price_mask, 
        sell_fill_price_mask, commission_cost_mask, balance_change_mask, 
        balance_change_commission_mask, volume_mask, tag_mask]
    mask = combine_filter_masks(masks)

    return mask


@app.callback(
    Output('table-mask', 'data'), 
    [Input('selected-orders', 'data'),
     Input('real-simulator-both', 'value'),
     Input('stocks-options-both', 'value'),
     Input('long-short-both', 'value'),
     Input('bull-bear-both', 'value'),
     Input('buy-sell-dates', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('quarter-of-year', 'value'),
     Input('month-of-year', 'value'),
     Input('day-of-week', 'value'),
     Input('hour-of-day', 'value'),
     Input('market-time-of-day', 'value'),
     Input('stock-symbol', 'value'),
     Input('fill-price-buy', 'value'),
     Input('fill-price-sell', 'value'),
     Input('commission-buy-sell', 'value'),
     Input('balance-change', 'value'),
     Input('balance-change-commission', 'value'),
     Input('shares-num-fill', 'value'),
     Input('tags', 'value')])
def table_mask(
    df_json: str, real_simulated: list[str], stocks_options: list[str], 
    long_short: list[str], bull_bear: list[str], buy_sell_date: str, 
    start_date: str, end_date: str, 
    quarter_of_year: list[str], month_of_year: list[str], 
    day_of_week: list[str], hour_of_day: list[int], 
    market_time_of_day: list[str], stock_symbol: list[str], 
    fill_price_buy: list[float], fill_price_sell: list[float], 
    commission_buy_sell: list[float], balance_change: list[float], 
    balance_change_commission: list[float], shares_num_fill: list[float], 
    tags: list[str]) -> str:


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = filter_trades_to_mask(
        df, real_simulated, stocks_options, long_short, bull_bear, 
        buy_sell_date, start_date, end_date, quarter_of_year, month_of_year, 
        day_of_week, hour_of_day, market_time_of_day, stock_symbol, 
        fill_price_buy, fill_price_sell, commission_buy_sell, balance_change, 
        balance_change_commission, shares_num_fill, tags)

    mask_json = mask.to_json()
    assert isinstance(mask_json, str)

    return mask_json



##################################################
# CALCULATE AND DISPLAY TRADE STATISTICS
##################################################


@dataclass
class statistic_description:
    description: str = ''
    statistic: float = np.nan


def calculate_trade_statistics(df: pd.DataFrame) -> list[statistic_description]: 

    # 'trade' and 'position' have the same meaning

    if df.empty:
        return 4 * [statistic_description()]

    trade_gain_mask = df['balance_change'] > 0
    trade_loss_mask = df['balance_change'] < 0
    trade_gain_commission_mask = df['balance_change_commission'] > 0
    trade_loss_commission_mask = df['balance_change_commission'] < 0

    mean_price_change_for_gain = df.loc[
        trade_gain_mask, 'balance_change'].mean()

    mean_price_change_for_loss = df.loc[
        trade_loss_mask, 'balance_change'].mean()

    total_price_change_for_gain = df.loc[
        trade_gain_mask, 'balance_change'].sum() 
    total_price_change_for_loss = df.loc[
        trade_loss_mask, 'balance_change'].sum()

    mean_price_change_for_gain_commission = (
        df.loc[trade_gain_commission_mask, 
            'balance_change_commission'].mean())

    mean_price_change_for_loss_commission = (
        df.loc[trade_loss_commission_mask, 
            'balance_change_commission'].mean())

    total_price_change_for_gain_commission = (
        df.loc[trade_gain_commission_mask, 
            'balance_change_commission'].sum())
    total_price_change_for_loss_commission = (
        df.loc[trade_loss_commission_mask, 
            'balance_change_commission'].sum())

    v01 = statistic_description('Profit-Loss (P/L) Ratio without Commissions')
    if mean_price_change_for_loss != 0:
        v01.statistic = abs(
            mean_price_change_for_gain / 
            mean_price_change_for_loss)

    v02 = statistic_description('Profit Factor without Commissions')
    if total_price_change_for_loss != 0:
        v02.statistic = abs(
            total_price_change_for_gain / 
            total_price_change_for_loss)

    v03 = statistic_description('Profit-Loss (P/L) Ratio with Commissions')
    if mean_price_change_for_loss_commission != 0:
        v03.statistic = abs(
            mean_price_change_for_gain_commission / 
            mean_price_change_for_loss_commission)

    v04 = statistic_description('Profit Factor (with Commissions)')
    if total_price_change_for_loss_commission != 0:
        v04.statistic = abs(
            total_price_change_for_gain_commission / 
            total_price_change_for_loss_commission)


    stat_var_names = [
        'v' + str(i) if len(str(i)) > 1 
        else 'v0' + str(i) 
        for i in range(1, 5)]

    # list comprehension doesn't work; raises KeyError
    #st = [locals()[e] for e in stat_var_names]
    stat_list = []
    for e in stat_var_names:

        try:
            stat_list.append(locals()[e])

            # round statistics that are floats to 2 decimal places
            if isinstance(locals()[e].statistic, float):
                locals()[e].statistic = round(locals()[e].statistic, 2)

        except:
            pass

    return stat_list


@app.callback(
    Output('profit-table', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def profit_table(df_json: str, table_mask: str) -> go.Figure:


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = pd.read_json(io.StringIO(table_mask), typ='series')
    stats = calculate_trade_statistics(df.loc[mask, :])


    # make plot 
    ################################################## 

    # 'without commissions' color
    no_comm_col = 'lightseagreen'
    # 'with commissions' color
    comm_col = 'orange'

    fig = go.Figure(data=[go.Table(
        columnwidth=[30, 30, 30],
        header={
            'values': ['', 'w/o Commissions', 'w/ Commissions'],
            'fill': {'color': ['black']},
            'font': {'color': ['gray', no_comm_col, comm_col]},
            'font_size': 14,
            },
        cells={
            'values': [
                ['Profit-Loss (P/L) Ratio', 'Profit Factor'], 
                [stats[0].statistic, stats[2].statistic], 
                [stats[1].statistic, stats[3].statistic]],
            'fill': {'color': ['lightgray', no_comm_col, comm_col]},
            'align': ['left', 'center', 'center'],
            'font_size': 14,
            'height': 30})])

    return fig



##################################################
# TABLE OF INDIVIDUAL ORDERS/TRANSACTIONS
##################################################


@app.callback(
    Output('orders-table', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def orders_table(df_json: str, table_mask: str) -> go.Figure:


    # set up dataframe 
    ################################################## 

    df = convert_orders_json_to_df(df_json)
    mask = pd.read_json(io.StringIO(table_mask), typ='series')


    # sum commission costs that are divided between buy and sell orders
    if 'commission_cost' not in df.columns:
        df['commission_cost'] = (
            df['commission_cost_buy'] + df['commission_cost_sell'])


    # filter columns and create prettier column names
    colnames_pretty_map = {
        #'positions_idx': 'Index',
        'order_fill_cancel_time_buy': 'Time',
        'order_fill_cancel_time': 'Time',
        'symbol': 'Symbol',
        'num_buy_sell_orders': '# of Orders',
        'shares_num_fill': '# of Shares',
        'match_shares_num_fill': '# of Shares',
        'commission_cost': 'Commission',
        'spent_outflow': 'Spent Outflow',
        'balance_change': 'Balance Change',
        'fill_price_buy': 'Buy Price',
        'buy_mean_fill_price': 'Buy Price',
        'fill_price_sell': 'Sell Price',
        'sell_mean_fill_price': 'Sell Price',
        'fill_price_change': 'Price Change'}

    colnames = []
    colnames_pretty = []
    for e in colnames_pretty_map.keys():
        if e in df.columns:
            colnames.append(e)
            colnames_pretty.append(colnames_pretty_map[e])

    table_df = df.loc[mask, colnames].copy()


    # sort orders/transactions by order fill time
    # 'try... except' is an inelegant way to select between two options, but it 
    #   is faster than checking for the conditional with an 'if... else'
    try:
        table_df = table_df.sort_values('order_fill_cancel_time_buy')
    except:
        table_df = table_df.sort_values('order_fill_cancel_time')


    # add column for price change as a percentage
    colname = 'Price Change %'
    table_df[colname] = (
        (table_df.iloc[:, -1] / table_df.iloc[:, -3]) * 100).round(1)
    colnames.append(colname)
    colnames_pretty.append(colname)


    # format dataframe for prettier presentation
    ################################################## 

    # capitalize stock/equity/option symbol/abbreviation/ticker
    table_df['symbol'] = table_df['symbol'].str.upper()

    # format dates and times
    for e in table_df:  
        if pd.api.types.is_datetime64_any_dtype(table_df[e]):
            table_df[e] = table_df[e].dt.strftime('%Y-%m-%d %H:%M:%S')

    # round decimal numbers
    for e in table_df:  
        if pd.api.types.is_float_dtype(table_df[e]):
            table_df[e] = table_df[e].round(2)


    # make table
    ################################################## 

    colnames_width_map = {
        'Index': 3,
        'Time': 9,
        'Symbol': 5,
        '# of Orders': 4,
        '# of Shares': 4,
        'Commission': 5,
        'Spent Outflow': 4,
        'Balance Change': 4,
        'Buy Price': 3,
        'Sell Price': 3,
        'Price Change': 3,
        'Price Change %': 3}

    column_widths = [colnames_width_map[e] for e in colnames_pretty]

    data = go.Table(
        columnwidth=column_widths,
        header={
            'values': colnames_pretty,
            'fill': {'color': ['black']},
            'font': {'color': ['white']},
            'font_size': 14},
        cells={
            'values': table_df[colnames].transpose().values.tolist(),
            'fill': {'color': ['lightgray']},
            'align': ['center'],
            'font_size': 14,
            'height': 30})

    layout = go.Layout({
        'font_size': 16,
        'title': 'Selected Positions',
        'title': f'Selected Positions <sub>(Total = {len(table_df)})</sub>',
        'title_x': 0.5})

    fig = go.Figure(data=data, layout=layout)

    return fig



##################################################
# PLOTS OF TRADE DATA
##################################################


@app.callback(
    Output('number-of-positions-by-date-plot', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def number_of_positions_by_date_plot(
    df_json: str, table_mask: str) -> go.Figure:

    if df_json == empty_df_json:
        return go.Figure()


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = pd.read_json(io.StringIO(table_mask), typ='series')
    df_mask = df.loc[mask, 'max_date'].dt.date.copy()
    plot_df = df_mask.value_counts().reset_index()


    # make plot 
    ################################################## 

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 16,
        'title': f'Number of Positions <sub>(Total = {len(df_mask)})</sub> per Day ',
        'title_x': 0.5,
        'xaxis_tickangle': 45,
        'yaxis_title': 'Number of Positions',
        })

    fig = go.Figure(layout=layout)
    try:
        # TODO:  works for deprecated version of Pandas; REMOVE
        fig.add_trace(go.Bar(x=plot_df['index'], y=plot_df['max_date']))
    except:
        fig.add_trace(go.Bar(x=plot_df['max_date'], y=plot_df['count']))
    #fig.add_trace(go.Bar(x=plot_df['max_date'], y=plot_df['count']))

    return fig


@app.callback(
    Output('spent-outflow-by-date-plot', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def spent_outflow_by_date_plot(df_json: str, table_mask: str) -> go.Figure:

    if df_json == empty_df_json:
        return go.Figure()


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = pd.read_json(io.StringIO(table_mask), typ='series')

    colnames = ['spent_outflow', 'spent_outflow_commission', 'max_date']
    df1 = df.loc[mask, colnames].copy()
    df1['date'] = df1['max_date'].dt.date
    plot_df = df1.drop('max_date', axis=1).groupby('date').sum() * -1
    plot_df['commissions'] = (
        plot_df['spent_outflow_commission'] - plot_df['spent_outflow'])


    # make plot 
    ################################################## 

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 16,
        'title': f'Spent Outflow per Day '
            f'<br><sub>(Total amount spent on all transactions during a day)'
            f'</sub>',
        'title_x': 0.5,
        'xaxis_tickangle': 45,
        'yaxis_title': 'Spent Outflow'})

    fig = go.Figure(
        layout=layout,
        data=[
            go.Bar(
                name='Spent Outflow', 
                x=plot_df.index, y=plot_df['spent_outflow']),
            go.Bar(
                name='Commissions', 
                x=plot_df.index, y=plot_df['commissions'])])
    fig.update_layout(barmode='stack')

    return fig


@app.callback(
    Output('number-of-positions-by-gain-loss-plot', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def number_of_positions_by_gain_loss_plot(
    df_json: str, table_mask: str) -> go.Figure:


    def bin_price_to_positive_negative_neither(prices: pd.Series) -> pd.Series:
        """
        Categorize prices into those less than, equal to, or greater than zero,
            or "losses", "net_zeroes", and "gains"
        """

        # for dollars, magnitude of 0.0001 is sufficiently close to zero to be
        #   a bin threshold
        bins = pd.cut(
            prices,
            bins=(float('-inf'), -0.0001, 0.0001, float('inf')), 
            right=False, 
            labels=('Loss', 'No Change', 'Gain'))
        assert isinstance(bins, pd.Series)

        return bins


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = pd.read_json(io.StringIO(table_mask), typ='series')

    if sum(mask) < 1:
        return go.Figure(layout={
            'template': 'plotly_white', 
            'title': 'Number of Positions: NO DATA'})

    colnames = ['symbol', 'max_date', 'balance_change', 'balance_change_commission']
    balance_colnames = ['balance_change', 'balance_change_commission']
    df1 = df.loc[mask, colnames].copy()
    df1 = df1.sort_values('max_date').reset_index(drop=True)
    df2 = df1[balance_colnames].apply(bin_price_to_positive_negative_neither, 1)
    plot_df = df2.apply(lambda x: x.value_counts())
    plot_df = plot_df.sort_index() 
    plot_df2 = ((plot_df / plot_df.sum()).round(2) * 100).astype(int)
    plot_df2 = plot_df2.astype(str) + '%'

    # calculate 'Gain' as percentage of all positions (i.e., accuracy)
    max_percent = 100
    df3 = np.where(df2 == 'Gain', max_percent, 0)

    # calculate rolling means
    window_len = max(1, len(df3) // 8)
    rolling_mean_df = pd.DataFrame(df3).rolling(window_len, center=False).mean()
    rolling_mean_df.columns = balance_colnames  


    # make plot 
    ################################################## 

    # when equity/stock symbols are aggregated, individual symbols may repeat;
    #   extract unique symbols for display
    unique_symbols = (
        df1['symbol']
        .apply(lambda x: str(set(x.split(', ')))[1:-1].replace("'", ""))
        .str.upper())

    # 'without commissions' color
    # 'lightseagreen = '20b2aa':
    #   https://developer.mozilla.org/en-US/docs/Web/CSS/named-color
    #no_comm_col = '#20b2aa'
    no_comm_col = 'lightseagreen'

    # 'with commissions' color
    comm_col = 'orange'

    rolling_title01 = (
        f'Rolling Mean of Accuracy (%) w/o Commissions:  Window Length = {window_len}')
    rolling_title02 = (
        f'Rolling Mean of Accuracy (%) w/ Commissions:  Window Length = {window_len}')
    fig = subplots.make_subplots(
        rows=3, cols=1, 
        row_heights=[3, 1, 1], 
        vertical_spacing=0.10,
        subplot_titles=['', rolling_title01, rolling_title02])
    fig.layout['annotations'][0].update(font={'color': no_comm_col})
    fig.layout['annotations'][1].update(font={'color': comm_col})

    stacked_bar_labels = ['Without Commissions', 'With Commissions']
    fig.add_trace(
        row=1, col=1, 
        trace=go.Bar(
            name=plot_df.index[0], 
            x=stacked_bar_labels, y=plot_df.iloc[0, :], 
            text=plot_df2.iloc[0, :], textposition='auto',
            marker_color='red'))
    fig.add_trace(
        row=1, col=1, 
        trace=go.Bar(
            name=plot_df.index[1], 
            x=stacked_bar_labels, y=plot_df.iloc[1, :], 
            text=plot_df2.iloc[1, :], textposition='auto',
            marker_color='gray'))
    fig.add_trace(
        row=1, col=1, 
        trace=go.Bar(
            name=plot_df.index[2], 
            x=stacked_bar_labels, y=plot_df.iloc[2, :], 
            text=plot_df2.iloc[2, :], textposition='auto',
            marker_color='green'))
    fig.update_layout(barmode='stack')

    # display rolling mean of accuracy (percentage of positions that are 
    #   Gains), excluding commissions
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=rolling_mean_df.index, 
            y=rolling_mean_df['balance_change'], 
            mode='lines',
            name='rolling mean',
            showlegend=False,
            marker_color='black',
            fill='tozeroy',
            fillcolor='green'))
    fig.update_xaxes(
        row=2, col=1,
        #range=[0, max_percent],
        showticklabels=False,
        showgrid=False,
        tickvals=df1.index,
        ticktext=df1['max_date'].astype(str) + ' ' + unique_symbols)

    # display rolling mean of accuracy (percentage of positions that are 
    #   Gains), including commissions
    fig.add_trace(
        row=3, col=1, 
        trace=go.Scatter(
            x=rolling_mean_df.index, 
            y=rolling_mean_df['balance_change_commission'], 
            mode='lines',
            name='rolling mean',
            showlegend=False,
            marker_color='black',
            fill='tozeroy',
            fillcolor='green'))
    fig.update_xaxes(
        row=3, col=1,
        #range=[0, max_percent],
        showticklabels=False,
        showgrid=False,
        tickvals=df1.index,
        ticktext=df1['max_date'].astype(str) + ' ' + unique_symbols)

    fig.update_layout(
        template='plotly_white',
        showlegend=True,
        font_size=16,
        title=f'Number of Positions <sub>(Total = {plot_df.iloc[:, 0].sum()})'
            '</sub> by Loss or Gain (or Neither)'
            f'<br><sub>Accuracy = Percentage of Positions that are Gains</sub> ',
        title_x=0.5,
        yaxis_title='Number of Positions')

    return fig



##################################################
# PLOTS OF TRADE DATA THAT ARE GROUPED/FILTERED BY GAINS/LOSSES
##################################################
# In some cases, the user may want to view data either filtered or grouped by
#   whether the transactions gained or lost money (or a certain amount of money)
# This section includes the dataclasses and functions to calculate these 
#   gain/loss groups, as well as the plots that use them
##################################################


@dataclass
class balance_changes:

    all: float = np.nan
    gain: float = np.nan
    loss: float = np.nan
    all_commission: float = np.nan
    gain_commission: float = np.nan
    loss_commission: float = np.nan

    def __post_init__(self):

        # round floats to 2 decimal places for display
        for field in fields(self):
            setattr(self, field.name, round(getattr(self, field.name), 2))


@dataclass
class time_changes:

    all: float = np.nan
    gain: float = np.nan
    loss: float = np.nan
    gain_commission: float = np.nan
    loss_commission: float = np.nan

    def __post_init__(self):

        # round Timedeltas for display
        for field in fields(self):
            setattr( self, field.name, getattr(self, field.name).round('1s') )


def get_row_indices_for_date_start(series: pd.Series) -> np.ndarray:
    """
    In a vector of elements (dates), identify indices where the element is 
        different from the prior element in the series
    """

    difference_bool = (~(series == series.shift()))
    indices = np.where(difference_bool)
    indices = np.append(indices, len(series))
    indices = indices - 0.5

    return indices  


def get_balance_change_masks(
    df: pd.DataFrame, zero_error_threshold=1e-4) -> pd.DataFrame:
    """
    Given a dataframe of stock/equity trades/positions, identify which 
        transactions (dataframe rows) gained or lost money (i.e., increased or
        decreased the balance) before and after accounting for commissions

    Results are returned as a dataframe where each row is a transaction and each
        column is a Boolean series

    Example input for 4 transactions:

            balance_change      balance_change_commission
                      1000                            990 
                      -500                           -550
                         5                             -5
                         0                              0

    Example result for 4 transactions:

            gain    loss    gain_commission     loss_commission
            True    False   True                False
            False   True    False               True
            True    False   False               True
            False   False   False               False

    In the first row of the example dataframe above, the first transaction 
        gained money and including the commission did not alter that fact
    The second transaction lost money and including the commission did not alter 
        that fact
    The third transaction gained money but including the commission turned the 
        transaction into a net loss
    The fourth transaction neither gained nor lost money

    'zero_error_threshold' - the price data are stored as float data types, so 
        a zero might not be stored exactly as a zero, so this 'error threshold'
        provides a range around zero:  values inside the range are treated as 
        equal to zero
    """

    assert 'balance_change' in df.columns
    assert 'balance_change_commission' in df.columns

    gain_mask = df['balance_change'] > zero_error_threshold
    loss_mask = df['balance_change'] < -zero_error_threshold
    gain_commission_mask = df['balance_change_commission'] > zero_error_threshold
    loss_commission_mask = df['balance_change_commission'] < -zero_error_threshold

    colnames = ['gain', 'loss', 'gain_commission', 'loss_commission']
    mask_df = pd.concat(
        [gain_mask, loss_mask, gain_commission_mask, loss_commission_mask], 
        axis=1)
    mask_df.columns = colnames

    return mask_df


@app.callback(
    Output('balance-change-by-position-chronologically', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def balance_change_by_position_chronologically(
    df_json: str, table_mask: str) -> go.Figure:

    if df_json == empty_df_json:
        return go.Figure()


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    #mask = [True] * len(df)
    #mask = [False] * len(df)
    mask = pd.read_json(io.StringIO(table_mask), typ='series')

    colnames = [
        'balance_change', 'balance_change_commission', 'max_date', 'symbol']
    plot_df = df.loc[mask, colnames].copy().sort_values('max_date')
    plot_df['index'] = range(len(plot_df))
    plot_df['date'] = plot_df['max_date'].dt.date

    window_len = max(1, len(plot_df) // 8)
    plot_df['rolling_median_balance_change'] = (
        plot_df['balance_change'].rolling(window_len).median())
    plot_df['rolling_mean_balance_change'] = (
        plot_df['balance_change'].rolling(window_len).mean().round(4))
    plot_df['rolling_mean_positive_balance_change'] = (
        np.where(
            plot_df['rolling_mean_balance_change'] > 0, 
            plot_df['rolling_mean_balance_change'], 0))
    plot_df['rolling_mean_negative_balance_change'] = (
        np.where(
            plot_df['rolling_mean_balance_change'] < 0, 
            plot_df['rolling_mean_balance_change'], 0))

    plot_df['rolling_median_balance_change_commission'] = (
        plot_df['balance_change_commission'].rolling(window_len).median())
    plot_df['rolling_mean_balance_change_commission'] = (
        plot_df['balance_change_commission'].rolling(window_len).mean().round(4))
    plot_df['rolling_mean_positive_balance_change_commission'] = (
        np.where(
            plot_df['rolling_mean_balance_change_commission'] > 0, 
            plot_df['rolling_mean_balance_change_commission'], 0))
    plot_df['rolling_mean_negative_balance_change_commission'] = (
        np.where(
            plot_df['rolling_mean_balance_change_commission'] < 0, 
            plot_df['rolling_mean_balance_change_commission'], 0))


    # calculate masks for gains and losses and calculate statistics
    ################################################## 

    mask_df = get_balance_change_masks(plot_df)

    balance_means = balance_changes(
        plot_df['balance_change'].mean(),
        plot_df.loc[mask_df['gain'], 'balance_change'].mean(),
        plot_df.loc[mask_df['loss'], 'balance_change'].mean(),
        plot_df['balance_change_commission'].mean(),
        plot_df.loc[mask_df['gain_commission'], 'balance_change_commission'].mean(),
        plot_df.loc[mask_df['loss_commission'], 'balance_change_commission'].mean())


    # make plot 
    ################################################## 

    # when equity/stock symbols are aggregated, individual symbols may repeat;
    #   extract unique symbols for display
    unique_symbols = (
        plot_df['symbol']
        .apply(lambda x: str(set(x.split(', ')))[1:-1].replace("'", ""))
        .str.upper())

    # 'without commissions' color
    # 'lightseagreen = '20b2aa':
    #   https://developer.mozilla.org/en-US/docs/Web/CSS/named-color
    #no_comm_col = '#20b2aa'
    no_comm_col = 'lightseagreen'

    # 'with commissions' color
    comm_col = 'orange'

    rolling_title01 = (
        f'Rolling Median, Mean w/o Commissions:  Window Length = {window_len}')
    rolling_title02 = (
        f'Rolling Median, Mean w/ Commissions:  Window Length = {window_len}')
    fig = subplots.make_subplots(
        rows=3, cols=3, 
        shared_yaxes=True, 
        column_widths=[6, 1, 1], 
        row_heights=[8, 1, 1], 
        horizontal_spacing=0,
        vertical_spacing=0.03,
        subplot_titles=[
            '', '', '', 
            rolling_title01, '', '', 
            rolling_title02, '', ''])
    fig.layout['annotations'][0].update(font={'color': no_comm_col})
    fig.layout['annotations'][1].update(font={'color': comm_col})

    # consecutive positions with the same symbol have the same color on the bar 
    #   plot; consecutive positions with different symbols have different colors
    alternating_symbols = (
        ~(plot_df['symbol'] == plot_df['symbol'].shift())).cumsum().mod(2)
    alternating_colors_dict = {0: no_comm_col, 1: 'lightskyblue'}
    alternating_colors = alternating_symbols.replace(alternating_colors_dict)

    y_min = plot_df['balance_change_commission'].min()
    y_max = plot_df['balance_change'].max()
    y_max_with_large_buffer = y_max + y_max * 0.17
    y_max_with_small_buffer = y_max + y_max * 0.08

    # show individual positions
    fig.add_trace(
        row=1, col=1, 
        trace=go.Bar(
            x=plot_df['index'], 
            y=plot_df['balance_change'], 
            name='Without Commissions',
            showlegend=True,
            marker_color=alternating_colors))
    fig.add_trace(
        row=1, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['balance_change_commission'], 
            name='With Commissions',
            showlegend=True,
            mode='markers',
            #size=4,
            opacity=0.8,
            marker_color=comm_col))
    fig.update_xaxes(
        row=1, col=1,
        showticklabels=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['max_date'].astype(str) + ' ' + unique_symbols)
    fig.update_yaxes(
        row=1, col=1,
        range=[y_min, y_max_with_large_buffer])

    # consecutive positions with the same date have the same color on the bar 
    #   plot; consecutive positions with different dates have different colors
    dates_index = get_row_indices_for_date_start(plot_df['date'])
    shade_color='rgba(25,25,25,0.05)'
    for i in range(1, len(dates_index)-1, 2):
        fig.add_shape(
            type='rect',
            xref='x', yref='y',
            x0=dates_index[i], y0=y_min,
            x1=dates_index[i+1], y1=y_max_with_small_buffer,
            fillcolor=shade_color,
            line=dict(color=shade_color, width=0))


    # show distributions for all positions, only gains, and only losses without 
    #   commissions
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='All',
            showlegend=False,
            y=plot_df['balance_change'], 
            meanline={'visible': True},
            #marker_color='gray',
            fillcolor=no_comm_col,
            line_color='gray'))
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Gains',
            showlegend=False,
            y=plot_df['balance_change'].loc[mask_df['gain']],
            meanline={'visible': True},
            #marker_color='green',
            fillcolor=no_comm_col,
            line_color='green'))
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Losses',
            showlegend=False,
            y=plot_df['balance_change'].loc[mask_df['loss']],
            meanline={'visible': True},
            #marker_color='red',
            fillcolor=no_comm_col,
            line_color='red'))

    # show distributions for all positions, only gains, and only losses with 
    #   commissions
    fig.add_trace(
        row=1, col=3, 
        trace=go.Violin(
            name='All',
            showlegend=False,
            y=plot_df['balance_change_commission'], 
            meanline={'visible': True},
            fillcolor=comm_col,
            line_color='gray'))
    fig.add_trace(
        row=1, col=3, 
        trace=go.Violin(
            name='Gains',
            showlegend=False,
            y=plot_df['balance_change_commission'].loc[mask_df['gain_commission']],
            meanline={'visible': True},
            fillcolor=comm_col,
            line_color='green'))
    fig.add_trace(
        row=1, col=3, 
        trace=go.Violin(
            name='Losses',
            showlegend=False,
            y=plot_df['balance_change_commission'].loc[mask_df['loss_commission']],
            meanline={'visible': True},
            fillcolor=comm_col,
            line_color='red'))

    # display rolling median, mean without commissions
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['rolling_mean_positive_balance_change'], 
            mode='lines',
            name='rolling mean',
            marker_color='black',
            fill='tozeroy',
            fillcolor='green'))
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['rolling_mean_negative_balance_change'], 
            mode='lines',
            name='rolling mean',
            marker_color='black',
            fill='tozeroy',
            fillcolor='red'))
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['rolling_median_balance_change'], 
            mode='lines',
            line={'dash': 'dot'},
            name='rolling median',
            marker_color='black'))
    fig.update_xaxes(
        row=2, col=1,
        showticklabels=False,
        showgrid=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['max_date'].astype(str) + ' ' + unique_symbols)

    # display rolling median, mean with commissions
    fig.add_trace(
        row=3, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['rolling_mean_positive_balance_change_commission'], 
            mode='lines',
            name='rolling mean',
            showlegend=False,
            marker_color='black',
            fill='tozeroy',
            fillcolor='green'))
    fig.add_trace(
        row=3, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['rolling_mean_negative_balance_change_commission'], 
            mode='lines',
            name='rolling mean',
            showlegend=False,
            marker_color='black',
            fill='tozeroy',
            fillcolor='red'))
    fig.add_trace(
        row=3, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['rolling_median_balance_change_commission'], 
            mode='lines',
            line={'dash': 'dot'},
            name='rolling median',
            showlegend=False,
            marker_color='black'))
    fig.update_xaxes(
        row=3, col=1,
        showticklabels=False,
        showgrid=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['max_date'].astype(str) + ' ' + unique_symbols)

    fig.update_layout(
        template='plotly_white',
        legend={'orientation': 'h'},
        font_size=16,
        title=f'Balance Change by Position <sub>({len(plot_df)} positions)</sub>'
            f'<br><sub><span style="color:{no_comm_col}">Average Balance Change '
                f'w/o Commissions = {balance_means.all}, </span>'
            f'<span style="color:green">Gains = {balance_means.gain}, </span>'
            f'<span style="color:red">Losses = {balance_means.loss}</span></sub>'
            f'<br><sub><span style="color:{comm_col}">Average Balance Change w/ '
                f'Commissions = {balance_means.all_commission}, </span>'
            f'<span style="color:green">Gains = {balance_means.gain_commission}, '
                f'</span>'
            f'<span style="color:red">Losses = {balance_means.loss_commission}'
                f'</span></sub>',
        title_x=0.5,
        yaxis_title='Balance Change')

    return fig


def insert_start_row(df: pd.DataFrame, start_row: list[Any]) -> pd.DataFrame:
    """
    Inserts provided row ('start_row') at start of dataframe ('df'), i.e., as
        the first row of the dataframe
    """

    df = df.copy()
    df.index = range(1, len(df)+1)
    df.loc[0] = start_row
    df = df.sort_index()

    return df


@app.callback(
    Output('cumulative-balance-change-by-position-chronologically', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def cumulative_balance_change_by_position_chronologically(
    df_json: str, table_mask: str) -> go.Figure:


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = pd.read_json(io.StringIO(table_mask), typ='series')

    if sum(mask) < 1:
        return go.Figure(layout={
            'template': 'plotly_white', 
            'title': 'Number of Positions: NO DATA'})

    colnames = [
        'balance_change', 'balance_change_commission', 'max_date', 'symbol']
    plot_df = df.loc[mask, colnames].copy().sort_values('max_date')

    # add row that starts balance at zero
    start_row = [0, 0, plot_df['max_date'].iloc[0], 'start at zero']
    plot_df = insert_start_row(plot_df, start_row)

    plot_df['balance_change_cumulative'] = plot_df['balance_change'].cumsum()
    plot_df['balance_change_commission_cumulative'] = (
        plot_df['balance_change_commission'].cumsum())
    plot_df['index'] = range(len(plot_df))
    plot_df['date'] = plot_df['max_date'].dt.date

    plot_df['commissions_cumulative'] = (
        plot_df['balance_change_cumulative'] - 
        plot_df['balance_change_commission_cumulative'])

    y_min = plot_df[
        ['balance_change_cumulative', 
         'balance_change_commission_cumulative']].min().min()
    y_max = plot_df[
        ['balance_change_cumulative', 
         'balance_change_commission_cumulative']].max().max()

    buffer = max(abs(y_min), abs(y_max)) * 0.05
    y_min_with_buffer = y_min + min(buffer, -buffer)
    y_max_with_buffer  = y_max + buffer


    # calculate masks for gains and losses and calculate statistics
    ################################################## 

    mask_df = get_balance_change_masks(plot_df)

    balance_sums = balance_changes(
        plot_df['balance_change_cumulative'].iloc[-1],
        plot_df.loc[mask_df['gain'], 'balance_change'].sum(),
        plot_df.loc[mask_df['loss'], 'balance_change'].sum(),
        plot_df['balance_change_commission_cumulative'].iloc[-1],
        plot_df.loc[mask_df['gain_commission'], 'balance_change_commission'].sum(),
        plot_df.loc[mask_df['loss_commission'], 'balance_change_commission'].sum())


    # make plot 
    ################################################## 

    # when equity/stock symbols are aggregated, individual symbols may repeat;
    #   extract unique symbols for display
    unique_symbols = (
        plot_df['symbol']
        .apply(lambda x: str(set(x.split(', ')))[1:-1].replace("'", ""))
        .str.upper())

    # 'without commissions' color
    no_comm_col = 'lightseagreen'
    # 'with commissions' color
    comm_col = 'orange'

    fig = subplots.make_subplots(
        rows=1, cols=2, 
        shared_yaxes=False, 
        column_widths=[6, 1], 
        horizontal_spacing=0.06)

    # show individual positions successively accumulating
    fig.add_trace(
        row=1, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], y=plot_df['balance_change_cumulative'], 
            mode='lines',
            name='Without Commissions',
            marker_color=no_comm_col))
    fig.add_trace(
        row=1, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['balance_change_commission_cumulative'], 
            name='With Commissions',
            mode='lines',
            marker_color=comm_col))

    # to make cumulative commissions appear only upon hover, create it as a 
    #   separate trace that is invisible on the plot
    fig.add_trace(
        row=1, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], y=plot_df['commissions_cumulative'], 
            name='Commissions',
            showlegend=False,
            mode='lines',
            marker_color='black',
            opacity=0))

    fig.update_xaxes(
        row=1, col=1, 
        showticklabels=False,
        showgrid=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['max_date'].astype(str) + ' ' + unique_symbols)

    # set y-axis range manually so that invisible 'commissions_cumulative' trace 
    #   does not affect it
    fig.update_yaxes(
        row=1, col=1, 
        range=[y_min_with_buffer, y_max_with_buffer])

    # show cumulative positions
    fig.add_trace(
        row=1, col=2, 
        trace=go.Bar(
            name='All',
            showlegend=False,
            x=['Gain w/o C', 'Gain w/ C', 'Loss w/o C', 'Loss w/ C'],
            y=[balance_sums.gain, 
               balance_sums.gain_commission, 
               abs(balance_sums.loss), 
               abs(balance_sums.loss_commission)],
            marker_line_color=[no_comm_col, comm_col, no_comm_col, comm_col],
            marker_color=['green', 'green', 'red', 'red'],
            marker_line_width=5))

    # consecutive positions with the same date have the same color on the bar 
    #   plot; consecutive positions with different dates have different colors
    dates_index = get_row_indices_for_date_start(plot_df['date'])
    shade_color='rgba(25,25,25,0.05)'
    for i in range(1, len(dates_index)-1, 2):
        fig.add_shape(
            type='rect',
            xref='x', yref='y',
            x0=dates_index[i], y0=y_min_with_buffer,
            x1=dates_index[i+1], y1=y_max_with_buffer,
            fillcolor=shade_color,
            line=dict(color=shade_color, width=0))


    fig.update_layout(
        template='plotly_white',
        legend={'orientation': 'h'},
        font_size=16,
        title=f'Cumulative Balance Change by Position <sub>({len(plot_df)} '
                f'positions)</sub>'
            f'<br><sub><span style="color:{no_comm_col}">Cumulative Balance '
                f'Change w/o Commissions = {balance_sums.all}, </span>'
            f'<span style="color:green">Gains = {balance_sums.gain}, </span>'
            f'<span style="color:red">Losses = {balance_sums.loss}</span></sub>'
            f'<br><sub><span style="color:{comm_col}">Cumulative Balance Change '
                f'w/ Commissions = {balance_sums.all_commission}, </span>'
            f'<span style="color:green">Gains = {balance_sums.gain_commission}, '
                f'</span>'
            f'<span style="color:red">Losses = {balance_sums.loss_commission}'
                f'</span></sub>',
        title_x=0.5,
        yaxis_title='Cumulative Balance Change',
        hovermode='x unified')

    return fig


@app.callback(
    Output('cumulative-price-change-per-share-by-position-chronologically', 
        'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def cumulative_price_change_per_share_by_position_chronologically(
    df_json: str, table_mask: str) -> go.Figure:


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = pd.read_json(io.StringIO(table_mask), typ='series')

    if sum(mask) < 1:
        return go.Figure(layout={
            'template': 'plotly_white', 
            'title': 'Number of Positions: NO DATA'})

    colnames = [
        'fill_price_change', 'max_date', 'symbol', 'balance_change', 
        'balance_change_commission']
    plot_df = df.loc[mask, colnames].copy().sort_values('max_date')

    # add row that starts balance at zero
    start_row = [0, plot_df['max_date'].iloc[0], 'start at zero', 0, 0]
    plot_df = insert_start_row(plot_df, start_row)

    plot_df['index'] = range(len(plot_df))
    plot_df['date'] = plot_df['max_date'].dt.date
    plot_df['fill_price_change_cumulative'] = (
        plot_df['fill_price_change'].cumsum().round(2))


    # calculate masks for gains and losses and calculate statistics
    ################################################## 

    mask_df = get_balance_change_masks(plot_df)

    balance_sums = balance_changes(
        plot_df['fill_price_change_cumulative'].iloc[-1],
        plot_df.loc[mask_df['gain'], 'fill_price_change'].sum(),
        plot_df.loc[mask_df['loss'], 'fill_price_change'].sum(),
        plot_df['fill_price_change_cumulative'].iloc[-1],
        plot_df.loc[mask_df['gain_commission'], 'fill_price_change'].sum(),
        plot_df.loc[mask_df['loss_commission'], 'fill_price_change'].sum())


    # make plot 
    ################################################## 

    # when equity/stock symbols are aggregated, individual symbols may repeat;
    #   extract unique symbols for display
    unique_symbols = (
        plot_df['symbol']
        .apply(lambda x: str(set(x.split(', ')))[1:-1].replace("'", ""))
        .str.upper())

    # 'without commissions' color
    no_comm_col = 'lightseagreen'
    # 'with commissions' color
    comm_col = 'orange'

    fig = subplots.make_subplots(
        rows=1, cols=2, 
        shared_yaxes=False, 
        column_widths=[6, 1], 
        horizontal_spacing=0.06)

    y_min = plot_df['fill_price_change_cumulative'].min()
    y_max = plot_df['fill_price_change_cumulative'].max()
    buffer = max(abs(y_min), abs(y_max)) * 0.09
    y_min_with_buffer = y_min + min(buffer, -buffer)
    y_max_with_buffer  = y_max + buffer

    # show individual positions successively accumulating
    fig.add_trace(
        row=1, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], y=plot_df['fill_price_change_cumulative'], 
            mode='lines',
            name='Cumulative Price Change per Share',
            showlegend=False,
            marker_color='gray'))

    fig.update_xaxes(
        row=1, col=1, 
        showticklabels=False,
        showgrid=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['max_date'].astype(str) + ' ' + unique_symbols)
    fig.update_yaxes(
        row=1, col=1,
        range=[y_min_with_buffer, y_max_with_buffer])

    # show cumulative positions
    fig.add_trace(
        row=1, col=2, 
        trace=go.Bar(
            name='All',
            showlegend=False,
            x=['Gain w/o C', 'Gain w/ C', 'Loss w/o C', 'Loss w/ C'],
            y=[balance_sums.gain, 
               balance_sums.gain_commission, 
               abs(balance_sums.loss), 
               abs(balance_sums.loss_commission)],
            marker_line_color=[no_comm_col, comm_col, no_comm_col, comm_col],
            marker_color=['green', 'green', 'red', 'red'],
            marker_line_width=5))

    # consecutive positions with the same date have the same color on the bar 
    #   plot; consecutive positions with different dates have different colors
    y_min = plot_df['fill_price_change_cumulative'].min()
    y_max = plot_df['fill_price_change_cumulative'].max()
    dates_index = get_row_indices_for_date_start(plot_df['date'])
    shade_color='rgba(25,25,25,0.05)'
    for i in range(1, len(dates_index)-1, 2):
        fig.add_shape(
            type='rect',
            xref='x', yref='y',
            x0=dates_index[i], y0=y_min_with_buffer,
            x1=dates_index[i+1], y1=y_max_with_buffer,
            fillcolor=shade_color,
            line=dict(color=shade_color, width=0))


    fig.update_layout(
        template='plotly_white',
        legend={'orientation': 'h'},
        font_size=16,
        title=f'Cumulative Price Change per Share by Position '
                f'<sub>({len(plot_df)} positions)</sub>'
            f'<br><sub>Cumulative Per-Share Price Change = {balance_sums.all}'
                f'</sub>'
            f'<br><sub><span style="color:{no_comm_col}">'
                f'w/o Commissions: </span>'
            f'<span style="color:green">Gains = {balance_sums.gain}, </span>'
            f'<span style="color:red">Losses = {balance_sums.loss}</span></sub>'
            f'<br><sub><span style="color:{comm_col}">'
                f'w/ Commissions:, </span>'
            f'<span style="color:green">Gains = {balance_sums.gain_commission}, '
                f'</span>'
            f'<span style="color:red">Losses = {balance_sums.loss_commission}'
                f'</span></sub>',
        title_x=0.5,
        yaxis_title='Cumulative Price Change per Share',
        hovermode='x unified')

    return fig


@app.callback(
    Output('price-change-per-share-by-position-chronologically', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def price_change_per_share_by_position_chronologically(
    df_json: str, table_mask: str) -> go.Figure:

    if df_json == empty_df_json:
        return go.Figure()


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = pd.read_json(io.StringIO(table_mask), typ='series')

    colnames = [
        'fill_price_change', 'max_date', 'symbol', 'balance_change', 
        'balance_change_commission']
    plot_df = df.loc[mask, colnames].copy().sort_values('max_date')

    plot_df['index'] = range(len(plot_df))
    plot_df['date'] = plot_df['max_date'].dt.date

    window_len = max(1, len(plot_df) // 8)
    plot_df['rolling_median'] = (
        plot_df['fill_price_change'].rolling(window_len).median())
    plot_df['rolling_mean'] = (
        plot_df['fill_price_change'].rolling(window_len).mean().round(4))
    plot_df['rolling_mean_positive'] = (
        np.where(plot_df['rolling_mean'] > 0, plot_df['rolling_mean'], 0))
    plot_df['rolling_mean_negative'] = (
        np.where(plot_df['rolling_mean'] < 0, plot_df['rolling_mean'], 0))


    # calculate masks for gains and losses and calculate statistics
    ################################################## 

    mask_df = get_balance_change_masks(plot_df)

    balance_means = balance_changes(
        plot_df['fill_price_change'].mean(),
        plot_df.loc[mask_df['gain'], 'fill_price_change'].mean(),
        plot_df.loc[mask_df['loss'], 'fill_price_change'].mean(),
        plot_df['fill_price_change'].mean(),
        plot_df.loc[mask_df['gain_commission'], 'fill_price_change'].mean(),
        plot_df.loc[mask_df['loss_commission'], 'fill_price_change'].mean())


    # make plot 
    ################################################## 

    # when equity/stock symbols are aggregated, individual symbols may repeat;
    #   extract unique symbols for display
    unique_symbols = (
        plot_df['symbol']
        .apply(lambda x: str(set(x.split(', ')))[1:-1].replace("'", ""))
        .str.upper())

    # 'without commissions' color
    no_comm_col = 'lightseagreen'
    # 'with commissions' color
    comm_col = 'orange'

    rolling_title = f'Rolling Median, Mean:  Window Length = {window_len}'
    fig = subplots.make_subplots(
        rows=2, cols=2, 
        shared_yaxes=True, 
        column_widths=[5, 1], 
        row_heights=[8, 1], 
        horizontal_spacing=0,
        vertical_spacing=0.03,
        subplot_titles=['', '', rolling_title, ''])

    # consecutive positions with the same symbol have the same color on the bar 
    #   plot; consecutive positions with different symbols have different colors
    alternating_symbols = (
        ~(plot_df['symbol'] == plot_df['symbol'].shift())).cumsum().mod(2)
    alternating_colors_dict = {0: 'black', 1: 'gray'}
    alternating_colors = alternating_symbols.replace(alternating_colors_dict)

    y_min = plot_df['fill_price_change'].min()
    y_min_with_buffer = y_min + y_min * 0.1
    y_max = plot_df['fill_price_change'].max()
    y_max_with_large_buffer = y_max + y_max * 0.19
    y_max_with_small_buffer = y_max + y_max * 0.08

    # show individual positions
    fig.add_trace(
        row=1, col=1, 
        trace=go.Bar(
            name='All',
            showlegend=True,
            x=plot_df['index'], 
            y=plot_df['fill_price_change'],
            marker_color=alternating_colors))
    fig.update_xaxes(
        row=1, col=1,
        showticklabels=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['max_date'].astype(str) + ' ' + unique_symbols)
    fig.update_yaxes(
        row=1, col=1,
        range=[y_min_with_buffer, y_max_with_large_buffer])

    # consecutive positions with the same date have the same color on the bar 
    #   plot; consecutive positions with different dates have different colors
    dates_index = get_row_indices_for_date_start(plot_df['date'])
    shade_color='rgba(25,25,25,0.05)'
    for i in range(1, len(dates_index)-1, 2):
        fig.add_shape(
            type='rect',
            xref='x', yref='y',
            x0=dates_index[i], y0=y_min_with_buffer,
            x1=dates_index[i+1], y1=y_max_with_small_buffer,
            fillcolor=shade_color,
            line=dict(color=shade_color, width=0))


    # show distributions for all positions
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='All',
            showlegend=False,
            y=plot_df['fill_price_change'], 
            meanline={'visible': True},
            fillcolor=no_comm_col,
            line_color='gray'))

    # show distributions for all positions, only gains, and only losses without 
    #   commissions
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Gains, w/o C',
            showlegend=True,
            y=plot_df['fill_price_change'].loc[mask_df['gain']],
            meanline={'visible': True},
            fillcolor=no_comm_col,
            line_color='green'))
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Losses, w/o C',
            showlegend=True,
            y=plot_df['fill_price_change'].loc[mask_df['loss']],
            meanline={'visible': True},
            fillcolor=no_comm_col,
            line_color='red'))

    # show distributions for all positions, only gains, and only losses with 
    #   commissions
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Gains, w/ C',
            showlegend=True,
            y=plot_df['fill_price_change'].loc[mask_df['gain_commission']],
            meanline={'visible': True},
            fillcolor=comm_col,
            line_color='green'))
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Losses, w/ C',
            showlegend=True,
            y=plot_df['fill_price_change'].loc[mask_df['loss_commission']],
            meanline={'visible': True},
            fillcolor=comm_col,
            line_color='red'))
    fig.update_xaxes(
        row=1, col=2,
        showticklabels=False)

    # display rolling median, mean
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], y=plot_df['rolling_mean_positive'], 
            mode='lines',
            name='rolling mean',
            marker_color='black',
            fill='tozeroy',
            fillcolor='green'))
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], y=plot_df['rolling_mean_negative'], 
            mode='lines',
            name='rolling mean',
            marker_color='black',
            fill='tozeroy',
            fillcolor='red'))
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], y=plot_df['rolling_median'], 
            mode='lines',
            line={'dash': 'dot'},
            name='rolling median',
            marker_color='black'))
    fig.update_xaxes(
        row=2, col=1,
        showticklabels=False,
        showgrid=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['max_date'].astype(str) + ' ' + unique_symbols)

    fig.update_layout(
        template='plotly_white',
        legend={'orientation': 'h'},
        font_size=16,
        title=f'Price Change per Share by Position <sub>({len(plot_df)} positions)</sub>'
            f'<br><sub>Average Per-Share Price Change = {balance_means.all}'
            f'<br><span style="color:{no_comm_col}">w/o Commissions:</span> '
            f'<span style="color:green">Gains = {balance_means.gain}</span>, '
            f'<span style="color:red">Losses = {balance_means.loss} </span>'
            f'<br><span style="color:{comm_col}">w/ Commissions:</span> '
            f'<span style="color:green">Gains = {balance_means.gain_commission}</span>, '
            f'<span style="color:red">Losses = {balance_means.loss_commission}</span></sub>',
        title_x=0.5,
        yaxis_title='Price Change per Share')

    return fig


def calculate_geometric_mean(series: pd.Series) -> float:

    if len(series) == 0:
        return 0

    series_excluding_zeroes = series.loc[series != 0]
    geometric_mean = series_excluding_zeroes.prod()**(1/len(series_excluding_zeroes))

    return geometric_mean 


def calculate_geometric_mean_percent(series: pd.Series) -> float:

    geometric_mean = calculate_geometric_mean(series)

    # rescale change factor to percentage
    rescaled_geometric_mean = 100 * (geometric_mean - 1)

    return rescaled_geometric_mean 


@app.callback(
    Output('price-percentage-change-by-position-chronologically', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def price_percentage_change_by_position_chronologically(
    df_json: str, table_mask: str) -> go.Figure:

    if df_json == empty_df_json:
        return go.Figure()


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = pd.read_json(io.StringIO(table_mask), typ='series')

    # column names for Orders Position Change table
    colnames01 = [
        'fill_price_buy', 'fill_price_sell', 'fill_price_change', 'max_date', 
        'symbol', 'balance_change', 'balance_change_commission']
    # column names for tables other than Orders Position Change 
    colnames02 = [
        'buy_mean_fill_price', 'sell_mean_fill_price', 'fill_price_change', 
        'max_date', 'symbol', 'balance_change', 'balance_change_commission']
    try:
        plot_df = df.loc[mask, colnames01].copy().sort_values('max_date')
    except:
        plot_df = df.loc[mask, colnames02].copy().sort_values('max_date')

    plot_df['index'] = range(len(plot_df))
    plot_df['date'] = plot_df['max_date'].dt.date
    plot_df['price_percentage_change'] = 100 * (
        plot_df['fill_price_change'] / plot_df.iloc[:, 0]).round(4) 
    # the change factor is the ending/sell price divided by the starting/buy price
    plot_df['price_change_factor'] = (plot_df.iloc[:, 1] / plot_df.iloc[:, 0])


    window_len = max(1, len(plot_df) // 8)
    plot_df['rolling_mean'] = 100 * (
        plot_df['price_change_factor'].rolling(window_len).apply(calculate_geometric_mean).round(4) - 1)
    plot_df['rolling_mean_positive'] = (
        np.where(plot_df['rolling_mean'] > 0, plot_df['rolling_mean'], 0))
    plot_df['rolling_mean_negative'] = (
        np.where(plot_df['rolling_mean'] < 0, plot_df['rolling_mean'], 0))


    # calculate masks for gains and losses and calculate statistics
    ################################################## 

    mask_df = get_balance_change_masks(plot_df)

    mean_all = calculate_geometric_mean_percent(plot_df['price_change_factor'])
    mean_gain = calculate_geometric_mean_percent(
        plot_df.loc[mask_df['gain'], 'price_change_factor'])
    mean_loss = calculate_geometric_mean_percent(
        plot_df.loc[mask_df['loss'], 'price_change_factor'])
    mean_gain_commission = calculate_geometric_mean_percent(
        plot_df.loc[mask_df['gain_commission'], 'price_change_factor'])
    mean_loss_commission = calculate_geometric_mean_percent(
        plot_df.loc[mask_df['loss_commission'], 'price_change_factor'])

    balance_means = balance_changes(
        mean_all, mean_gain, mean_loss,
        mean_all, mean_gain_commission, mean_loss_commission)


    # make plot 
    ################################################## 

    # when equity/stock symbols are aggregated, individual symbols may repeat;
    #   extract unique symbols for display
    unique_symbols = (
        plot_df['symbol']
        .apply(lambda x: str(set(x.split(', ')))[1:-1].replace("'", ""))
        .str.upper())

    # 'without commissions' color
    no_comm_col = 'lightseagreen'
    # 'with commissions' color
    comm_col = 'orange'

    rolling_title = f'Rolling Geometric Mean:  Window Length = {window_len}'
    fig = subplots.make_subplots(
        rows=2, cols=2, 
        shared_yaxes=True, 
        column_widths=[5, 1], 
        row_heights=[8, 1], 
        horizontal_spacing=0,
        vertical_spacing=0.03,
        subplot_titles=['', '', rolling_title, ''])

    # consecutive positions with the same symbol have the same color on the bar 
    #   plot; consecutive positions with different symbols have different colors
    alternating_symbols = (
        ~(plot_df['symbol'] == plot_df['symbol'].shift())).cumsum().mod(2)
    alternating_colors_dict = {0: 'black', 1: 'gray'}
    alternating_colors = alternating_symbols.replace(alternating_colors_dict)

    y_min = plot_df['price_percentage_change'].min()
    y_min_with_buffer = y_min + y_min * 0.1
    y_max = plot_df['price_percentage_change'].max()
    y_max_with_large_buffer = y_max + y_max * 0.19
    y_max_with_small_buffer = y_max + y_max * 0.08

    # show individual positions
    fig.add_trace(
        row=1, col=1, 
        trace=go.Bar(
            name='All',
            showlegend=True,
            x=plot_df['index'], 
            y=plot_df['price_percentage_change'],
            marker_color=alternating_colors))
    fig.update_xaxes(
        row=1, col=1,
        showticklabels=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['max_date'].astype(str) + ' ' + unique_symbols)
    fig.update_yaxes(
        row=1, col=1,
        range=[y_min_with_buffer, y_max_with_large_buffer])

    # consecutive positions with the same date have the same color on the bar 
    #   plot; consecutive positions with different dates have different colors
    dates_index = get_row_indices_for_date_start(plot_df['date'])
    shade_color='rgba(25,25,25,0.05)'
    for i in range(1, len(dates_index)-1, 2):
        fig.add_shape(
            type='rect',
            xref='x', yref='y',
            x0=dates_index[i], y0=y_min_with_buffer,
            x1=dates_index[i+1], y1=y_max_with_small_buffer,
            fillcolor=shade_color,
            line=dict(color=shade_color, width=0))


    # show distributions for all positions
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='All',
            showlegend=False,
            y=plot_df['price_percentage_change'], 
            meanline={'visible': True},
            fillcolor=no_comm_col,
            line_color='gray'))

    # show distributions for all positions, only gains, and only losses without 
    #   commissions
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Gains, w/o C',
            showlegend=True,
            y=plot_df['price_percentage_change'].loc[mask_df['gain']],
            meanline={'visible': True},
            fillcolor=no_comm_col,
            line_color='green'))
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Losses, w/o C',
            showlegend=True,
            y=plot_df['price_percentage_change'].loc[mask_df['loss']],
            meanline={'visible': True},
            fillcolor=no_comm_col,
            line_color='red'))

    # show distributions for all positions, only gains, and only losses with 
    #   commissions
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Gains, w/ C',
            showlegend=True,
            y=plot_df['price_percentage_change'].loc[mask_df['gain_commission']],
            meanline={'visible': True},
            fillcolor=comm_col,
            line_color='green'))
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Losses, w/ C',
            showlegend=True,
            y=plot_df['price_percentage_change'].loc[mask_df['loss_commission']],
            meanline={'visible': True},
            fillcolor=comm_col,
            line_color='red'))
    fig.update_xaxes(
        row=1, col=2,
        showticklabels=False)

    # display rolling median, mean
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], y=plot_df['rolling_mean_positive'], 
            mode='lines',
            name='rolling mean',
            marker_color='black',
            fill='tozeroy',
            fillcolor='green'))
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], y=plot_df['rolling_mean_negative'], 
            mode='lines',
            name='rolling mean',
            marker_color='black',
            fill='tozeroy',
            fillcolor='red'))
    fig.update_xaxes(
        row=2, col=1,
        showticklabels=False,
        showgrid=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['max_date'].astype(str) + ' ' + unique_symbols)

    fig.update_layout(
        template='plotly_white',
        legend={'orientation': 'h'},
        font_size=16,
        title=f'Price Percentage Change by Position <sub>({len(plot_df)} positions)</sub>'
            f'<br><sub>Average (Geometric Mean) Price Percentage Change = {balance_means.all}%'
            f'<br><span style="color:{no_comm_col}">w/o Commissions:</span> '
            f'<span style="color:green">Gains = {balance_means.gain}%</span>, '
            f'<span style="color:red">Losses = {balance_means.loss}%</span>'
            f'<br><span style="color:{comm_col}">w/ Commissions:</span> '
            f'<span style="color:green">Gains = {balance_means.gain_commission}%</span>, '
            f'<span style="color:red">Losses = {balance_means.loss_commission}%</span></sub>',
        title_x=0.5,
        yaxis_title='Price Percentage Change per Share (%)')

    return fig


@app.callback(
    Output('position-hold-times', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def position_hold_times(df_json: str, table_mask: str) -> go.Figure:


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = pd.read_json(io.StringIO(table_mask), typ='series')

    if 'order_submit_time_buy' in df.columns:
        order_submit_time_colname = 'order_submit_time_buy'
        order_fill_cancel_time_colname = 'order_fill_cancel_time_sell'
    elif 'order_submit_time' in df.columns:
        order_submit_time_colname = 'order_submit_time'
        order_fill_cancel_time_colname = 'order_fill_cancel_time'
    else:
        return go.Figure(layout={
            'template': 'plotly_white', 
            'title': 'Position Hold Times: NO DATA'})

    colnames = [
        order_submit_time_colname, order_fill_cancel_time_colname, 
        'balance_change', 'balance_change_commission']
    plot_df = df.loc[mask, colnames].copy()

    plot_df['hold_time'] = (
        plot_df[order_fill_cancel_time_colname] - 
        plot_df[order_submit_time_colname])


    # calculate masks for gains and losses and calculate statistics
    ################################################## 

    mask_df = get_balance_change_masks(plot_df)

    time_means = time_changes(
        plot_df['hold_time'].mean(),
        plot_df.loc[mask_df['gain'], 'hold_time'].mean(),
        plot_df.loc[mask_df['loss'], 'hold_time'].mean(),
        plot_df.loc[mask_df['gain_commission'], 'hold_time'].mean(),
        plot_df.loc[mask_df['loss_commission'], 'hold_time'].mean())


    # make plot 
    ################################################## 

    # 'without commissions' color
    no_comm_col = 'lightseagreen'
    # 'with commissions' color
    comm_col = 'orange'

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 16,
        'title': f'Position Hold Times <sub>({len(plot_df)} positions)</sub>'
            f'<br><sub><span style="color:"gray">Average Hold Times shown below '
                f'x-axis labels ',
        'title_x': 0.5,
        'xaxis_tickangle': 45,
        'yaxis_title': 'Time (T = Seconds)',
        })

    fig = go.Figure(layout=layout)

    fig.add_trace(go.Violin(
        name=f'All<br>{time_means.all}',
        y=plot_df['hold_time'],
        meanline={'visible': True},
        fillcolor='gray',
        line_color='black'))

    fig.add_trace(go.Violin(
        name=f'Gains, w/o Commissions<br>{time_means.gain}',
        y=plot_df.loc[mask_df['gain'], 'hold_time'],
        meanline={'visible': True},
        fillcolor=no_comm_col,
        line_color='green'))
    fig.add_trace(go.Violin(
        name=f'Losses, w/o Commissions<br>{time_means.loss}',
        y=plot_df.loc[mask_df['loss'], 'hold_time'],
        meanline={'visible': True},
        fillcolor=no_comm_col,
        line_color='red'))

    fig.add_trace(go.Violin(
        name=f'Gains, w/ Commissions<br>{time_means.gain_commission}',
        y=plot_df.loc[mask_df['gain_commission'], 'hold_time'],
        meanline={'visible': True},
        fillcolor=comm_col,
        line_color='green'))
    fig.add_trace(go.Violin(
        name=f'Losses, w/ Commissions<br>{time_means.loss_commission}',
        y=plot_df.loc[mask_df['loss_commission'], 'hold_time'],
        meanline={'visible': True},
        fillcolor=comm_col,
        line_color='red'))

    return fig


@app.callback(
    Output('position-volumes', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def position_volumes(df_json: str, table_mask: str) -> go.Figure:


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = pd.read_json(io.StringIO(table_mask), typ='series')

    if 'match_shares_num_fill' in df.columns:
        shares_num_fill_colname = 'match_shares_num_fill'
    elif 'shares_num_fill' in df.columns:
        shares_num_fill_colname = 'shares_num_fill'
    else:
        return go.Figure(layout={
            'template': 'plotly_white', 
            'title': 'Position Volumes: NO DATA'})

    colnames = [
        shares_num_fill_colname, 'balance_change', 'balance_change_commission']
    plot_df = df.loc[mask, colnames]


    # calculate masks for gains and losses and calculate statistics
    ################################################## 

    mask_df = get_balance_change_masks(plot_df)

    balance_means = balance_changes(
        plot_df[shares_num_fill_colname].mean(),
        plot_df.loc[mask_df['gain'], shares_num_fill_colname].mean(),
        plot_df.loc[mask_df['loss'], shares_num_fill_colname].mean(),
        plot_df.loc[mask_df['gain_commission'], shares_num_fill_colname].mean(),
        plot_df.loc[mask_df['loss_commission'], shares_num_fill_colname].mean())


    # make plot 
    ################################################## 

    # 'without commissions' color
    no_comm_col = 'lightseagreen'
    # 'with commissions' color
    comm_col = 'orange'

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 16,
        'title': f'Position Volumes <sub>({len(plot_df)} positions)</sub>'
            f'<br><sub><span style="color:"gray">Average Position Filled Volumes '
                f'shown below x-axis labels ',
        'title_x': 0.5,
        'xaxis_tickangle': 45,
        'yaxis_title': 'Volume (Number of Shares)',
        })

    fig = go.Figure(layout=layout)

    fig.add_trace(go.Violin(
        name=f'All<br>{balance_means.all}',
        y=plot_df[shares_num_fill_colname],
        meanline={'visible': True},
        fillcolor='gray',
        line_color='black'))

    fig.add_trace(go.Violin(
        name=f'Gains, w/o Commissions<br>{balance_means.gain}',
        y=plot_df.loc[mask_df['gain'], shares_num_fill_colname],
        meanline={'visible': True},
        fillcolor=no_comm_col,
        line_color='green'))
    fig.add_trace(go.Violin(
        name=f'Losses, w/o Commissions<br>{balance_means.loss}',
        y=plot_df.loc[mask_df['loss'], shares_num_fill_colname],
        meanline={'visible': True},
        fillcolor=no_comm_col,
        line_color='red'))

    fig.add_trace(go.Violin(
        name=f'Gains, w/ Commissions<br>{balance_means.gain_commission}',
        y=plot_df.loc[mask_df['gain_commission'], shares_num_fill_colname],
        meanline={'visible': True},
        fillcolor=comm_col,
        line_color='green'))
    fig.add_trace(go.Violin(
        name=f'Losses, w/ Commissions<br>{balance_means.loss_commission}',
        y=plot_df.loc[mask_df['loss_commission'], shares_num_fill_colname],
        meanline={'visible': True},
        fillcolor=comm_col,
        line_color='red'))

    return fig


@app.callback(
    Output('position-commissions', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def position_commissions(df_json: str, table_mask: str) -> go.Figure:

    if df_json == empty_df_json:
        return go.Figure()


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = pd.read_json(io.StringIO(table_mask), typ='series')

    colnames = ['balance_change', 'balance_change_commission']
    plot_df = df.loc[mask, colnames].copy()

    plot_df['commission'] = (
        plot_df['balance_change'] - 
        plot_df['balance_change_commission']).round(2)


    # calculate masks for gains and losses and calculate statistics
    ################################################## 

    mask_df = get_balance_change_masks(plot_df)

    balance_means = balance_changes(
        plot_df['commission'].mean(),
        plot_df.loc[mask_df['gain'], 'commission'].mean(),
        plot_df.loc[mask_df['loss'], 'commission'].mean(),
        plot_df['commission'].mean(),
        plot_df.loc[mask_df['gain_commission'], 'commission'].mean(),
        plot_df.loc[mask_df['loss_commission'], 'commission'].mean())


    # make plot 
    ################################################## 

    # 'without commissions' color
    no_comm_col = 'lightseagreen'
    # 'with commissions' color
    comm_col = 'orange'

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 16,
        'title': f'Commissions <sub>({len(plot_df)} positions)</sub>'
            f'<br><sub><span style="color:"gray">Average Commission per Position '
                f'shown below x-axis labels ',
        'title_x': 0.5,
        'xaxis_tickangle': 45,
        'yaxis_title': 'Commissions',
        })

    fig = go.Figure(layout=layout)

    fig.add_trace(go.Violin(
        name=f'All<br>{balance_means.all}',
        y=plot_df['commission'],
        meanline={'visible': True},
        fillcolor='gray',
        line_color='black'))

    fig.add_trace(go.Violin(
        name=f'Gains, w/o Commissions<br>{balance_means.gain}',
        y=plot_df.loc[mask_df['gain'], 'commission'],
        meanline={'visible': True},
        fillcolor=no_comm_col,
        line_color='green'))
    fig.add_trace(go.Violin(
        name=f'Losses, w/o Commissions<br>{balance_means.loss}',
        y=plot_df.loc[mask_df['loss'], 'commission'],
        meanline={'visible': True},
        fillcolor=no_comm_col,
        line_color='red'))

    fig.add_trace(go.Violin(
        name=f'Gains, w/ Commissions<br>{balance_means.gain_commission}',
        y=plot_df.loc[mask_df['gain_commission'], 'commission'],
        meanline={'visible': True},
        fillcolor=comm_col,
        line_color='green'))
    fig.add_trace(go.Violin(
        name=f'Losses, w/ Commissions<br>{balance_means.loss_commission}',
        y=plot_df.loc[mask_df['loss_commission'], 'commission'],
        meanline={'visible': True},
        fillcolor=comm_col,
        line_color='red'))

    return fig


@app.callback(
    Output('balance-change-by-day', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def balance_change_by_day(
    df_json: str, table_mask: str) -> go.Figure:

    if df_json == empty_df_json:
        return go.Figure()


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = pd.read_json(io.StringIO(table_mask), typ='series')

    colnames = [
        'balance_change', 'balance_change_commission', 'max_date', 'symbol']
    plot_df = df.loc[mask, colnames].copy().sort_values('max_date')
    plot_df['date'] = plot_df['max_date'].dt.date
    agg_dict = {
        'symbol': list_of_strings_without_nans, 
        'balance_change': 'sum', 
        'balance_change_commission': 'sum'}
    plot_df = plot_df.groupby('date').agg(agg_dict).reset_index()
    plot_df['index'] = range(len(plot_df))


    window_len = max(1, len(plot_df) // 8)
    plot_df['rolling_median_balance_change'] = (
        plot_df['balance_change'].rolling(window_len).median())
    plot_df['rolling_mean_balance_change'] = (
        plot_df['balance_change'].rolling(window_len).mean().round(4))
    plot_df['rolling_mean_positive_balance_change'] = (
        np.where(
            plot_df['rolling_mean_balance_change'] > 0, 
            plot_df['rolling_mean_balance_change'], 0))
    plot_df['rolling_mean_negative_balance_change'] = (
        np.where(
            plot_df['rolling_mean_balance_change'] < 0, 
            plot_df['rolling_mean_balance_change'], 0))

    plot_df['rolling_median_balance_change_commission'] = (
        plot_df['balance_change_commission'].rolling(window_len).median())
    plot_df['rolling_mean_balance_change_commission'] = (
        plot_df['balance_change_commission'].rolling(window_len).mean().round(4))
    plot_df['rolling_mean_positive_balance_change_commission'] = (
        np.where(
            plot_df['rolling_mean_balance_change_commission'] > 0, 
            plot_df['rolling_mean_balance_change_commission'], 0))
    plot_df['rolling_mean_negative_balance_change_commission'] = (
        np.where(
            plot_df['rolling_mean_balance_change_commission'] < 0, 
            plot_df['rolling_mean_balance_change_commission'], 0))


    # calculate masks for gains and losses and calculate statistics
    ################################################## 

    mask_df = get_balance_change_masks(plot_df)

    balance_means = balance_changes(
        plot_df['balance_change'].mean(),
        plot_df.loc[mask_df['gain'], 'balance_change'].mean(),
        plot_df.loc[mask_df['loss'], 'balance_change'].mean(),
        plot_df['balance_change_commission'].mean(),
        plot_df.loc[mask_df['gain_commission'], 'balance_change_commission'].mean(),
        plot_df.loc[mask_df['loss_commission'], 'balance_change_commission'].mean())


    # make plot 
    ################################################## 

    # when equity/stock symbols are aggregated, individual symbols may repeat;
    #   extract unique symbols for display
    unique_symbols = (
        plot_df['symbol']
        .apply(lambda x: str(set(x))[1:-1].replace("'", ""))
        .str.upper())

    # 'without commissions' color
    # 'lightseagreen = '20b2aa':
    #   https://developer.mozilla.org/en-US/docs/Web/CSS/named-color
    #no_comm_col = '#20b2aa'
    no_comm_col = 'lightseagreen'

    # 'with commissions' color
    comm_col = 'orange'

    rolling_title01 = (
        f'Rolling Median, Mean w/o Commissions:  Window Length = {window_len}')
    rolling_title02 = (
        f'Rolling Median, Mean w/ Commissions:  Window Length = {window_len}')
    fig = subplots.make_subplots(
        rows=3, cols=3, 
        shared_yaxes=True, 
        column_widths=[6, 1, 1], 
        row_heights=[8, 1, 1], 
        horizontal_spacing=0,
        vertical_spacing=0.03,
        subplot_titles=[
            '', '', '', 
            rolling_title01, '', '', 
            rolling_title02, '', ''])
    fig.layout['annotations'][0].update(font={'color': no_comm_col})
    fig.layout['annotations'][1].update(font={'color': comm_col})

    # consecutive positions with the same symbol have the same color on the bar 
    #   plot; consecutive positions with different symbols have different colors
    alternating_symbols = (
        ~(plot_df['symbol'] == plot_df['symbol'].shift())).cumsum().mod(2)
    alternating_colors_dict = {0: no_comm_col, 1: 'lightskyblue'}
    alternating_colors = alternating_symbols.replace(alternating_colors_dict)

    y_min = plot_df['balance_change_commission'].min()
    y_max = plot_df['balance_change'].max()
    y_max_with_large_buffer = y_max + y_max * 0.17
    y_max_with_small_buffer = y_max + y_max * 0.08

    # show individual positions
    fig.add_trace(
        row=1, col=1, 
        trace=go.Bar(
            x=plot_df['index'], 
            y=plot_df['balance_change'], 
            name='Without Commissions',
            showlegend=True,
            marker_color=alternating_colors))
    fig.add_trace(
        row=1, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['balance_change_commission'], 
            name='With Commissions',
            showlegend=True,
            mode='markers',
            #size=4,
            opacity=0.8,
            marker_color=comm_col))
    fig.update_xaxes(
        row=1, col=1,
        showticklabels=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['date'].astype(str) + ' ' + unique_symbols)
    fig.update_yaxes(
        row=1, col=1,
        range=[y_min, y_max_with_large_buffer])

    # consecutive positions with the same date have the same color on the bar 
    #   plot; consecutive positions with different dates have different colors
    dates_index = get_row_indices_for_date_start(plot_df['date'])
    shade_color='rgba(25,25,25,0.05)'
    for i in range(1, len(dates_index)-1, 2):
        fig.add_shape(
            type='rect',
            xref='x', yref='y',
            x0=dates_index[i], y0=y_min,
            x1=dates_index[i+1], y1=y_max_with_small_buffer,
            fillcolor=shade_color,
            line=dict(color=shade_color, width=0))


    # show distributions for all positions, only gains, and only losses without 
    #   commissions
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='All',
            showlegend=False,
            y=plot_df['balance_change'], 
            meanline={'visible': True},
            #marker_color='gray',
            fillcolor=no_comm_col,
            line_color='gray'))
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Gains',
            showlegend=False,
            y=plot_df['balance_change'].loc[mask_df['gain']],
            meanline={'visible': True},
            #marker_color='green',
            fillcolor=no_comm_col,
            line_color='green'))
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Losses',
            showlegend=False,
            y=plot_df['balance_change'].loc[mask_df['loss']],
            meanline={'visible': True},
            #marker_color='red',
            fillcolor=no_comm_col,
            line_color='red'))

    # show distributions for all positions, only gains, and only losses with 
    #   commissions
    fig.add_trace(
        row=1, col=3, 
        trace=go.Violin(
            name='All',
            showlegend=False,
            y=plot_df['balance_change_commission'], 
            meanline={'visible': True},
            fillcolor=comm_col,
            line_color='gray'))
    fig.add_trace(
        row=1, col=3, 
        trace=go.Violin(
            name='Gains',
            showlegend=False,
            y=plot_df['balance_change_commission'].loc[mask_df['gain_commission']],
            meanline={'visible': True},
            fillcolor=comm_col,
            line_color='green'))
    fig.add_trace(
        row=1, col=3, 
        trace=go.Violin(
            name='Losses',
            showlegend=False,
            y=plot_df['balance_change_commission'].loc[mask_df['loss_commission']],
            meanline={'visible': True},
            fillcolor=comm_col,
            line_color='red'))

    # display rolling median, mean without commissions
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['rolling_mean_positive_balance_change'], 
            mode='lines',
            name='rolling mean',
            marker_color='black',
            fill='tozeroy',
            fillcolor='green'))
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['rolling_mean_negative_balance_change'], 
            mode='lines',
            name='rolling mean',
            marker_color='black',
            fill='tozeroy',
            fillcolor='red'))
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['rolling_median_balance_change'], 
            mode='lines',
            line={'dash': 'dot'},
            name='rolling median',
            marker_color='black'))
    fig.update_xaxes(
        row=2, col=1,
        showticklabels=False,
        showgrid=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['date'].astype(str) + ' ' + unique_symbols)

    # display rolling median, mean with commissions
    fig.add_trace(
        row=3, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['rolling_mean_positive_balance_change_commission'], 
            mode='lines',
            name='rolling mean',
            showlegend=False,
            marker_color='black',
            fill='tozeroy',
            fillcolor='green'))
    fig.add_trace(
        row=3, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['rolling_mean_negative_balance_change_commission'], 
            mode='lines',
            name='rolling mean',
            showlegend=False,
            marker_color='black',
            fill='tozeroy',
            fillcolor='red'))
    fig.add_trace(
        row=3, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], 
            y=plot_df['rolling_median_balance_change_commission'], 
            mode='lines',
            line={'dash': 'dot'},
            name='rolling median',
            showlegend=False,
            marker_color='black'))
    fig.update_xaxes(
        row=3, col=1,
        showticklabels=False,
        showgrid=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['date'].astype(str) + ' ' + unique_symbols)

    fig.update_layout(
        template='plotly_white',
        legend={'orientation': 'h'},
        font_size=16,
        title=f'Balance Change by Day <sub>({len(plot_df)} days)</sub>'
            f'<br><sub><span style="color:{no_comm_col}">Average Balance Change '
                f'w/o Commissions = {balance_means.all}, </span>'
            f'<span style="color:green">Gains = {balance_means.gain}, </span>'
            f'<span style="color:red">Losses = {balance_means.loss}</span></sub>'
            f'<br><sub><span style="color:{comm_col}">Average Balance Change w/ '
                f'Commissions = {balance_means.all_commission}, </span>'
            f'<span style="color:green">Gains = {balance_means.gain_commission}, '
                f'</span>'
            f'<span style="color:red">Losses = {balance_means.loss_commission}'
                f'</span></sub>',
        title_x=0.5,
        yaxis_title='Balance Change')

    return fig


@app.callback(
    Output('price-change-per-share-by-day', 'figure'), 
    [Input('selected-orders', 'data'),
     Input('table-mask', 'data')])
def price_change_per_share_by_day(
    df_json: str, table_mask: str) -> go.Figure:

    if df_json == empty_df_json:
        return go.Figure()


    # set up dataframe for plotting
    ################################################## 

    df = convert_orders_json_to_df(df_json)

    mask = pd.read_json(io.StringIO(table_mask), typ='series')

    colnames = [
        'fill_price_change', 'max_date', 'symbol', 'balance_change', 
        'balance_change_commission']
    plot_df = df.loc[mask, colnames].copy().sort_values('max_date')

    plot_df['date'] = plot_df['max_date'].dt.date
    agg_dict = {
        'symbol': list_of_strings_without_nans, 
        'balance_change': 'sum', 
        'fill_price_change': 'sum', 
        'balance_change_commission': 'sum'}
    plot_df = plot_df.groupby('date').agg(agg_dict).reset_index()
    plot_df['index'] = range(len(plot_df))

    window_len = max(1, len(plot_df) // 8)
    plot_df['rolling_median'] = (
        plot_df['fill_price_change'].rolling(window_len).median())
    plot_df['rolling_mean'] = (
        plot_df['fill_price_change'].rolling(window_len).mean().round(4))
    plot_df['rolling_mean_positive'] = (
        np.where(plot_df['rolling_mean'] > 0, plot_df['rolling_mean'], 0))
    plot_df['rolling_mean_negative'] = (
        np.where(plot_df['rolling_mean'] < 0, plot_df['rolling_mean'], 0))


    # calculate masks for gains and losses and calculate statistics
    ################################################## 

    mask_df = get_balance_change_masks(plot_df)

    balance_means = balance_changes(
        plot_df['fill_price_change'].mean(),
        plot_df.loc[mask_df['gain'], 'fill_price_change'].mean(),
        plot_df.loc[mask_df['loss'], 'fill_price_change'].mean(),
        plot_df['fill_price_change'].mean(),
        plot_df.loc[mask_df['gain_commission'], 'fill_price_change'].mean(),
        plot_df.loc[mask_df['loss_commission'], 'fill_price_change'].mean())


    # make plot 
    ################################################## 

    # when equity/stock symbols are aggregated, individual symbols may repeat;
    #   extract unique symbols for display
    unique_symbols = (
        plot_df['symbol']
        .apply(lambda x: str(set(x))[1:-1].replace("'", ""))
        .str.upper())

    # 'without commissions' color
    no_comm_col = 'lightseagreen'
    # 'with commissions' color
    comm_col = 'orange'

    rolling_title = f'Rolling Median, Mean:  Window Length = {window_len}'
    fig = subplots.make_subplots(
        rows=2, cols=2, 
        shared_yaxes=True, 
        column_widths=[5, 1], 
        row_heights=[8, 1], 
        horizontal_spacing=0,
        vertical_spacing=0.03,
        subplot_titles=['', '', rolling_title, ''])

    # consecutive positions with the same symbol have the same color on the bar 
    #   plot; consecutive positions with different symbols have different colors
    alternating_symbols = (
        ~(plot_df['symbol'] == plot_df['symbol'].shift())).cumsum().mod(2)
    alternating_colors_dict = {0: 'black', 1: 'gray'}
    alternating_colors = alternating_symbols.replace(alternating_colors_dict)

    y_min = plot_df['fill_price_change'].min()
    y_min_with_buffer = y_min + y_min * 0.1
    y_max = plot_df['fill_price_change'].max()
    y_max_with_large_buffer = y_max + y_max * 0.19
    y_max_with_small_buffer = y_max + y_max * 0.08

    # show individual positions
    fig.add_trace(
        row=1, col=1, 
        trace=go.Bar(
            name='All',
            showlegend=True,
            x=plot_df['index'], 
            y=plot_df['fill_price_change'],
            marker_color=alternating_colors))
    fig.update_xaxes(
        row=1, col=1,
        showticklabels=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['date'].astype(str) + ' ' + unique_symbols)
    fig.update_yaxes(
        row=1, col=1,
        range=[y_min_with_buffer, y_max_with_large_buffer])

    # consecutive positions with the same date have the same color on the bar 
    #   plot; consecutive positions with different dates have different colors
    dates_index = get_row_indices_for_date_start(plot_df['date'])
    shade_color='rgba(25,25,25,0.05)'
    for i in range(1, len(dates_index)-1, 2):
        fig.add_shape(
            type='rect',
            xref='x', yref='y',
            x0=dates_index[i], y0=y_min_with_buffer,
            x1=dates_index[i+1], y1=y_max_with_small_buffer,
            fillcolor=shade_color,
            line=dict(color=shade_color, width=0))


    # show distributions for all positions
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='All',
            showlegend=False,
            y=plot_df['fill_price_change'], 
            meanline={'visible': True},
            fillcolor=no_comm_col,
            line_color='gray'))

    # show distributions for all positions, only gains, and only losses without 
    #   commissions
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Gains, w/o C',
            showlegend=True,
            y=plot_df['fill_price_change'].loc[mask_df['gain']],
            meanline={'visible': True},
            fillcolor=no_comm_col,
            line_color='green'))
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Losses, w/o C',
            showlegend=True,
            y=plot_df['fill_price_change'].loc[mask_df['loss']],
            meanline={'visible': True},
            fillcolor=no_comm_col,
            line_color='red'))

    # show distributions for all positions, only gains, and only losses with 
    #   commissions
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Gains, w/ C',
            showlegend=True,
            y=plot_df['fill_price_change'].loc[mask_df['gain_commission']],
            meanline={'visible': True},
            fillcolor=comm_col,
            line_color='green'))
    fig.add_trace(
        row=1, col=2, 
        trace=go.Violin(
            name='Losses, w/ C',
            showlegend=True,
            y=plot_df['fill_price_change'].loc[mask_df['loss_commission']],
            meanline={'visible': True},
            fillcolor=comm_col,
            line_color='red'))
    fig.update_xaxes(
        row=1, col=2,
        showticklabels=False)

    # display rolling median, mean
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], y=plot_df['rolling_mean_positive'], 
            mode='lines',
            name='rolling mean',
            marker_color='black',
            fill='tozeroy',
            fillcolor='green'))
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], y=plot_df['rolling_mean_negative'], 
            mode='lines',
            name='rolling mean',
            marker_color='black',
            fill='tozeroy',
            fillcolor='red'))
    fig.add_trace(
        row=2, col=1, 
        trace=go.Scatter(
            x=plot_df['index'], y=plot_df['rolling_median'], 
            mode='lines',
            line={'dash': 'dot'},
            name='rolling median',
            marker_color='black'))
    fig.update_xaxes(
        row=2, col=1,
        showticklabels=False,
        showgrid=False,
        tickvals=plot_df['index'],
        ticktext=plot_df['date'].astype(str) + ' ' + unique_symbols)

    fig.update_layout(
        template='plotly_white',
        legend={'orientation': 'h'},
        font_size=16,
        title=f'Price Change per Share by Day <sub>({len(plot_df)} days)</sub>'
            f'<br><sub>Average Per-Share Price Change = {balance_means.all}'
            f'<br><span style="color:{no_comm_col}">w/o Commissions:</span> '
            f'<span style="color:green">Gains = {balance_means.gain}</span>, '
            f'<span style="color:red">Losses = {balance_means.loss} </span>'
            f'<br><span style="color:{comm_col}">w/ Commissions:</span> '
            f'<span style="color:green">Gains = {balance_means.gain_commission}</span>, '
            f'<span style="color:red">Losses = {balance_means.loss_commission}</span></sub>',
        title_x=0.5,
        yaxis_title='Price Change per Share')

    return fig


if __name__ == '__main__':
    # for development
    #app.run(debug=True, port=8050, use_reloader=True)
    #app.run(port=8888, debug=True) 

    # for production
    app.run(host='0.0.0.0', port=8050, debug=True)
