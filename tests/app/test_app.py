 
import base64
from typing import Callable, Any, Sequence
from datetime import date
import numpy as np
import pandas as pd

import pytest
from hypothesis import given, settings, reproduce_failure
import hypothesis.strategies as st
import hypothesis.extra.pandas as hpd
import hypothesis.extra.numpy as hnp

from ...src.app.app import (
    date_colnames,
    empty_df_json,
    convert_columns_to_date_type, 
    convert_orders_json_to_df, 
    toggle_classname, 
    select_orders_table, 
    round_power_of_ten, 
    date_range_picker_parameters, 
    hour_of_day_slider_values, 
    fill_price_buy_slider_values, 
    fill_price_buy_slider_range, 
    fill_price_sell_slider_values, 
    fill_price_sell_slider_range, 
    commission_buy_sell_slider_values, 
    commission_buy_sell_slider_range, 
    balance_change_slider_values, 
    balance_change_slider_range, 
    balance_change_commission_slider_values, 
    balance_change_commission_slider_range, 
    shares_num_fill_slider_values, 
    shares_num_fill_slider_range, 
    market_time_of_day_categories, 
    stock_symbol_categories, 
    tags_categories, 
    select_real_simulator_both_trades, 
    filter_dates_to_masks, 
    combine_filter_masks, 
    statistic_description, 
    calculate_trade_statistics, 
    profit_table, 
    number_of_positions_by_date_plot, 
    spent_outflow_by_date_plot, 
    number_of_positions_by_gain_loss_plot, 
    get_balance_change_masks, 
    balance_change_by_position_chronologically, 
    cumulative_balance_change_by_position_chronologically, 
    price_change_per_share_by_position_chronologically, 
    calculate_geometric_mean, 
    calculate_geometric_mean_percent, 
    price_percentage_change_by_position_chronologically, 
    position_hold_times, 
    position_volumes, 
    position_commissions, 
    )

from ...src.app.sidebar import slider_style01

date_colnames = pd.Series(date_colnames).unique().tolist()


##################################################
# HELPER FUNCTION
##################################################

def bdata_convert(bdata_dict: dict[str, str]):
    """ 
    Convert base64-encoded binary data to a Numpy array
    Plotly Dash uses a compact binary encoding for array data in JSON, 
        especially for large or typed arrays. The 'bdata' key contains a 
        base64-encoded binary representation of the array, and 'dtype' 
        specifies the NumPy dtype.
    """ 

    assert 'dtype' in bdata_dict, "Missing 'dtype' in bdata_dict"
    assert 'bdata' in bdata_dict, "Missing 'bdata' in bdata_dict"

    dtype = np.dtype(bdata_dict['dtype'])
    bdata = base64.b64decode(bdata_dict['bdata'])
    return np.frombuffer(bdata, dtype=dtype)


##################################################
# BASIC CONVERSION UTILITIES
##################################################


def test_convert_columns_to_date_type_01():
    """
    Test no columns converted
    """

    colnames = ('col1', 'col2', 'col3')
    assert isinstance(colnames, tuple)

    df1 = pd.DataFrame(
        ((0, 1,    1.5e12),
         (5, 1e12,   1e12)), columns=colnames)

    df2 = convert_columns_to_date_type(df1)

    assert (df2 == df1).all().all()


def test_convert_columns_to_date_type_02():
    """
    Test some columns converted
    """

    colnames = (date_colnames[0], 'col2', date_colnames[-1])

    df1 = pd.DataFrame(
        ((0, 1,    1.5e12),
         (5, 1e12,   1e12)), columns=colnames)

    df2 = convert_columns_to_date_type(df1)

    correct_result = pd.DataFrame(
        ((pd.to_datetime(0, unit='ms'), 1,    pd.to_datetime(1.5e12, unit='ms')),
         (pd.to_datetime(5, unit='ms'), 1e12, pd.to_datetime(  1e12, unit='ms'))), 
        columns=colnames)

    assert (df2 == correct_result).all().all()


def test_convert_columns_to_date_type_03():
    """
    Test all columns converted
    """

    colnames = (date_colnames[0], date_colnames[-1])

    df1 = pd.DataFrame(
        ((0, 1.5e12),
         (5, 1e12)), columns=colnames)

    df2 = convert_columns_to_date_type(df1)

    correct_result = pd.DataFrame(
        ((pd.to_datetime(0, unit='ms'), pd.to_datetime(1.5e12, unit='ms')),
         (pd.to_datetime(5, unit='ms'), pd.to_datetime(  1e12, unit='ms'))), 
        columns=colnames)

    assert (df2 == correct_result).all().all()


@st.composite
def draw_integer_array_and_date_n(
    draw, array_min_side: int=1) -> tuple[np.ndarray, int]:
    """
    Draw samples of Numpy integer arrays (whose columns may be converted to 
        dates), and randomly choose a number of columns to convert to dates
        (which can be no greater than the number of columns in the array)

    NOTE:
    'hypothesis.extra.pandas' is more appropriate for DataFrames with columns of
        heterogeneous data types; homogeneous columns/rows are more efficiently
        generated in Numpy first
    It would be better to refactor this function into two functions:  one to 
        generate the Numpy array, and a second one to generate 'date_n_max';
        however, I'm unsure how to get this to work elegantly with Hypothesis
    """

    shape = draw(
        hnp.array_shapes(
            min_dims=2, max_dims=2, min_side=array_min_side, max_side=20))
    elements = st.integers(min_value=-9e12, max_value=9e12)

    int_array = draw(
        hnp.arrays(
            dtype=draw(hnp.integer_dtypes(endianness='<', sizes=(64,))), 
            shape=shape,
            elements=elements))

    # 'date_n' can be no greater than the number of columns in the array and no
    #   greater than the number of 'date_colnames'
    date_n_max = min(shape[1], len(date_colnames))
    date_n = draw(st.integers(min_value=0, max_value=date_n_max))

    return int_array, date_n 


@given(int_array_date_n=draw_integer_array_and_date_n())
def test_convert_columns_to_date_type_04(
    int_array_date_n: tuple[np.ndarray, int]):
    """
    Test multitude of inputs
    """

    int_array = int_array_date_n[0]
    date_n = int_array_date_n[1]
    column_n = int_array.shape[1]

    # replace randomly chosen column names with names from 'date_colnames'
    colnames = np.array(['col' + str(i) for i in range(column_n)], dtype=object)
    date_replace_idx = np.random.choice(column_n, date_n, False)
    replacement_names = np.random.choice(
        date_colnames, len(date_replace_idx), False)
    colnames[date_replace_idx] = replacement_names

    df1 = pd.DataFrame(int_array, columns=colnames)
    df2 = convert_columns_to_date_type(df1)

    # test that result is exactly correct
    correct_result = df1.copy()
    for e in correct_result.columns:
        if e in date_colnames:
            correct_result[e] = pd.to_datetime(correct_result[e], unit='ms')
    assert (df2 == correct_result).all().all()

    # test that 'date_colnames' columns all have correct date data type
    if len(date_replace_idx) > 0:
        assert df2[replacement_names].dtypes.unique()[0].str == '<M8[ns]'

    # test that non-'date_colnames' columns all have correct integer data type
    non_date_colnames = list(set(df2.columns) - set(date_colnames))
    if len(non_date_colnames) > 0:
        df2[non_date_colnames].dtypes.unique()[0].str == '<i8'


##################################################


def test_convert_orders_json_to_df_01():
    """
    Test empty json input, 'convert_dates' = True
    """

    json_input = empty_df_json

    df1 = convert_orders_json_to_df(json_input, True)

    correct_result = pd.DataFrame()

    assert (df1 == correct_result).all().all()


def test_convert_orders_json_to_df_02():
    """
    Test empty json input, 'convert_dates' = False
    """

    json_input = empty_df_json

    df1 = convert_orders_json_to_df(json_input, False)

    correct_result = pd.DataFrame()

    assert (df1 == correct_result).all().all()


def test_convert_orders_json_to_df_03():
    """
    Test json input, 'convert_dates' = True
    """

    json_input = (
        '{"columns":["col1", "col2", "' + date_colnames[0] + '"],'
        '"index":[0, 1],'
        '"data":[[0, 1, 1.5e12], [2, 3, 1e12]]}')

    df1 = convert_orders_json_to_df(json_input, True)

    colnames = ('col1', 'col2', date_colnames[0])
    correct_result = pd.DataFrame(
        ((0, 1, pd.to_datetime(1.5e12, unit='ms')),
         (2, 3, pd.to_datetime(  1e12, unit='ms'))), columns=colnames)

    assert (df1 == correct_result).all().all()


def test_convert_orders_json_to_df_04():
    """
    Test json input, 'convert_dates' = False
    """

    json_input = (
        '{"columns":["col1", "col2", "' + date_colnames[0] + '"],'
        '"index":[0, 1],'
        '"data":[[0, 1, 1.5e12], [2, 3, 1e12]]}')

    df1 = convert_orders_json_to_df(json_input, False)

    colnames = ('col1', 'col2', date_colnames[0])
    correct_result = pd.DataFrame(
        ((0, 1, 1.5e12),
         (2, 3,   1e12)), columns=colnames)

    assert (df1 == correct_result).all().all()


def convert_integer_array_to_json(
    int_array: np.ndarray, date_n: int) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Convert integer array to JSON string that can represent a Pandas DataFrame
    Replace a random number ('date_n') of column names with names from a list of
        pre-specified column names ('date_colnames')

    NOTE:
    Assumes that JSON input will always have fields 'columns', 'index', and
        'data'
    """

    row_n = int_array.shape[0]
    column_n = int_array.shape[1]

    # replace randomly chosen column names with names from 'date_colnames'
    colnames = np.array(['col' + str(i) for i in range(column_n)], dtype=object)
    date_replace_idx = np.random.choice(column_n, date_n, False)
    replacement_names = np.random.choice(
        date_colnames, len(date_replace_idx), False)
    colnames[date_replace_idx] = replacement_names

    # construct JSON input from integer array
    json_colnames = str(['"' + e + '"' for e in colnames]).replace("\'", "")
    json_idx = str(list(range(row_n))).replace("\'", "")
    json_data = [int_array[i, :].tolist() for i in range(row_n)]
    json_input = (
        '{"columns":' + json_colnames + ','
        '"index":' + json_idx + ','
        '"data":' + str(json_data) + '}')
    json_input = json_input.replace("'", '"')

    return json_input, colnames, date_replace_idx 


@given(
    int_array_date_n=draw_integer_array_and_date_n(),
    convert_dates=st.booleans())
@settings(print_blob=True)
def test_convert_orders_json_to_df_05(
    int_array_date_n: tuple[np.ndarray, int], convert_dates: bool):
    """
    Test multitude of inputs
    """

    int_array = int_array_date_n[0]
    date_n = int_array_date_n[1]
    json_input, colnames, _ = convert_integer_array_to_json(int_array, date_n)

    df1 = convert_orders_json_to_df(json_input, convert_dates)

    # create correct result using already-tested function 
    #   'convert_columns_to_date_type'
    int_df = pd.DataFrame(int_array, columns=colnames)
    if convert_dates:
        correct_result = convert_columns_to_date_type(int_df)
    else:
        correct_result = int_df

    # test that result is exactly correct
    assert (df1 == correct_result).all().all()


##################################################
# PARAMETERS FOR SIDEBAR USER SELECTIONS
##################################################


def test_toggle_classname_01():
    """
    Test 'collapsed' result
    """

    n = 1
    classname = ''

    result = toggle_classname(n, classname)

    assert result == 'collapsed'


def test_toggle_classname_02():
    """
    Test non-'collapsed' result
    """

    n = 0
    classname = ''

    result = toggle_classname(n, classname)

    assert result == ''


def test_toggle_classname_03():
    """
    Test non-'collapsed' result
    """

    n = 1
    classname = 'test'

    result = toggle_classname(n, classname)

    assert result == ''


##################################################


def test_select_orders_table_01():
    """
    Test 'position-change' input
    """

    pathname = '/position-change'

    # 'orders_*' inputs string are actually JSON, but tested function merely 
    #   switches among them, so test only that correct string is returned
    result = select_orders_table(
        pathname, 'orders_position_change', 'orders_ongoing_position', 
        'orders_symbol_day')

    assert result == 'orders_position_change'


def test_select_orders_table_02():
    """
    Test 'ongoing-position' input
    """

    pathname = '/ongoing-position'

    # 'orders_*' inputs string are actually JSON, but tested function merely 
    #   switches among them, so test only that correct string is returned
    result = select_orders_table(
        pathname, 'orders_position_change', 'orders_ongoing_position', 
        'orders_symbol_day')

    assert result == 'orders_ongoing_position'


def test_select_orders_table_03():
    """
    Test 'symbol-day' input
    """

    pathname = '/symbol-day'

    # 'orders_*' inputs string are actually JSON, but tested function merely 
    #   switches among them, so test only that correct string is returned
    result = select_orders_table(
        pathname, 'orders_position_change', 'orders_ongoing_position', 
        'orders_symbol_day')

    assert result == 'orders_symbol_day'


def test_select_orders_table_04():
    """
    Test invalid input
    """

    pathname = ''

    # 'orders_*' inputs string are actually JSON, but tested function merely 
    #   switches among them, so test only that correct string is returned
    result = select_orders_table(
        pathname, 'orders_position_change', 'orders_ongoing_position', 
        'orders_symbol_day')

    assert result == 'orders_position_change'


##################################################

from math import log, ceil, floor


@given(value=st.floats(min_value=1e-6, max_value=1e12))
def test_round_power_of_ten_01(value: float):
    """
    This test is arguably superfluous, because it merely tests that the code in
        the function 'round_power_of_ten' is the same code below
    However, in thinking about tests as a way to show how the code works, this 
        test explicitly denotes the 'min_value' of 'value'
    """

    result = round_power_of_ten(value, high_order_magnitude=False)

    assert result == floor(log(value, 10) * 100) / 100

    # verify that there are no more than 2 digits to the right of the decimal
    assert len(str(result).split('.')[-1]) <= 2


@given(value=st.floats(min_value=1e-6, max_value=1e12))
def test_round_power_of_ten_02(value: float):
    """
    This test is arguably superfluous, because it merely tests that the code in
        the function 'round_power_of_ten' is the same code below
    However, given that tests can be a way to show how the code works, this test 
        explicitly denotes the 'min_value' of 'value'
    """

    result = round_power_of_ten(value, high_order_magnitude=True)

    assert result == ceil(log(value, 10) * 100) / 100

    # verify that there are no more than 2 digits to the right of the decimal
    assert len(str(result).split('.')[-1]) <= 2


@given(value=st.floats(min_value=-1e12, max_value=-1.e-6))
def test_round_power_of_ten_03(value: float):

    with pytest.raises(ValueError):
        round_power_of_ten(value, high_order_magnitude=True)


# the tests of the function 'round_power_of_ten' are included to provide the 
#   reader with some easy-to-understand examples of what the function is doing

def test_round_power_of_ten_04():

    value = 1
    result = round_power_of_ten(value, high_order_magnitude=True)
    assert result == 0


def test_round_power_of_ten_05():

    value = 1
    result = round_power_of_ten(value, high_order_magnitude=False)
    assert result == 0


def test_round_power_of_ten_06():

    value = 0.9
    result = round_power_of_ten(value, high_order_magnitude=True)
    assert result == -0.04


def test_round_power_of_ten_07():

    value = 0.9
    result = round_power_of_ten(value, high_order_magnitude=False)
    assert result == -0.05


def test_round_power_of_ten_08():

    value = 1.3
    result = round_power_of_ten(value, high_order_magnitude=True)
    assert result == 0.12


def test_round_power_of_ten_09():

    value = 1.3
    result = round_power_of_ten(value, high_order_magnitude=False)
    assert result == 0.11


##################################################


def test_date_range_picker_parameters_01():
    """
    Test empty json input
    """

    json_input = empty_df_json

    result = date_range_picker_parameters(json_input)

    dt = date.today()
    correct_result = (dt, dt, dt, dt, dt)

    assert result == correct_result


@given(int_array_date_n=draw_integer_array_and_date_n())
@settings(print_blob=True)
def test_date_range_picker_parameters_02(
    int_array_date_n: tuple[np.ndarray, int]):
    """
    Test multitude of inputs
    """

    int_array = int_array_date_n[0]

    # assume that there is at least one date column
    date_n = max(1, int_array_date_n[1])

    json_input, _, date_replace_idx = convert_integer_array_to_json(
        int_array, date_n)

    ed1, ld1, ld2, ed2, ld3 = date_range_picker_parameters(json_input)

    # calculate correct results
    date_int_min = int_array[:, date_replace_idx].min()
    date_int_max = int_array[:, date_replace_idx].max()
    correct_earliest_date = pd.to_datetime(date_int_min, unit='ms')
    correct_latest_date = pd.to_datetime(date_int_max, unit='ms')

    assert correct_earliest_date.date() == ed1
    assert correct_latest_date.date() == ld1 == ld2 == ld3


# Slider values
##################################################


@given(float_list=st.lists(
    st.floats(min_value=0, max_value=99), 
    min_size=2, 
    max_size=2))
@settings(print_blob=True)
def test_hour_of_day_slider_values(float_list: list[float]):
    """
    Most of the functionality of 'hour_of_day_slider_values' is in the function 
        'convert_hour_to_string', which is tested elsewhere
    """

    result = hour_of_day_slider_values(float_list)

    assert len(result) == 2

    # test that the returned strings are consistent with representing times
    result01 = result[0]
    result02 = result[1]

    assert ':' in result01
    assert ':' in result02

    assert len(result01) == 4 or len(result01) == 5
    assert len(result02) == 4 or len(result02) == 5
    

# Slider values
##################################################

@pytest.mark.skip(
    reason='Passed function produces error if this function runs independently')
def test_slider_values(
    slider_values: Callable, float_list: list[float], log_values: bool):
    """
    The sliders with logarithm values are tested the same way
    The sliders are 'buy', 'sell', 'commission', and 'shares_num'
    """

    result = slider_values(float_list)

    assert len(result) == 2

    result01 = result[0]
    result02 = result[1]

    if log_values:
        assert result01 == round(10 ** float_list[0], 2)
        assert result02 == round(10 ** float_list[1], 2)
    else:
        assert result01 == round(float_list[0], 2)
        assert result02 == round(float_list[1], 2)
    

@given(float_list=st.lists(
    st.floats(min_value=-100, max_value=100, allow_infinity=False), 
    min_size=2, 
    max_size=2))
def test_fill_price_buy_slider_values(float_list: list[float]):
    test_slider_values(fill_price_buy_slider_values, float_list, True)


@given(float_list=st.lists(
    st.floats(min_value=-100, max_value=100, allow_infinity=False), 
    min_size=2, 
    max_size=2))
def test_fill_price_sell_slider_values(float_list: list[float]):
    test_slider_values(fill_price_sell_slider_values, float_list, True)


@given(float_list=st.lists(
    st.floats(min_value=-100, max_value=100, allow_infinity=False), 
    min_size=2, 
    max_size=2))
def test_commission_slider_values(float_list: list[float]):
    test_slider_values(commission_buy_sell_slider_values, float_list, True)


@given(float_list=st.lists(
    st.floats(min_value=-100, max_value=100, allow_infinity=False), 
    min_size=2, 
    max_size=2))
def test_balance_change_slider_values(float_list: list[float]):
    test_slider_values(balance_change_slider_values, float_list, False)


@given(float_list=st.lists(
    st.floats(min_value=-100, max_value=100, allow_infinity=False), 
    min_size=2, 
    max_size=2))
def test_balance_change_commission_slider_values(float_list: list[float]):
    test_slider_values(
        balance_change_commission_slider_values, float_list, False)


@given(float_list=st.lists(
    st.floats(min_value=-100, max_value=100, allow_infinity=False), 
    min_size=2, 
    max_size=2))
def test_shares_num_fill_slider_values(float_list: list[float]):
    test_slider_values(shares_num_fill_slider_values, float_list, True)


'''
# Slider display toggle
##################################################


@pytest.mark.skip(
    reason='Passed function produces error if this function runs independently')
def test_fill_price_slider_display_toggle_01(buy_or_sell: Callable):
    """
    The toggle for the buy slider and the sell slider are tested the same way
    Test empty json input
    """

    json_input = empty_df_json

    result = buy_or_sell(json_input)

    assert result == {'display': 'none'}


def test_fill_price_buy_slider_display_toggle_01():
    test_fill_price_slider_display_toggle_01(
        fill_price_buy_slider_display_toggle)


def test_fill_price_sell_slider_display_toggle_01():
    test_fill_price_slider_display_toggle_01(
        fill_price_sell_slider_display_toggle)


@st.composite
def draw_string_list_and_index(draw) -> tuple[list[str], int]:
    """
    Generate list of strings and an index to that list
    """

    string_list = draw(
        st.lists(
            st.text(
                # whitelist:  include only letters
                alphabet=st.characters(whitelist_categories='L'), 
                min_size=1), 
            min_size=1))
    idx = draw(st.integers(min_value=0, max_value=len(string_list)-1))

    return string_list, idx


@pytest.mark.skip(
    reason='Passed function produces error if this function runs independently')
def test_fill_price_slider_display_toggle_02(
    buy_or_sell: Callable, target_colname: str, 
    string_list_idx: tuple[list[str], int], include_target_colname: bool):
    """
    The toggle for the buy slider and the sell slider are tested the same way
    Test that 'fill_price_buy_slider_display_toggle' (or 
        'fill_price_sell_slider_display_toggle') returns the correct slider
        style based on whether the target column name 'fill_price_buy' (or 
        'fill_price_sell') is in the JSON input
    """

    string_list = string_list_idx[0]
    idx = string_list_idx[1]

    if include_target_colname:
        string_list[idx] = target_colname
    json_colnames = str(['"' + e + '"' for e in string_list]).replace("\'", "")

    # 'data' field might not match randomly generated 'columns', yielding a JSON
    #   that is invalid for representing a Pandas DataFrame, but the function
    #   of 'fill_price_buy_slider_display_toggle' depends only on 'columns',
    #   so this invalidity is ignored
    json_input = (
        '{"columns":' + json_colnames + ','
        '"index":[0, 1],'
        '"data":[[0, 1, 1.5e12], [2, 3, 1e12]]}')

    result = buy_or_sell(json_input)

    if include_target_colname:
        assert result == slider_style01
    else:
        assert result == {'display': 'none'}
    

@given(
    string_list_idx=draw_string_list_and_index(),
    include_target_colname=st.booleans())
def test_fill_price_buy_slider_display_toggle_02(
    string_list_idx: tuple[list[str], int], include_target_colname: bool):

    test_fill_price_slider_display_toggle_02(
        fill_price_buy_slider_display_toggle, 'fill_price_buy', 
        string_list_idx, include_target_colname)
    

@given(
    string_list_idx=draw_string_list_and_index(),
    include_target_colname=st.booleans())
def test_fill_price_sell_slider_display_toggle_02(
    string_list_idx: tuple[list[str], int], include_target_colname: bool):

    test_fill_price_slider_display_toggle_02(
        fill_price_sell_slider_display_toggle, 'fill_price_sell', 
        string_list_idx, include_target_colname)
'''
    

# Slider range:  empty input
##################################################


@pytest.mark.skip(
    reason='Passed function produces error if this function runs independently')
def test_slider_range_empty_input(slider_range: Callable, default_result: int):
    """
    Test empty json input
    """

    json_input = empty_df_json

    result = slider_range(json_input)

    dr = default_result
    assert result == (dr, dr, (dr, dr))


def test_fill_price_buy_slider_range_01():
    test_slider_range_empty_input(fill_price_buy_slider_range, -2)


def test_fill_price_sell_slider_range_01():
    test_slider_range_empty_input(fill_price_sell_slider_range, 0)


def test_commission_slider_range_01():
    test_slider_range_empty_input(commission_buy_sell_slider_range, 0)


def test_balance_change_slider_range_01():
    test_slider_range_empty_input(balance_change_slider_range, 0)


def test_balance_change_commission_slider_range_01():
    test_slider_range_empty_input(balance_change_commission_slider_range, 0)


def test_shares_num_fill_slider_range_01():
    test_slider_range_empty_input(shares_num_fill_slider_range, 0)


# Slider range:  fill price (buy and sell)
##################################################


@pytest.mark.skip(
    reason='Passed function produces error if this function runs independently')
def test_fill_price_slider_range(
    slider_range: Callable, int_array_date_n: tuple[np.ndarray, int], 
    insert_target_colname: bool, target_colname: str):
    """
    Test multitude of inputs
    """

    int_array = int_array_date_n[0]
    date_n = int_array_date_n[1]
    json_input, colnames, date_replace_idx = convert_integer_array_to_json(
        int_array, date_n)

    # examples generated by existing code include target_colname too rarely, so
    #   insert it based on Boolean 'insert_target_colname', assuming there is a
    #   place to insert it (i.e., 'len(date_replace_idx) > 0')
    if insert_target_colname and len(date_replace_idx) > 0:
        rng = np.random.default_rng()
        replace_idx = rng.choice(len(date_replace_idx), 1)
        json_input = json_input.replace(colnames[replace_idx][0], target_colname)
        colnames[replace_idx] = target_colname

    #dr = default_result
    dr = -2
    if target_colname in colnames:
        col_idx = np.where(colnames == target_colname)[0]
        col_max = int_array[:, col_idx].max()

        # 'round_power_of_ten' entails taking a log, so 'col_max' must be > 0
        if col_max > 0:
            result = slider_range(json_input)
            range_max = 1.02 * round_power_of_ten(col_max)
            assert result == (dr, range_max, (dr, range_max))

    else:
        result = slider_range(json_input)
        assert result == (dr, dr, (dr, dr))


@given(
    int_array_date_n=draw_integer_array_and_date_n(),
    insert_target_colname=st.booleans())
def test_fill_price_buy_slider_range_02(
    int_array_date_n: tuple[np.ndarray, int], insert_target_colname: bool):

    test_fill_price_slider_range(
        fill_price_buy_slider_range, int_array_date_n, insert_target_colname, 
        'fill_price_buy')


@given(
    int_array_date_n=draw_integer_array_and_date_n(),
    insert_target_colname=st.booleans())
def test_fill_price_sell_slider_range_02(
    int_array_date_n: tuple[np.ndarray, int], insert_target_colname: bool):

    test_fill_price_slider_range(
        fill_price_sell_slider_range, int_array_date_n, insert_target_colname, 
        'fill_price_sell')


# Slider range:  commission
##################################################


@st.composite
def draw_integer_array_and_column_idx(
    draw, array_min_side: int=1) -> tuple[np.ndarray, int]:
    """
    Draw samples of Numpy integer arrays and randomly choose a column index
    """

    shape = draw(
        hnp.array_shapes(
            min_dims=2, max_dims=2, min_side=array_min_side, max_side=20))
    elements = st.integers(min_value=-9e12, max_value=9e12)

    int_array = draw(
        hnp.arrays(
            dtype=draw(hnp.integer_dtypes(endianness='<', sizes=(64,))), 
            shape=shape,
            elements=elements))

    column_idx = draw(st.integers(min_value=0, max_value=int_array.shape[1]-1))

    return int_array, column_idx  


@given(
    int_array_col_idx=draw_integer_array_and_column_idx(array_min_side=2),
    insert_target_colname=st.booleans())
@settings(print_blob=True)
def test_commission_slider_range_02(
    int_array_col_idx: tuple[np.ndarray, int], insert_target_colname: bool):
    """
    Test multitude of inputs
    """

    int_array = int_array_col_idx[0]

    # columns 'commission_cost', 'commission_cost_buy', and 
    #   'commission_cost_sell' cannot have negative numbers, so remove them
    int_array = np.abs(int_array)

    col_idx = int_array_col_idx[1]
    json_input, colnames, _ = convert_integer_array_to_json(int_array, 0)

    # either column 'commission_cost' or columns 'commission_cost_buy' and
    #   'commission_cost_sell' must be present
    if insert_target_colname:
        json_input = json_input.replace(
            '"' + colnames[col_idx] + '"', '"commission_cost"')
        col_max = int_array[:, col_idx].max()

    else:
        json_input = json_input.replace(
            '"' + colnames[col_idx] + '"', '"commission_cost_buy"')

        # randomly select a second column to designate as 'commission_cost_sell'
        col2_idx_options = [i for i in range(int_array.shape[1])]
        col2_idx_options.remove(col_idx)
        rng = np.random.default_rng(seed=96445)
        col2_idx = rng.choice(col2_idx_options, size=1)[0]
        json_input = json_input.replace(
            colnames[col2_idx], 'commission_cost_sell')

        col_idxs = np.array([col_idx, col2_idx])
        col_max = int_array[:, col_idxs].sum(axis=1).max()


    range_min = -2

    # 'round_power_of_ten' entails taking a log, so 'col_max' must be > 0
    if col_max > 0:
        range_max = 1.02 * round_power_of_ten(col_max)
        result = commission_buy_sell_slider_range(json_input)
        assert result == (range_min, range_max, (range_min, range_max))
    else:
        pass


# Slider range:  balance change
##################################################


@pytest.mark.skip(
    reason='Passed function produces error if this function runs independently')
def test_balance_change_slider_range(
    slider_range: Callable, int_array_col_idx: tuple[np.ndarray, int], 
    target_colname: str):
    """
    Test multitude of inputs
    """

    int_array = int_array_col_idx[0]
    col_idx = int_array_col_idx[1]
    json_input, colnames, _ = convert_integer_array_to_json(int_array, 0)
    json_input = json_input.replace(colnames[col_idx], target_colname)

    col_min = int_array[:, col_idx].min()
    col_max = int_array[:, col_idx].max()

    result = slider_range(json_input)

    assert result == (col_min, col_max, (col_min, col_max))


@given(int_array_col_idx=draw_integer_array_and_column_idx())
def test_balance_change_slider_range_02(
    int_array_col_idx: tuple[np.ndarray, int]):

    test_balance_change_slider_range(
        balance_change_slider_range, int_array_col_idx, 'balance_change')


@given(int_array_col_idx=draw_integer_array_and_column_idx())
def test_balance_change_commission_slider_range_02(
    int_array_col_idx: tuple[np.ndarray, int]):

    test_balance_change_slider_range(
        balance_change_commission_slider_range, int_array_col_idx, 
        'balance_change_commission')


# Slider range:  shares num fill
##################################################


@given(
    int_array_col_idx=draw_integer_array_and_column_idx(),
    insert_target_colname=st.booleans())
def test_shares_num_fill_slider_range_02(
    int_array_col_idx: tuple[np.ndarray, int], insert_target_colname: bool):
    """
    Test multitude of inputs
    """

    int_array = int_array_col_idx[0]

    # columns 'shares_num_fill' and 'match_shares_num_fill' cannot have negative 
    #   numbers, so remove them
    int_array = np.abs(int_array)

    col_idx = int_array_col_idx[1]
    json_input, colnames, _ = convert_integer_array_to_json(int_array, 0)

    if insert_target_colname:
        json_input = json_input.replace(colnames[col_idx], 'shares_num_fill')
    else:
        json_input = json_input.replace(
            colnames[col_idx], 'match_shares_num_fill')

    col_min = int_array[:, col_idx].min()
    col_max = int_array[:, col_idx].max()

    # 'round_power_of_ten' entails taking a log, so input must be > 0
    if col_min > 0 and col_max > 0:
        range_min = round_power_of_ten(col_min, False) 
        range_max = 1.02 * round_power_of_ten(col_max)
        result = shares_num_fill_slider_range(json_input)
        assert result == (range_min, range_max, (range_min, range_max))
    else:
        pass


# Categories:  empty input
##################################################


def test_market_time_of_day_categories_01():
    """
    Test empty json input
    """

    json_input = empty_df_json

    result = market_time_of_day_categories(json_input)

    assert len(result) == 2
    assert result[0] == [{'': ''}]
    assert result[1].empty


def test_stock_symbol_categories_01():
    """
    Test empty json input
    """

    json_input = empty_df_json

    result = stock_symbol_categories(json_input, [], [])

    assert len(result) == 3
    assert result == ([], [{'': ''}], [])



def test_tags_categories_01():
    """
    Test empty json input
    """

    json_input = empty_df_json

    result = tags_categories(json_input, [], [])

    assert len(result) == 3
    assert result == ([], [{'': ''}], [])


# Categories:  non-empty input
##################################################


@st.composite
def draw_string_array_and_column_idx(
    draw, array_min_side: int=1, char_min_size: int=1) -> tuple[np.ndarray, int]:
    """
    Draw samples of Numpy integer arrays and randomly choose a column index
    """

    shape = draw(
        hnp.array_shapes(
            min_dims=2, max_dims=2, min_side=array_min_side, max_side=20))

    str_array = draw(
        hnp.arrays(
            dtype=np.str_,
            shape=shape,
            elements=st.text(
                alphabet=st.characters(whitelist_categories='L'),
                min_size=char_min_size)))

    column_idx = draw(st.integers(min_value=0, max_value=str_array.shape[1]-1))

    return str_array, column_idx  


@given(str_array_col_idx=draw_string_array_and_column_idx())
@settings(print_blob=True)
def test_market_time_of_day_categories_02(
    str_array_col_idx: tuple[np.ndarray, int]):
    """
    Test multitude of inputs

    NOTE:
    The original idea behind this test was to use Numpy where the tested function 
        uses Pandas to calculate the same thing; after all, if one calculates the
        same thing in two (or more) different ways and gets the same result, one
        can be more confident that the calculations are correct
    However, the Pandas string 'title' function returns results for some inputs 
        that are difficult to fully account for and reproduce independently; thus, 
        this test ends up being rather convoluted and should probably be revised
    """

    str_array = str_array_col_idx[0]
    col_idx = str_array_col_idx[1]
    json_input, colnames, _ = convert_integer_array_to_json(str_array, 0)

    # insert correct column name:  'max_date_market_time'
    json_input = json_input.replace(colnames[col_idx], 'max_date_market_time')

    # get unique values in order of their appearance in the correct column 
    value, order_idx = np.unique(str_array[:, col_idx], return_index=True)
    value = value[np.argsort(order_idx)]

    options = [
        {'label': pd.Series(e).str.title().iloc[0], 
         'value': e.lower()} 
        for e in value]

    result = market_time_of_day_categories(json_input)

    result0 = [{k: v for k, v in e.items()} for e in result[0]]
    for e in result0:
        e['value'] = e['value'].lower()

    assert result0 == options
    assert (result[1] == value).all()



@pytest.mark.skip(
    reason='TODO: Produces unsolved error: FAILED '
    'tests/app/test_app.py::test_stock_symbol_categories_02 - '
    'dash.exceptions.MissingCallbackContextException: '
    'dash.callback_context.triggered is only available from a callback!')
@given(str_array_col_idx=draw_string_array_and_column_idx())
@settings(print_blob=True)
def test_stock_symbol_categories_02(
    str_array_col_idx: tuple[np.ndarray, int]):
    """
    Test multitude of inputs
    """

    str_array = str_array_col_idx[0]
    col_idx = str_array_col_idx[1]
    json_input, colnames, _ = convert_integer_array_to_json(str_array, 0)

    # insert correct column name:  'symbol'
    json_input = json_input.replace(colnames[col_idx], 'symbol')

    # get unique values in order of their appearance in the correct column 
    value, order_idx = np.unique(str_array[:, col_idx], return_index=True)
    value = value[np.argsort(order_idx)]

    options = [{'label': e.upper(), 'value': e} for e in value]

    result = stock_symbol_categories(json_input, [], [])

    result0 = [{k: v for k, v in e.items()} for e in result[0]]

    assert result0 == options
    assert (result[1] == value).all()



@pytest.mark.skip(
    reason='TODO: Produces unsolved error: FAILED '
    'tests/app/test_app.py::test_tags_categories_02 - '
    'dash.exceptions.MissingCallbackContextException: '
    'dash.callback_context.triggered is only available from a callback!')
@given(str_array_col_idx=draw_string_array_and_column_idx(char_min_size=4))
@settings(print_blob=True)
def test_tags_categories_02(
    str_array_col_idx: tuple[np.ndarray, int]):
    """
    Test multitude of inputs

    In the 'tags' column of the dataframe generated from the JSON input, each
        cell may have multiple tags separated by a divider (comma-whitespace, 
        ', ')
    These tags need to be separated and put into a list (or other 
        array/iterable) so that each one can be assigned to its own dictionary
        ('options')      
    """

    str_array = str_array_col_idx[0]

    # simple, easy, slow way to insert tag-dividing characters (a comma, followed 
    #   by a whitespace) into some test cases
    # NOTE:  it would be better for Hypothesis to randomize comma insertion
    for i in range(str_array.shape[0]):
        for j in range(str_array.shape[1]):
            if len(str_array[i, j]) > 3:
                str_array[i, j] = (
                    str_array[i, j][:-3] + ', ' + str_array[i, j][-2:])

    col_idx = str_array_col_idx[1]
    json_input, colnames, _ = convert_integer_array_to_json(str_array, 0)

    # test both valid column names 'tags' and 'tags_buy'
    for colname in ('tags', 'tags_buy'):

        # insert correct column name:  'tags'
        json_input = json_input.replace(colnames[col_idx], colname)

        # get unique values in order of their appearance in the correct column 
        value, order_idx = np.unique(str_array[:, col_idx], return_index=True)
        value = value[np.argsort(order_idx)]

        # divide tags by the comma-whitespace (', ') and assign each unique tag 
        #   to a list
        unique_value_tags = []
        for e in value:
            value_split = e.split(', ')
            for f in value_split:
                if f not in unique_value_tags and f != '' and f != None:
                    unique_value_tags.append(f)

        options = [{'label': e, 'value': e} for e in unique_value_tags]

        result = tags_categories(json_input, [], [])

        assert result == options



##################################################
# DATA PIPELINE FOR (PROVIDED) TABLE
##################################################
# This section's functions input and output Pandas DataFrames and are tested with 
#   Pandera-specified schemas
##################################################



##################################################
# FILTER DATA BASED ON USER SELECTIONS
##################################################


# Real or Simulator:  return True
##################################################


def test_select_real_simulator_both_trades_01():
    result = select_real_simulator_both_trades(
        ['Simulator'], pd.Series('simulator'))
    assert len(result) == 1
    assert result.iloc[0] == True


def test_select_real_simulator_both_trades_02():
    result = select_real_simulator_both_trades(
        ['Real'], pd.Series('real'))
    assert len(result) == 1
    assert result.iloc[0] == True


def test_select_real_simulator_both_trades_03():
    result = select_real_simulator_both_trades(
        ['Simulator', 'Real'], pd.Series('simulator'))
    assert len(result) == 1
    assert result.iloc[0] == True


def test_select_real_simulator_both_trades_04():
    result = select_real_simulator_both_trades(
        ['Simulator', 'Real'], pd.Series('real'))
    assert len(result) == 1
    assert result.iloc[0] == True


# Real or Simulator:  return False
##################################################


def test_select_real_simulator_both_trades_05():
    result = select_real_simulator_both_trades(
        ['simulator'], pd.Series('simulator'))
    assert len(result) == 1
    assert result.iloc[0] == False


def test_select_real_simulator_both_trades_06():
    result = select_real_simulator_both_trades(
        ['real'], pd.Series('real'))
    assert len(result) == 1
    assert result.iloc[0] == False


def test_select_real_simulator_both_trades_07():
    result = select_real_simulator_both_trades(
        ['Simulator', 'real'], pd.Series('simulator'))
    assert len(result) == 1
    assert result.iloc[0] == False


def test_select_real_simulator_both_trades_08():
    result = select_real_simulator_both_trades(
        ['simulator', 'Real'], pd.Series('simulator'))
    assert len(result) == 1
    assert result.iloc[0] == False


def test_select_real_simulator_both_trades_09():
    result = select_real_simulator_both_trades(
        ['Simulator', 'Real'], pd.Series(''))
    assert len(result) == 1
    assert result.iloc[0] == False


# Filter dates to masks
##################################################


def test_filter_dates_to_masks_01():

    min_col = 'min'
    max_col = 'max'
    df = pd.DataFrame({
        min_col: [1, 2, 3, 4, 1, 2, 3, 4],
        max_col: [1, 2, 3, 1, 2, 3, 1, 2]})
    
    selection = [2, 3]
    both_cols = True

    result = filter_dates_to_masks(df, both_cols, min_col, max_col, selection)

    mask = pd.Series([False, True, True, False, False, True, False, False])

    assert (result == mask).all()


def test_filter_dates_to_masks_02():

    min_col = 'min'
    max_col = 'max'
    df = pd.DataFrame({
        min_col: [1, 2, 3, 4, 1, 2, 3, 4],
        max_col: [1, 2, 3, 1, 2, 3, 1, 2]})
    
    selection = [2, 3]
    both_cols = False

    result = filter_dates_to_masks(df, both_cols, min_col, max_col, selection)

    mask = pd.Series([False, True, True, False, True, True, False, True])

    assert (result == mask).all()


@st.composite
def draw_integer_array(draw, dim=2, array_min_side: int=1) -> np.ndarray:
    """
    Draw samples of Numpy integer arrays
    """

    shape = draw(
        hnp.array_shapes(
            min_dims=dim, max_dims=dim, min_side=array_min_side, max_side=20))
    elements = st.integers(min_value=-9e12, max_value=9e12)

    int_array = draw(
        hnp.arrays(
            dtype=draw(hnp.integer_dtypes(endianness='<', sizes=(64,))), 
            shape=shape,
            elements=elements))

    return int_array


@st.composite
def draw_dataframe_with_column_idxs(
    draw, array_min_side: int=1
    ) -> tuple[pd.DataFrame, tuple[int, int], list[int]]:
    """
    Draw samples of Pandas integer dataframes, two column indices from each 
        dataframe, and unique elements from those two columns
    """

    int_array = draw(draw_integer_array(array_min_side=array_min_side))

    # select columns
    col_idx_1 = draw(st.integers(min_value=0, max_value=int_array.shape[1]-1))
    col_idx_2 = draw(st.integers(min_value=0, max_value=int_array.shape[1]-1))

    # ensure that the two column indices are different
    if col_idx_1 == col_idx_2:
        col_idx_2 = col_idx_1 + 1
    if col_idx_2 > int_array.shape[1]-1:
        col_idx_2 = col_idx_1 - 1

    colnames = ['col' + str(i) for i in range(int_array.shape[1])]
    df = pd.DataFrame(int_array, columns=colnames)

    # select elements from columns
    col_elements = np.unique(
        np.append(
            int_array[:, col_idx_1], 
            int_array[:, col_idx_2]))
    assert isinstance(col_elements, np.ndarray)

    col_elements_sampling = st.sampled_from(col_elements)
    min_size = max(1, len(col_elements) // 2)
    col_elements_sample = draw(
        st.lists(
            col_elements_sampling, 
            min_size=min_size, 
            max_size=len(col_elements)))

    return df, (col_idx_1, col_idx_2), col_elements_sample 


@given(
    df_idx_sample=draw_dataframe_with_column_idxs(array_min_side=2),
    both_cols=st.booleans())
@settings(print_blob=True)
def test_filter_dates_to_masks_03(
    df_idx_sample: tuple[pd.DataFrame, tuple[int, int], list[int]], 
    both_cols: bool):

    df = df_idx_sample[0]
    idx = df_idx_sample[1]
    sample = df_idx_sample[2]
    min_colname = df.columns[idx[0]] 
    max_colname = df.columns[idx[1]] 
    assert isinstance(min_colname, str)
    assert isinstance(max_colname, str)

    result = filter_dates_to_masks(
        df, both_cols, min_colname, max_colname, sample)

    if both_cols:
        mask = (
            df.iloc[:, idx[0]].isin(sample) & 
            df.iloc[:, idx[1]].isin(sample))
    else:
        mask = df.iloc[:, idx[1]].isin(sample)

    assert (result == mask).all()


# Combine filter masks
##################################################


@st.composite
def draw_boolean_array(draw, array_min_side: int=1) -> np.ndarray:
    """
    Draw samples of Numpy Boolean arrays
    """

    shape = draw(
        hnp.array_shapes(
            min_dims=2, max_dims=2, min_side=array_min_side, max_side=20))
    elements = st.booleans()

    bool_array = draw(
        hnp.arrays(dtype=np.bool_, shape=shape, elements=elements))

    return bool_array 


@given(bool_array=draw_boolean_array(array_min_side=2))
@settings(print_blob=True)
def test_combine_filter_masks_01(bool_array: np.ndarray):

    mask_list = [
        pd.Series(bool_array[:, i]) for i in range(bool_array.shape[1])]

    result = combine_filter_masks(mask_list)

    # 'combine_filter_masks' computes the mask in Pandas and uses the product, 
    #   while Numpy and summing is used here as an alternative
    #mask = pd.Series(bool_array.prod(axis=1))
    row_sums = bool_array.sum(axis=1)
    mask = row_sums == bool_array.shape[1]

    assert (result == mask).all()



##################################################
# CALCULATE AND DISPLAY TRADE STATISTICS
##################################################


# Calculate trade statistics
##################################################


def test_calculate_trade_statistics_01():

    df = pd.DataFrame()
    result = calculate_trade_statistics(df)

    assert isinstance(result, list)
    assert len(result) == 4
    for e in result:
        assert isinstance(e, statistic_description)
        assert e.description == ''
        assert np.isnan(e.statistic)


def test_calculate_trade_statistics_01b():

    df = pd.DataFrame({
        'balance_change': [],
        'balance_change_commission': []})

    result = calculate_trade_statistics(df)

    assert np.isnan(result[0].statistic)
    assert np.isnan(result[1].statistic)
    assert np.isnan(result[2].statistic)
    assert np.isnan(result[3].statistic)


def test_calculate_trade_statistics_02():

    df = pd.DataFrame({
        'balance_change': [-25, 25],
        'balance_change_commission': [-30, 20]})

    result = calculate_trade_statistics(df)

    assert result[0].statistic == 25/25
    assert result[1].statistic == 25/25
    assert result[2].statistic == round(20/30, 2)
    assert result[3].statistic == round(20/30, 2)


def test_calculate_trade_statistics_03():

    df = pd.DataFrame({
        'balance_change': [-120, 120, 0],
        'balance_change_commission': [-140, 100, -10]})

    result = calculate_trade_statistics(df)

    assert result[0].statistic == 120/120
    assert result[1].statistic == 120/120
    assert result[2].statistic == round(100/75, 2)
    assert result[3].statistic == round(100/150, 2)


@given(
    float_list=st.lists(st.floats(min_value=-1e12, max_value=1e12), min_size=1),
    elements_diff=st.floats(min_value=0, max_value=1e12))
@settings(print_blob=True)
def test_calculate_trade_statistics_04(
    float_list: list[float], elements_diff: float):
    """
    Calculated statistics are all ratios; ensure that they are all >= zero
    """

    float_array01 = np.array(float_list)
    float_array02 = float_array01 - elements_diff

    df = pd.DataFrame({
        'balance_change': float_array01,
        'balance_change_commission': float_array02})

    result = calculate_trade_statistics(df)

    # all ratios should be at least zero
    if not np.isnan(result[0].statistic):
        assert result[0].statistic >= 0
    if not np.isnan(result[1].statistic):
        assert result[1].statistic >= 0
    if not np.isnan(result[2].statistic):
        assert result[2].statistic >= 0
    if not np.isnan(result[3].statistic):
        assert result[3].statistic >= 0


# Profit table
##################################################


def test_profit_table_01():
    """
    Test that values in JSON input are maintained in plotting object
    """

    json_input = (
        '{"columns":["balance_change", "balance_change_commission"],'
        '"index":[0, 1, 2],'
        '"data":[[-120, -140], [120, 100], [0, -10]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = profit_table(json_input, table_mask)
    result_dict = result.to_dict()
    result_values = result_dict['data'][0]['cells']['values']

    assert result_values[1][0] == 120/120
    assert result_values[1][1] == round(100/75, 2)
    assert result_values[2][0] == 120/120
    assert result_values[2][1] == round(100/150, 2)



##################################################
# PLOTS OF TRADE DATA
##################################################


def test_number_of_positions_by_date_plot():
    """
    Test that values in JSON input are maintained in plotting object
    """

    json_input = (
        '{"columns":["max_date"],'
        '"index":[0, 1, 2],'
        '"data":[[1600000000000], [1610000000000], [1620000000000]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = number_of_positions_by_date_plot(json_input, table_mask)
    result_dict = result.to_dict()
    result_values = result_dict['data'][0]['x']

    assert result_values[0] == date(2020, 9, 13)
    assert result_values[1] == date(2021, 1,  7)
    assert result_values[2] == date(2021, 5,  3)


def test_spent_outflow_by_date_plot():
    """
    Test that values in JSON input are maintained in plotting object
    """

    json_input = (
        '{"columns":["spent_outflow", "spent_outflow_commission", "max_date"],'
        '"index":[0, 1, 2],'
        '"data":[[-20000, -20100, 1600000000000], [-100, -100, 1610000000000], '
            '[-3000, -3020, 1620000000000]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = spent_outflow_by_date_plot(json_input, table_mask)
    result_dict = result.to_dict()
    result_values = result_dict['data']

    assert result_values[0]['x'][0] == date(2020, 9, 13)
    assert result_values[0]['x'][1] == date(2021, 1,  7)
    assert result_values[0]['x'][2] == date(2021, 5,  3)

    assert result_values[1]['x'][0] == date(2020, 9, 13)
    assert result_values[1]['x'][1] == date(2021, 1,  7)
    assert result_values[1]['x'][2] == date(2021, 5,  3)

    assert bdata_convert(result_values[0]['y'])[0] == 20_000
    assert bdata_convert(result_values[0]['y'])[1] == 100
    assert bdata_convert(result_values[0]['y'])[2] == 3000

    assert bdata_convert(result_values[1]['y'])[0] == 100
    assert bdata_convert(result_values[1]['y'])[1] == 0
    assert bdata_convert(result_values[1]['y'])[2] == 20


def test_number_of_positions_by_gain_loss_plot():
    """
    Test that values in JSON input are maintained in plotting object
    """

    json_input = (
        '{"columns":["symbol", "max_date", "balance_change", '
            '"balance_change_commission"],'
        '"index":[0, 1, 2, 3],'
        '"data":[["aaa", 1600000000000, 8000, 7900], '
                '["bbb", 1610000000000, -100, -110], '
                '["ccc", 1620000000000, -3000, -3020], '
                '["ddd", 1630000000000, 0, 0]]}')

    table_mask = '{"0":true,"1":true,"2":true,"3":true}'

    result = number_of_positions_by_gain_loss_plot(json_input, table_mask)
    result_dict = result.to_dict()
    result_values = result_dict['data']

    # 'number_of_positions_by_gain_loss_plot' calculates the percentages of the
    #   rows/transactions that were losses, gains, or neither, so these 
    #   percentages are in the plotting object, rather than the raw 'data' from
    #   'json_input' above
    assert result_values[0]['name']    == 'Loss'
    assert result_values[0]['text'][0] == '50%'
    assert result_values[0]['text'][1] == '50%'

    assert result_values[1]['name']    == 'No Change'
    assert result_values[1]['text'][0] == '25%'
    assert result_values[1]['text'][1] == '25%'

    assert result_values[2]['name']    == 'Gain'
    assert result_values[2]['text'][0] == '25%'
    assert result_values[2]['text'][1] == '25%'



##################################################
# PLOTS OF TRADE DATA THAT ARE GROUPED/FILTERED BY GAINS/LOSSES
##################################################
# In some cases, the user may want to view data either filtered or grouped by
#   whether the transactions gained or lost money (or a certain amount of money)
# This section includes the dataclasses and functions to calculate these 
#   gain/loss groups, as well as the plots that use them
##################################################


# Create masks for gains and losses
##################################################


def test_get_balance_change_masks_01():

    colnames = ('balance_change', 'balance_change_commission')
    assert isinstance(colnames, tuple)

    df1 = pd.DataFrame(
        ((1000,  990),
         (-500, -500),
         (   5,   -5),
         (   0,    0)), columns=colnames)

    result = get_balance_change_masks(df1)

    colnames = ('gain', 'loss', 'gain_commission', 'loss_commission')
    assert isinstance(colnames, tuple)

    df2 = pd.DataFrame(
        (( True,  False,  True, False),
         (False,   True, False,  True),
         ( True,  False, False,  True),
         (False,  False, False, False)), columns=colnames)

    assert (result == df2).all().all()


@given(df_idx_sample=draw_dataframe_with_column_idxs(array_min_side=2))
@settings(print_blob=True)
def test_get_balance_change_masks_02(
    df_idx_sample: tuple[pd.DataFrame, tuple[int, int], list[int]]):
    """
    Tests that marginal sums for rows and columns of Boolean dataframe are 
        correct
    Uses integers instead of floats, as in the tested function, so the
        'zero_error_threshold' parameter is not tested
    """

    df = df_idx_sample[0]
    idx = df_idx_sample[1]
    colnames = df.columns.tolist()
    colnames[idx[0]] = 'balance_change'
    colnames[idx[1]] = 'balance_change_commission'
    df.columns = colnames

    result = get_balance_change_masks(df)

    over_zero_mask = df.iloc[:, list(idx)] > 0
    col_sums = over_zero_mask.sum() 
    row_sums = over_zero_mask.sum(axis=1) 

    assert (col_sums.values == result.iloc[:, [0, 2]].sum().values).all()
    assert (row_sums == result.iloc[:, [0, 2]].sum(axis=1)).all()

    under_zero_mask = df.iloc[:, list(idx)] < 0
    col_sums = under_zero_mask.sum() 
    row_sums = under_zero_mask.sum(axis=1) 

    assert (col_sums.values == result.iloc[:, [1, 3]].sum().values).all()
    assert (row_sums == result.iloc[:, [1, 3]].sum(axis=1)).all()


# 
##################################################


def test_balance_change_by_position_chronologically():
    """
    Test that values in JSON input are maintained in plotting object
    """

    json_input = (
        '{"columns":["balance_change", "balance_change_commission", "max_date", '
            '"symbol"],'
        '"index":[0, 1, 2],'
        '"data":['
            '[-120, -140, 1600000000000, "aaa"], '
            '[120, 100, 1610000000000, "bbb"], '
            '[0, -10, 1620000000000, "ccc"]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = balance_change_by_position_chronologically(json_input, table_mask)
    result_dict = result.to_dict()
    result_data = result_dict['data']
    result_layout = result_dict['layout']

    assert (bdata_convert(result_data[0]['y']) == [-120, 120,   0]).all()
    assert (bdata_convert(result_data[1]['y']) == [-140, 100, -10]).all()
    assert (bdata_convert(result_data[2]['y']) == [-120, 120,   0]).all()
    assert (bdata_convert(result_data[5]['y']) == [-140, 100, -10]).all()
    assert result_layout['yaxis']['range'][0] == -140

    assert (
        result_dict['layout']['xaxis']['ticktext'] == 
        ['2020-09-13 12:26:40 AAA', 
         '2021-01-07 06:13:20 BBB', 
         '2021-05-03 00:00:00 CCC']).all()


def test_cumulative_balance_change_by_position_chronologically():
    """
    Test that values in JSON input are maintained in plotting object
    """

    json_input = (
        '{"columns":["balance_change", "balance_change_commission", "max_date", '
            '"symbol"],'
        '"index":[0, 1, 2],'
        '"data":['
            '[-120, -140, 1600000000000, "aaa"], '
            '[120, 100, 1610000000000, "bbb"], '
            '[0, -10, 1620000000000, "ccc"]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = cumulative_balance_change_by_position_chronologically(
        json_input, table_mask)
    result_dict = result.to_dict()
    result_data = result_dict['data']

    assert (bdata_convert(result_data[0]['y']) == [0, -120,   0,   0]).all()
    assert (bdata_convert(result_data[1]['y']) == [0, -140, -40, -50]).all()
    assert (bdata_convert(result_data[2]['y']) == [0,   20,  40,  50]).all()
    assert (result_data[3]['y'] == [120,  100, 120, 150])

    assert (
        result_dict['layout']['xaxis']['ticktext'] == 
        ['2020-09-13 12:26:40 START AT ZERO', 
         '2020-09-13 12:26:40 AAA', 
         '2021-01-07 06:13:20 BBB', 
         '2021-05-03 00:00:00 CCC']).all()


def test_price_change_per_share_by_position_chronologically():
    """
    Test that values in JSON input are maintained in plotting object
    """

    json_input = (
        '{"columns":["balance_change", "balance_change_commission", "max_date", '
            '"symbol", "fill_price_change"],'
        '"index":[0, 1, 2],'
        '"data":['
            '[-120, -140, 1600000000000, "aaa", -0.50], '
            '[120, 100, 1610000000000, "bbb", 10], '
            '[0, -10, 1620000000000, "ccc", 0]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = price_change_per_share_by_position_chronologically(
        json_input, table_mask)
    result_dict = result.to_dict()
    result_data = result_dict['data']

    assert (bdata_convert(result_data[0]['y']) == [-0.5, 10, 0]).all()
    assert (bdata_convert(result_data[1]['y']) == [-0.5, 10, 0]).all()


def test_calculate_geometric_mean_01():

    series = pd.Series([2, 18])
    result = calculate_geometric_mean(series)

    assert np.isclose(result, 6, atol=1e-02)


def test_calculate_geometric_mean_02():

    series = pd.Series([3, 5, 12])
    result = calculate_geometric_mean(series)

    assert np.isclose(result, 5.65, atol=1e-02)


def test_calculate_geometric_mean_03():

    series = pd.Series([4, 10, 16, 24])
    result = calculate_geometric_mean(series)

    assert np.isclose(result, 11.13, atol=1e-02)


def test_calculate_geometric_mean_04():

    series = pd.Series([1, 3, 9, 27, 81])
    result = calculate_geometric_mean(series)

    assert np.isclose(result, 9, atol=1e-04)


def test_calculate_geometric_mean_05():

    series = pd.Series([1/2, 1/3])
    result = calculate_geometric_mean(series)

    assert np.isclose(result, np.sqrt(1/6), atol=1e-04)


def test_calculate_geometric_mean_06():

    series = pd.Series([1/2, 2/3, 3/4])
    result = calculate_geometric_mean(series)

    assert np.isclose(result, np.power(1/4, 1/3), atol=1e-04)


def test_calculate_geometric_mean_percent_01():

    series = pd.Series([1/2, 1/3])
    result = calculate_geometric_mean_percent(series)

    assert np.isclose(result, (np.sqrt(1/6) - 1) * 100, atol=1e-04)


def test_calculate_geometric_mean_percent_02():

    series = pd.Series([1/2, 2/3, 3/4])
    result = calculate_geometric_mean_percent(series)

    assert np.isclose(result, (np.power(1/4, 1/3) - 1) * 100, atol=1e-04)


def test_price_percentage_change_by_position_chronologically_01():
    """
    Test that values in JSON input are maintained in plotting object

    Test with the first set of column names
    """

    json_input = (
        '{"columns":["fill_price_buy", "fill_price_sell" , '
            '"fill_price_change", "max_date", "symbol", "balance_change", '
            '"balance_change_commission"],'
        '"index":[0, 1, 2],'
        '"data":['
            '[8.15, 8.25, 0.1, 1600000000000, "aaa", 100, 90], '
            '[23, 22, -1, 1610000000000, "bbb", -1000, -1010], '
            '[60, 10, -50, 1620000000000, "ccc", -500, -503]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = price_percentage_change_by_position_chronologically(
        json_input, table_mask)
    result_dict = result.to_dict()
    result_data = result_dict['data']

    assert np.isclose(
        result_data[0]['y'], 
        [(0.1/8.15) * 100, 
         (-1/23) * 100, 
         (-50/60) * 100], atol=1e-02).all()

    assert np.isclose(
        result_data[1]['y'], 
        [(0.1/8.15) * 100, 
         (-1/23) * 100, 
         (-50/60) * 100], atol=1e-02).all()

    assert np.isclose(result_data[2]['y'], [(0.1/8.15) * 100], atol=1e-02).all()


def test_price_percentage_change_by_position_chronologically_02():
    """
    Test that values in JSON input are maintained in plotting object

    Test with the second set of column names
    """

    json_input = (
        '{"columns":["buy_mean_fill_price", "sell_mean_fill_price" , '
            '"fill_price_change", "max_date", "symbol", "balance_change", '
            '"balance_change_commission"],'
        '"index":[0, 1, 2],'
        '"data":['
            '[8.15, 8.25, 0.1, 1600000000000, "aaa", 100, 90], '
            '[23, 22, -1, 1610000000000, "bbb", -1000, -1010], '
            '[60, 10, -50, 1620000000000, "ccc", -500, -503]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = price_percentage_change_by_position_chronologically(
        json_input, table_mask)
    result_dict = result.to_dict()
    result_data = result_dict['data']

    assert np.isclose(
        result_data[0]['y'], 
        [(0.1/8.15) * 100, 
         (-1/23) * 100, 
         (-50/60) * 100], atol=1e-02).all()

    assert np.isclose(
        result_data[1]['y'], 
        [(0.1/8.15) * 100, 
         (-1/23) * 100, 
         (-50/60) * 100], atol=1e-02).all()

    assert np.isclose(result_data[2]['y'], [(0.1/8.15) * 100], atol=1e-02).all()


def test_position_hold_times_01():
    """
    Test that values in JSON input are maintained in plotting object
    """

    # 'order_buy' and 'order_sell' are not recognized columns, so plotting
    #   function should return 'no data' message
    json_input = (
        '{"columns":["balance_change", "balance_change_commission", '
            '"order_buy", "order_sell"],'
        '"index":[0, 1, 2],'
        '"data":['
            '[-120, -140, 1600000000000, 1600000010000], '
            '[120, 100, 1610000000000, 1610000100000], '
            '[0, -10, 1620000000000, 1620001000000]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = position_hold_times(json_input, table_mask)
    result_dict = result.to_dict()
    assert 'no data' in result_dict['layout']['title']['text'].lower()



def test_position_hold_times_02():
    """
    Test that values in JSON input are maintained in plotting object
    """

    json_input = (
        '{"columns":["balance_change", "balance_change_commission", '
            '"order_submit_time_buy", "order_fill_cancel_time_sell"],'
        '"index":[0, 1, 2],'
        '"data":['
            '[-120, -140, 1600000000000, 1600000010000], '
            '[120, 100, 1610000000000, 1610000100000], '
            '[0, -10, 1620000000000, 1620001000000]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = position_hold_times(json_input, table_mask)
    result_dict = result.to_dict()
    result_data = result_dict['data']

    assert result_data[0]['name'] == 'All<br>0 days 00:06:10'
    assert result_data[1]['name'] == 'Gains, w/o Commissions<br>0 days 00:01:40'
    assert result_data[2]['name'] == 'Losses, w/o Commissions<br>0 days 00:00:10' 
    assert result_data[3]['name'] == 'Gains, w/ Commissions<br>0 days 00:01:40'
    assert result_data[4]['name'] == 'Losses, w/ Commissions<br>0 days 00:08:25'


def test_position_hold_times_03():
    """
    Test that values in JSON input are maintained in plotting object
    """

    json_input = (
        '{"columns":["balance_change", "balance_change_commission", '
            '"order_submit_time", "order_fill_cancel_time"],'
        '"index":[0, 1, 2],'
        '"data":['
            '[-120, -140, 1600000000000, 1600000010000], '
            '[120, 100, 1610000000000, 1610000100000], '
            '[0, -10, 1620000000000, 1620001000000]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = position_hold_times(json_input, table_mask)
    result_dict = result.to_dict()
    result_data = result_dict['data']

    assert result_data[0]['name'] == 'All<br>0 days 00:06:10'
    assert result_data[1]['name'] == 'Gains, w/o Commissions<br>0 days 00:01:40'
    assert result_data[2]['name'] == 'Losses, w/o Commissions<br>0 days 00:00:10' 
    assert result_data[3]['name'] == 'Gains, w/ Commissions<br>0 days 00:01:40'
    assert result_data[4]['name'] == 'Losses, w/ Commissions<br>0 days 00:08:25'


def test_position_volumes_01():
    """
    Test that values in JSON input are maintained in plotting object
    """

    # 'match' is not a recognized column, so plotting function should return 'no 
    #   data' message
    json_input = (
        '{"columns":["balance_change", "balance_change_commission", "match"],'
        '"index":[0, 1, 2],'
        '"data":['
            '[-120, -140, 100], '
            '[120, 100, 2000], '
            '[0, -10, 50]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = position_volumes(json_input, table_mask)
    result_dict = result.to_dict()
    assert 'no data' in result_dict['layout']['title']['text'].lower()


def test_position_volumes_02():
    """
    Test that values in JSON input are maintained in plotting object
    """

    json_input = (
        '{"columns":["balance_change", "balance_change_commission", '
            '"match_shares_num_fill"],'
        '"index":[0, 1, 2],'
        '"data":['
            '[-120, -140, 100], '
            '[120, 100, 2000], '
            '[0, -10, 50]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = position_volumes(json_input, table_mask)
    result_dict = result.to_dict()
    result_data = result_dict['data']

    assert (result_data[0]['y'] == [100, 2000, 50]).all()
    assert (result_data[1]['y'] == [2000]).all()
    assert (result_data[2]['y'] == [100]).all()
    assert (result_data[3]['y'] == [2000]).all()
    assert (result_data[4]['y'] == [100, 50]).all()


def test_position_volumes_03():
    """
    Test that values in JSON input are maintained in plotting object
    """

    json_input = (
        '{"columns":["balance_change", "balance_change_commission", '
            '"shares_num_fill"],'
        '"index":[0, 1, 2],'
        '"data":['
            '[-120, -140, 100], '
            '[120, 100, 2000], '
            '[0, -10, 50]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = position_volumes(json_input, table_mask)
    result_dict = result.to_dict()
    result_data = result_dict['data']

    assert (result_data[0]['y'] == [100, 2000, 50]).all()
    assert (result_data[1]['y'] == [2000]).all()
    assert (result_data[2]['y'] == [100]).all()
    assert (result_data[3]['y'] == [2000]).all()
    assert (result_data[4]['y'] == [100, 50]).all()


def test_position_commissions():
    """
    Test that values in JSON input are maintained in plotting object
    """

    json_input = (
        '{"columns":["balance_change", "balance_change_commission"],'
        '"index":[0, 1, 2],'
        '"data":['
            '[-120, -140], '
            '[120, 100], '
            '[0, -10]]}')

    table_mask = '{"0":true,"1":true,"2":true}'

    result = position_commissions(json_input, table_mask)
    result_dict = result.to_dict()
    result_data = result_dict['data']

    assert (result_data[0]['y'] == [20, 20, 10]).all()
    assert (result_data[1]['y'] == [20]).all()
    assert (result_data[2]['y'] == [20]).all()
    assert (result_data[3]['y'] == [20]).all()
    assert (result_data[4]['y'] == [20, 10]).all()


























