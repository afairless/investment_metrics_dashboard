#! usr/bin/env python3

import json
import numpy as np
import pandas as pd
from pathlib import Path

from typing import Sequence

#import psycopg2
from sqlalchemy import create_engine

import pyarrow as pa
import pyarrow.parquet as pq

import pandera.pandas as pdr
from pandera.typing import DataFrame as pdr_DataFrame

from ..schemas.data_pipeline_schemas import (
    FilledOrdersForDashboard,
    FilledOrdersByPositionChange,
    FilledOrdersByOngoingPosition,
    FilledOrdersBySymbolDay,
    PriceWeightedByShareNumberResult,
    )


# 
##############################

def read_text_file(text_filename, return_string=False, keep_newlines=False):
    """
    Reads text file
    If 'return_string' is 'True', returns text in file as a single string
    If 'return_string' is 'False', returns list so that each line of text in
        file is a separate list item
    If 'keep_newlines' is 'True', newline markers '\n' are retained; otherwise,
        they are deleted

    :param text_filename: string specifying filepath or filename of text file
    :param return_string: Boolean indicating whether contents of text file
        should be returned as a single string or as a list of strings
    :return:
    """

    text_list = []

    try:
        with open(text_filename) as text:
            if return_string:
                # read entire text file as single string
                if keep_newlines:
                    text_list = text.read()
                else:
                    text_list = text.read().replace('\n', '')
            else:
                # read each line of text file as separate item in a list
                for line in text:
                    if keep_newlines:
                        text_list.append(line)
                    else:
                        text_list.append(line.rstrip('\n'))
            text.close()

        return text_list

    except:

        return ['There was an error when trying to read text file']


def write_list_to_text_file(a_list: list, text_filepath: Path, overwrite=False):
    """
    Writes a list of strings to a text file

    If 'overwrite' is 'True', any existing file at the path of 'text_filepath'
        will be overwritten
    If 'overwrite' is 'False', list of strings will be appended to any existing
        file at the path of 'text_filepath'
    """

    if overwrite:
        append_or_overwrite = 'w'
    else:
        append_or_overwrite = 'a'

    with open(text_filepath, append_or_overwrite, encoding='utf-8') as text_file:
        for e in a_list:
            text_file.write(str(e))
            text_file.write('\n')


# SET INPUT AND OUTPUT DIRECTORY PATHS
##############################

def get_filled_orders_input_filepath() -> Path:
    """
    Returns path to directory where input table of filled orders is stored
    """

    project_parent_directory = Path.cwd().parent.parent.parent
    input_path = (
        project_parent_directory / 'trades_plots_data' / 
        'filled_orders' / 'filled_orders.parquet')

    return input_path   


def get_filled_orders_by_position_change_output_path() -> Path:
    """
    Creates and returns path to directory where output will be saved
    """

    project_parent_directory = Path.cwd().parent.parent.parent
    output_path = (
        project_parent_directory / 'trades_plots_data' / 
        'filled_orders_by_position_change')
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path  


def get_filled_orders_by_ongoing_position_output_path() -> Path:
    """
    Creates and returns path to directory where output will be saved
    """

    project_parent_directory = Path.cwd().parent.parent.parent
    output_path = (
        project_parent_directory / 'trades_plots_data' / 
        'filled_orders_by_ongoing_position')
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path  


def get_filled_orders_by_symbol_day_output_path() -> Path:
    """
    Creates and returns path to directory where output will be saved
    """

    project_parent_directory = Path.cwd().parent.parent.parent
    output_path = (
        project_parent_directory / 'trades_plots_data' / 
        'filled_orders_by_symbol_day')
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path  


# LOAD, TRANSFORM, AND QUALITY-CHECK DATA
##############################

@pdr.check_output(FilledOrdersForDashboard.to_schema(), lazy=True)
def load_filled_orders() -> pdr_DataFrame[FilledOrdersForDashboard]:

    input_filepath = get_filled_orders_input_filepath()
    filled_orders = pq.read_table(input_filepath).to_pandas()

    return filled_orders 


def identify_stocks_or_options(symbol: pd.Series) -> np.ndarray:
    """
    Identifies whether a financial instrument is a stock or an option

    Returns Boolean Pandas Series, where 'True' denotes 'stock' and 'False'
        denotes 'option'
    """

    # stock/equity symbols have fewer characters than the threshold
    # options symbols have at least as many characters as the threshold
    symbol_length_threshold = 7

    stock_or_option_bool = symbol.str.len() < symbol_length_threshold

    return stock_or_option_bool


def match_cumulative_sums_by_row(
    series1: Sequence[float], series2: Sequence[float]) -> pd.DataFrame:
    """
    Given two sequences of numbers with the same sum, treat the sequences as if
        numbers from one sequence (the 'donor sequence') are being allocated to 
        the other sequence (the 'acceptor sequence') starting with the first 
        numbers in each sequence 
    The quantities in the two sequences need to be matched to each other in 
        order, so that the amount that each donor number contributes to each
        acceptor number is specified
    If each sequence is thought of as a column, each quantity match is specified 
        on its own row; thus, numbers/rows may need to be added to either or 
        both sequences so that the two sequences have the same length

    NOTE:  Conceptually, either sequence can be designated as the 'donor 
        sequence' or the 'acceptor sequence'; either way produces the same 
        results

    Example 1:

        Input:

            Row_Index   Sequence_1  Sequence_2
            1           100         50
            2                       50

        Output:

            Sequence_1_Row_Index   Sequence_1  Sequence_2   Sequence_2_Row_Index
            1                      100         50           1
            1                       50         50           2

        Comment:  
            Sequence_1 may be seen as donating 50 of its 100 units to the
                50-unit capacity of Sequence_2's first row and another 50 units 
                to the 50-unit capacity of Sequence_2's second row 
            Conversely, Sequence_2 may be seen as donating its first row's 50 
                units to Sequence_1's 100-unit capacity in its first row; with 
                those 50 units absorbed by Sequence_1, Sequence_2 then donates
                its second row of 50 units to the remaining 50 units of 
                Sequence_1's first-row capacity 


    Example 2:

        Input:

            Row_Index   Sequence_1  Sequence_2
            1            80         150
            2           120          50

        Output:

            Sequence_1_Row_Index   Sequence_1  Sequence_2   Sequence_2_Row_Index
            1                       80         150          1
            2                      120          70          1
            2                       50          50          2

        Comment:  
            Sequence_1's first row donates its 80 units to Sequence_2's 150-unit
                first-row capacity, leaving 70 units of capacity
            Sequence_1's second row then donates 70 units to Sequence_2's first
                row, which leaves Sequence_1's second row with 50 units
            Sequence_1's remaining 50 units are donated to Sequence_2's 50-unit
                capacity of its second row
    """

    assert sum(series1) == sum(series2)

    idx1 = 0
    idx2 = 0
    n1 = series1[idx1]
    n2 = series2[idx2]
    row_pairs = [(n1, n2, abs(n1 - n2), idx1, idx2)]

    while idx1 < len(series1)-1 or idx2 < len(series2)-1:

        n_diff = row_pairs[-1][2]

        if row_pairs[-1][0] > row_pairs[-1][1]:
            idx2 += 1
            n2 = series2[idx2]
            row_pairs.append((n_diff, n2, abs(n_diff - n2), idx1, idx2))

        elif row_pairs[-1][0] < row_pairs[-1][1]:
            idx1 += 1
            n1 = series1[idx1]
            row_pairs.append((n1, n_diff, abs(n1 - n_diff), idx1, idx2))

        else:
            idx1 += 1
            idx2 += 1
            n1 = series1[idx1]
            n2 = series2[idx2]
            row_pairs.append((n1, n2, abs(n1 - n2), idx1, idx2))

    colnames = ['series1', 'series2', 'diff', 'series1_idx', 'series2_idx']
    return pd.DataFrame(row_pairs, columns=colnames)


def match_buy_sell_orders(position_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match buy and sell orders by 'first-bought, first-sold' criterion
    In other words, the first share of a particular stock/equity that is 
        bought is matched to the first share of that stock/equity that is sold;
        the second bought share is matched to the second sold share, and so on
    With this criterion, we can determine the buy and sell prices -- and other
        data -- for each and every share
    These data for bought and sold shares are matched and shown on the same row
        in the resulting table
    Instead of having a row for each and every share (first share bought/sold,
        second share bought/sold, etc.), consecutive shares that are matched to
        the same data are aggregated onto a single row
    """

    # match buy and sell orders by 'first-bought, first-sold' criterion
    # matching row indices are provided as 'match_idx_buy' and 'match_idx_sell'
    ##############################

    series1 = position_df.loc[
        position_df['buy_or_sell'].str.contains('buy'), 'shares_num_fill']
    series2 = position_df.loc[
        position_df['buy_or_sell'].str.contains('sell'), 'shares_num_fill']

    buy_sell_share_matches = match_cumulative_sums_by_row(
        series1.to_list(), series2.to_list())

    buy_sell_share_matches.columns = [
        'match_shares_num_fill_buy', 'match_shares_num_fill_sell', 
        'match_shares_num_fill_diff', 'match_idx_buy', 'match_idx_sell']

    buy_sell_share_matches['match_shares_num_fill'] = (
        buy_sell_share_matches.iloc[:, :2].min(axis=1))

    buy_idx = series1.index[buy_sell_share_matches['match_idx_buy']]
    sell_idx = series2.index[buy_sell_share_matches['match_idx_sell']]


    # get columns that are the same for buy and sell orders
    ##############################

    colnames = ['positions_idx', 'symbol', 'simulator_or_real']
    common_df = position_df.loc[buy_idx, colnames]


    # get columns that are different for buy and sell orders
    ##############################

    buy_sell_colnames = [
        'order_submit_time', 'order_fill_cancel_time', 'shares_num_submit', 
        'limit_price', 'fill_price', 'order_status', 'order_route', 
        'order_duration', 'commission_cost', 'tags', 'notes']

    buy_df = position_df.loc[buy_idx, buy_sell_colnames]
    sell_df = position_df.loc[sell_idx, buy_sell_colnames]
    assert len(buy_df) == len(sell_df)

    buy_df.columns = [e + '_buy' for e in buy_sell_colnames]
    sell_df.columns = [e + '_sell' for e in buy_sell_colnames]


    # compile data from buy and sell orders
    ##############################

    common_df.index = range(len(common_df))
    buy_df.index = range(len(buy_df))
    sell_df.index = range(len(sell_df))

    df = pd.concat([common_df, buy_sell_share_matches, buy_df, sell_df], axis=1)

    return df


def identify_bull_bear_positions(df: pd.DataFrame) -> np.ndarray:
    """
    Classifies each position (i.e., each row in the dataframe) as either bullish 
        or bearish

    Bullish positions:
        1) Long on a stock/equity
        2) Long on a call option
        3) Short on a put option

    Bearish positions:
        1) Short on a stock/equity
        2) Short on a call option
        3) Long on a put option
    """

    stock_long_bool = (
        (df['stock_or_option'] == 'stock') & (df['long_or_short'] == 'long'))
    stock_short_bool = (
        (df['stock_or_option'] == 'stock') & (df['long_or_short'] == 'short'))

    # the 'c' in the regular expression identifies a call; the 'p' labels a put
    call_bool = (
        df['symbol'].str.contains('[a-z]{1,7} [0-9]{6}c[0-9]{1,5}', regex=True))
    put_bool = (
        df['symbol'].str.contains('[a-z]{1,7} [0-9]{6}p[0-9]{1,5}', regex=True))


    # calls are a subset of options, so a boolean conjunction of calls and 
    #   options should be the same as the calls alone
    assert ( 
        (call_bool & (df['stock_or_option'] == 'option')) == call_bool ).all()
    # puts are a subset of options, so a boolean conjunction of puts and 
    #   options should be the same as the puts alone

    assert ( 
        (put_bool & (df['stock_or_option'] == 'option')) == put_bool ).all()

    # the number of calls and puts together should equal the number of options
    assert (
        call_bool.sum() + put_bool.sum() == 
        (df['stock_or_option'] == 'option').sum())

    # bullish positions
    bull_bool = (
        stock_long_bool | 
        (call_bool & (df['long_or_short'] == 'long')) |
        (put_bool & (df['long_or_short'] == 'short')))

    # bearish positions
    bear_bool = (
        stock_short_bool | 
        (call_bool & (df['long_or_short'] == 'short')) |
        (put_bool & (df['long_or_short'] == 'long')))

    # ensure that every position (row in the table) is classified as either 
    #   bullish and bearish
    assert (bull_bool.sum() + bear_bool.sum()) == len(df)

    bull_bear_labels = np.where(bull_bool, 'bullish', 'bearish')

    return bull_bear_labels 


@pdr.check_io(
    filled_orders=FilledOrdersForDashboard.to_schema(), 
    out=FilledOrdersByPositionChange.to_schema(), 
    lazy=True)
def convert_to_filled_orders_by_position_change(
    filled_orders: pd.DataFrame) -> pdr_DataFrame[FilledOrdersByPositionChange]:
    """
    Group filled orders by position change, i.e., a trader holds a single 
        "position" for as long as the trader is holding the same number of 
        shares in a particular equity/stock; if the trader buys or sells any
        shares, the trader has started a new position, which appears on a 
        separate row of the table
    """

    df = (
        filled_orders.groupby('positions_idx')
        .apply(match_buy_sell_orders, include_groups=True)
        .reset_index(drop=True))

    assert isinstance(df, pd.DataFrame)

    df['match_shares_num_fill_buy'] = df['match_shares_num_fill_buy'].astype(int)
    df['match_shares_num_fill_sell'] = df['match_shares_num_fill_sell'].astype(int)
    df['match_shares_num_fill_diff'] = df['match_shares_num_fill_diff'].astype(int)
    df['match_shares_num_fill'] = df['match_shares_num_fill'].astype(int)

    # stock/equity symbols have fewer characters than the threshold
    # options symbols have at least as many characters as the threshold
    stock_or_option_bool = identify_stocks_or_options(df['symbol'])
    df['stock_or_option'] = np.where(stock_or_option_bool, 'stock', 'option')

    order_time_mask = (
        df['order_fill_cancel_time_buy'] < df['order_fill_cancel_time_sell'])
    df['long_or_short'] = np.where(order_time_mask, 'long', 'short')

    df['bull_or_bear'] = identify_bull_bear_positions(df)

    df['fill_price_change'] = (
        df['fill_price_sell'] - df['fill_price_buy']).round(3)
    df['balance_change'] = (
        df['fill_price_change'] * df['match_shares_num_fill'])
    df['balance_change_commission'] = (
        df['balance_change'] - 
        df['commission_cost_buy'] - 
        df['commission_cost_sell'])

    df['spent_outflow'] = (
        -1 * df['fill_price_buy'] * df['match_shares_num_fill'])
    df['spent_outflow_commission'] = (
        df['spent_outflow'] - 
        df['commission_cost_buy'] - 
        df['commission_cost_sell'])

    return df


def list_of_strings_without_nans(x: Sequence) -> list:
    return [e for e in x if isinstance(e, str)]


@pdr.check_io(
    #filled_orders=FilledOrders.to_schema(), 
    out=PriceWeightedByShareNumberResult.to_schema(), 
    lazy=True)
def calculate_price_weighted_by_shares_number(
    filled_orders: pd.DataFrame, mask: pd.Series
    ) -> pdr_DataFrame[PriceWeightedByShareNumberResult]:
    """
    Calculate the price averaged by the number of shares in the rows of the
        dataframe 'filled_orders' specified by the Boolean 'mask'
    """

    colnames = ['positions_idx', 'shares_num_fill', 'fill_price']
    df = filled_orders.loc[mask, colnames].copy()

    # divide all share numbers within a position (as identified by 
    #   'positions_idx') by the smallest number of shares within that position
    #   to create ratios/weights of shares
    df['shares_min'] = (
        df.groupby('positions_idx')['shares_num_fill'].transform('min'))
    df['shares_weight'] = (df['shares_num_fill'] / df['shares_min'])

    # weight the price for each transaction within the position
    df['weighted_price'] = (df['fill_price'] * df['shares_weight'])

    # calculate a single weighted average price for the position
    colnames02 = ['positions_idx', 'shares_weight', 'weighted_price']
    grouped_df = df[colnames02].groupby('positions_idx').sum().reset_index()
    grouped_df['weighted_average_price'] = (
        grouped_df['weighted_price'] / grouped_df['shares_weight'])

    result_df = grouped_df[['positions_idx', 'weighted_average_price']]

    return result_df 


def combine_list_of_strings_into_one_string(x: Sequence) -> str:
    list_of_strings = [e for e in x if isinstance(e, str)]
    return ' '.join(list_of_strings)


@pdr.check_io(
    filled_orders=FilledOrdersForDashboard.to_schema(), 
    out=FilledOrdersByOngoingPosition.to_schema(), 
    lazy=True)
def convert_to_filled_orders_by_ongoing_position(
    filled_orders: pd.DataFrame, 
    ) -> pdr_DataFrame[FilledOrdersByOngoingPosition]:
    """
    Group filled orders by position, i.e., all orders that occur on an equity/
        stock while the trader holds at least one share are all part of a 
        single "position"
    In other words, from the time when a trader buys the first shares in a
        particular equity/stock, the trader may continue buying and selling
        shares and all of those transactions are part of the same position
        until the trader has sold all shares in that equity/stock
    """


    # after summing during aggregation below, these columns will show the total 
    #   amount of money (or buying power) expended on a position
    ##################################################

    filled_orders['spent_outflow'] = np.minimum(
        filled_orders['balance_change'], 0)
    filled_orders['spent_outflow_commission'] = np.minimum(
        filled_orders['balance_change_commission'], 0)


    # aggregate by position
    ##################################################

    agg_dict = {
        'order_submit_time': 'min', 
        'order_fill_cancel_time': 'max', 
        'symbol': ('first', 'count'), 
        'buy_or_sell': combine_list_of_strings_into_one_string,
        'shares_num_fill': 'sum', 
        'order_status': 'unique',
        'order_route': 'unique',
        'order_duration': 'unique', 
        'simulator_or_real': 'first', 
        'commission_cost': 'sum', 
        'tags': list_of_strings_without_nans, 
        'notes': list_of_strings_without_nans, 
        'balance_change': 'sum', 
        'balance_change_commission': 'sum',
        'spent_outflow': 'sum',
        'spent_outflow_commission': 'sum'}

    colnames = list(agg_dict.keys()) + ['positions_idx']
    filled_orders_by_position = (
        filled_orders[colnames].groupby('positions_idx').agg(agg_dict)
        .reset_index())


    # remove small floating-point errors
    ##################################################

    filled_orders_by_position['balance_change'] = (
        filled_orders_by_position['balance_change'].round(8))
    filled_orders_by_position['balance_change_commission'] = (
        filled_orders_by_position['balance_change_commission'].round(8))


    # sum in 'agg_dict' counts each share twice, when bought and sold, so halve 
    #   the sum
    ##################################################

    filled_orders_by_position['shares_num_fill'] = (
        filled_orders_by_position['shares_num_fill'] / 2).astype(int)


    # rename columns
    ##################################################

    filled_orders_by_position.columns = (
        filled_orders_by_position.columns.get_level_values(0))
    colnames = filled_orders_by_position.columns.tolist()
    colnames[4] = 'num_buy_sell_orders'
    filled_orders_by_position.columns = colnames


    # label stocks and options
    ##################################################

    # stock/equity symbols have fewer characters than the threshold
    # options symbols have at least as many characters as the threshold
    stock_or_option_bool = (
        identify_stocks_or_options(filled_orders_by_position['symbol']))
    filled_orders_by_position['stock_or_option'] = (
        np.where(stock_or_option_bool, 'stock', 'option'))


    # label long and short orders
    ##################################################

    # ongoing positions labeled 'sell short' should be the same ones labeled 
    #   'buy to cover'
    sell_short_bool = (
        filled_orders_by_position.loc[:, 'buy_or_sell'].str.contains('short'))
    buy_cover_bool = (
        filled_orders_by_position.loc[:, 'buy_or_sell'].str.contains('cover'))
    assert (sell_short_bool == buy_cover_bool).all()

    filled_orders_by_position['long_or_short'] = np.where( 
        sell_short_bool, 'short', 'long')
    filled_orders_by_position = filled_orders_by_position.drop(
        'buy_or_sell', axis=1)


    # label bullish and bearish orders
    ##################################################

    filled_orders_by_position['bull_or_bear'] = (
        identify_bull_bear_positions(filled_orders_by_position))


    # add per-share price change, which is the balance change divided by the 
    #   number of shares in the ongoing position
    ##################################################

    filled_orders_by_position['fill_price_change'] = (
        filled_orders_by_position['balance_change'] / 
        filled_orders_by_position['shares_num_fill'])


    # calculate average buy and sell fill prices weighted by number of shares
    ##################################################

    buy_mask = filled_orders['buy_or_sell'].str.contains('buy')
    average_price_df = calculate_price_weighted_by_shares_number(
        filled_orders, buy_mask)
    assert (
        filled_orders_by_position['positions_idx'] == 
        average_price_df['positions_idx']).all()
    filled_orders_by_position = filled_orders_by_position.merge(
        average_price_df, on='positions_idx') 
    filled_orders_by_position = (
        filled_orders_by_position.rename( 
            {'weighted_average_price': 'buy_mean_fill_price'}, axis=1))

    sell_mask = filled_orders['buy_or_sell'].str.contains('sell')
    average_price_df = calculate_price_weighted_by_shares_number(
        filled_orders, sell_mask)
    assert (
        filled_orders_by_position['positions_idx'] == 
        average_price_df['positions_idx']).all()
    filled_orders_by_position = filled_orders_by_position.merge(
        average_price_df, on='positions_idx') 
    filled_orders_by_position.rename(
        {'weighted_average_price': 'sell_mean_fill_price'}, axis=1)
    filled_orders_by_position = (
        filled_orders_by_position.rename( 
            {'weighted_average_price': 'sell_mean_fill_price'}, axis=1))

    # 'fill_price_change' should be equal to the difference between the average
    #   fill prices of the buy and sell transactions
    assert (
        ((filled_orders_by_position['sell_mean_fill_price'] - 
          filled_orders_by_position['buy_mean_fill_price']) - 
         filled_orders_by_position['fill_price_change']) < 1e-6).all()


    # convert columns where each cell is an array or list to string so that a
    #   SQL database can accept it
    ##################################################

    for e in filled_orders_by_position.columns:
        if (isinstance(filled_orders_by_position[e].iloc[0], np.ndarray) or 
            isinstance(filled_orders_by_position[e].iloc[0], list)):
            filled_orders_by_position[e] = (
                filled_orders_by_position[e].apply(lambda x: ', '.join(x)))

    return filled_orders_by_position


@pdr.check_io(
    filled_orders_by_position=FilledOrdersByOngoingPosition.to_schema(), 
    out=FilledOrdersBySymbolDay.to_schema(), 
    lazy=True)
def convert_to_filled_orders_by_symbol_day(
    filled_orders_by_position: pd.DataFrame
    ) -> pdr_DataFrame[FilledOrdersBySymbolDay]:
    """
    Group filled orders that have already been grouped by position by date, too
    This grouping lets multiple positions that occur on the same equity/stock
        on the same day to be viewed as a single event
    The submission date of the first 'buy' transaction/order of the position is 
        used as the grouping date; the 'sell' date is not used
    Swing trades, i.e., when a position spans multiple dates, should have 
        already been grouped into a single position previously, so this grouping
        should not affect them (there are no swing trade examples in the current
        data, so this has not been tested)
    """

    filled_orders_by_position = filled_orders_by_position.copy()
    filled_orders_by_position['positions_idx'] = (
        filled_orders_by_position['positions_idx'].astype(str))

    agg_dict = {
        'positions_idx': list_of_strings_without_nans, 
        'order_submit_time': 'min', 
        'order_fill_cancel_time': 'max', 
        'num_buy_sell_orders': 'sum',
        'shares_num_fill': 'sum', 
        'order_status': 'unique',
        'order_route': 'unique',
        'order_duration': 'unique', 
        'simulator_or_real': 'unique', 
        'commission_cost': 'sum', 
        'tags': list_of_strings_without_nans, 
        'notes': list_of_strings_without_nans, 
        'stock_or_option': 'first', 
        'long_or_short': 'unique', 
        'bull_or_bear': 'first', 
        'balance_change': 'sum', 
        'balance_change_commission': 'sum',
        'spent_outflow': 'sum',
        'spent_outflow_commission': 'sum',
        'fill_price_change': 'sum',
        'buy_mean_fill_price': 'sum',
        'sell_mean_fill_price': 'sum'}

    filled_orders_by_position['submit_date'] = (
        filled_orders_by_position['order_submit_time'].dt.date)
    group_colnames =  ['symbol', 'submit_date']
    colnames = list(agg_dict.keys()) + group_colnames
    filled_orders_by_symbol_day = (
        filled_orders_by_position[colnames].groupby(group_colnames)
        .agg(agg_dict)
        .reset_index())

    #filled_orders_by_symbol_day['submit_date'] = (
    #    pd.to_datetime(filled_orders_by_symbol_day['submit_date']))
    filled_orders_by_symbol_day['submit_date'] = (
        pd.to_datetime(filled_orders_by_symbol_day['submit_date']))


    # 'fill_price_change' should be equal to the difference between the average
    #   fill prices of the buy and sell transactions
    # question to ponder:  when these 3 variables are aggregated above, should 
    #   they be summed with no weighting (as above) or by weighting by share 
    #   size?
    assert (
        ((filled_orders_by_position['sell_mean_fill_price'] - 
          filled_orders_by_position['buy_mean_fill_price']) - 
         filled_orders_by_position['fill_price_change']) < 1e-6).all()


    # convert columns where each cell is an array or list to string so that a
    #   SQL database can accept it
    for e in filled_orders_by_symbol_day.columns:
        if (isinstance(filled_orders_by_symbol_day[e].iloc[0], np.ndarray) or 
            isinstance(filled_orders_by_symbol_day[e].iloc[0], list)):
            filled_orders_by_symbol_day[e] = (
                filled_orders_by_symbol_day[e].apply(lambda x: ', '.join(x)))

    return filled_orders_by_symbol_day


def save_output_tables(
    output_table: pd.DataFrame, output_path: Path, output_name: str, 
    csv_separator: str):

    import time

    output_filepath = output_path / (output_name + '.csv')

    output_table.to_csv(output_filepath, index=False, sep=csv_separator)

    current_time = time.time()
    current_time_str = str(current_time).replace('.', '_')
    output_filepath = (
        output_path / (output_name + '_backup_' + current_time_str + '.csv'))
    output_table.to_csv(output_filepath, index=False, sep=csv_separator)

    output_filepath = output_path / (output_name + '.parquet')
    table = pa.Table.from_pandas(output_table)
    pa.parquet.write_table(table, output_filepath)
    #table2 = pa.parquet.read_table(output_filepath)

    output_filepath = output_path / (output_name + '.sqlite')
    engine = create_engine('sqlite:///' + output_filepath.as_posix(), echo=True)
    conn = engine.connect()
    table_name = output_name
    output_table.to_sql(table_name, conn, if_exists='replace')
    conn.close()

    output_filepath = output_path / (output_name + '.json')
    json_table = output_table.to_json(orient='columns')
    assert isinstance(json_table, str)
    parsed_json_table = json.loads(json_table)
    with open(output_filepath, 'w') as json_file:
        json.dump(parsed_json_table, json_file, indent=2)

    #db_connect_filepath = (
    #    Path.home() / 'Documents' / 'data_trades_db_connect.txt')
    #db_connect_string = read_text_file(db_connect_filepath)[0]
    #output_filepath = output_path / (output_name + '.sqlite')
    #engine = create_engine('postgresql+psycopg2://' + db_connect_string)
    #conn = engine.connect()
    #table_name = output_name
    #output_table.to_sql(table_name, conn, if_exists='replace')
    #conn.close()

    #pd.read_sql('select * from transactions_compiled', engine)
    #pd.read_sql('select * from filled_orders', engine)
    #conn = psycopg2.connect(
    #    database='data_trades', user='', password='',
    #    host='localhost', port='5432')


def main():


    csv_separator = '|'


    # load filled-orders-by-transaction
    ##############################

    filled_orders = load_filled_orders()


    # convert filled-orders-by-transaction to filled-orders-by-position_change
    ##############################

    filled_orders_by_position_change = (
        convert_to_filled_orders_by_position_change(
            filled_orders))

    output_path = get_filled_orders_by_position_change_output_path() 

    save_output_tables(
        filled_orders_by_position_change, output_path, 
        'filled_orders_by_position_change', csv_separator)


    # convert filled-orders-by-transaction to filled-orders-by-position
    ##############################

    filled_orders_by_ongoing_position = (
        convert_to_filled_orders_by_ongoing_position(
            filled_orders))

    output_path = get_filled_orders_by_ongoing_position_output_path() 

    save_output_tables(
        filled_orders_by_ongoing_position, output_path, 
        'filled_orders_by_ongoing_position', csv_separator)


    # convert filled-orders-by-position to filled-orders-by-symbol-day
    ##############################

    filled_orders_by_symbol_day = convert_to_filled_orders_by_symbol_day(
        filled_orders_by_ongoing_position)

    output_path = get_filled_orders_by_symbol_day_output_path() 

    save_output_tables(
        filled_orders_by_symbol_day, output_path, 
        'filled_orders_by_symbol_day', csv_separator)


if __name__ == '__main__':
    main()
