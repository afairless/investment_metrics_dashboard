#! usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path

import pandera as pdr
from pandera.typing import DataFrame as pdr_DataFrame

from ...data_pipeline.transform_filled_orders import save_output_tables

from ...schemas.data_pipeline_schemas import (
    BrokerOrdersTagsNotesForDashboard,
    FilledOrdersForDashboard,
    )



# SET INPUT AND OUTPUT DIRECTORY PATHS
##############################

def get_example_data_input_path() -> Path:
    """
    Returns path to directory where input example data is stored
    """

    path = (
        Path(__file__).parent.parent / 'input_data' / 
        'filled_orders_input_example.csv')

    return path  


def get_example_data_output_path() -> Path:
    """
    Returns path to directory where example data output will be saved
    """

    output_path = Path(__file__).parent.parent / 'output_data'
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path  


# LOAD, TRANSFORM, AND QUALITY-CHECK DATA
##############################

@pdr.check_output(BrokerOrdersTagsNotesForDashboard.to_schema(), lazy=True)
def load_example_data_input(
    ) -> pdr_DataFrame[BrokerOrdersTagsNotesForDashboard]:

    input_filepath = get_example_data_input_path()

    schema = {
        'order_submit_time': str, 
        'order_fill_cancel_time': str, 
        'symbol': str, 
        'buy_or_sell': str, 
        'buy_sell_unit': int, 
        'shares_num_submit': int, 
        'shares_num_fill': float,
        'shares_num_not_fill': int,
        'limit_price': float, 
        'fill_price': float,
        'order_status': str,
        'order_route': str,
        'order_duration': str,
        'simulator_or_real': str,
        'commission_cost': float, 
        'contract_expiration_date': str, 
        'tags': str, 
        'notes': str}

    df = pd.read_csv(
        input_filepath, dtype=schema, usecols=schema.keys(), sep='|', 
        index_col=False)

    df['order_submit_time'] = pd.to_datetime(df['order_submit_time'])
    df['order_fill_cancel_time'] = pd.to_datetime(df['order_fill_cancel_time'])

    return df


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


@pdr.check_io(
    orders_df=BrokerOrdersTagsNotesForDashboard.to_schema(), 
    out=FilledOrdersForDashboard.to_schema(), 
    lazy=True)
def extract_filled_orders(
    orders_df:  pd.DataFrame) -> pdr_DataFrame[FilledOrdersForDashboard]:
    """
    Extract data for filled orders/transactions and calculate changes in shares
        held and balance/money and index each position
    """

    orders_df = orders_df.sort_values(['symbol', 'order_submit_time'])

    # extract filled orders from orders table
    filled_orders = orders_df.loc[~orders_df['fill_price'].isna(), :].copy()


    # create derived data/columns
    ##############################

    stock_or_option_bool = identify_stocks_or_options(filled_orders['symbol'])
    # options are counted as contracts, which each represent 100 shares; provide
    #   array of factors to convert contracts to shares
    options_factor = np.where(stock_or_option_bool, 1, 100)
    filled_orders['shares_num_submit'] = (
        options_factor * filled_orders['shares_num_submit'])
    filled_orders['shares_num_fill'] = (
        options_factor * filled_orders['shares_num_fill'])

    filled_orders['share_num_change'] = (
        filled_orders['buy_sell_unit'] * filled_orders['shares_num_fill'])
    filled_orders['balance_change'] = (
        filled_orders['share_num_change'] * filled_orders['fill_price'])
    filled_orders['balance_change_commission'] = (
        filled_orders['balance_change'] - filled_orders['commission_cost'])
    filled_orders['share_num_change'] = -filled_orders['share_num_change']
    filled_orders['share_num_held'] = filled_orders['share_num_change'].cumsum()

    # when all shares of a position are sold (i.e., 'share_num_held' equals 
    #   zero), a position is closed; a separate position is opened when shares 
    #   are subsequently purchased
    filled_orders['positions_idx'] = (
        (filled_orders['share_num_held'] == 0)
        .shift(fill_value=0)
        .cumsum()
        .astype(int))

    # when the number of shares for corresponding buy and sell orders are
    #   matched, they should cancel out (i.e., sum to) zero for each position
    # if some buy or sell orders are missing so that the number of shares on 
    #   the buy and sell side do not match, the final number of shares for a
    #   position will not be zero; the data processing can not handle this
    #   condition and will produce erroneous results; check for this condition
    #   for the last position of each symbol and raise an assertion error if it
    #   is present
    assert (
        filled_orders[['symbol', 'share_num_held']]
        .groupby('symbol')
        .last() == 0).all().iloc[0]

    return filled_orders


def main():

    csv_separator = '|'
    example_input_df = load_example_data_input()
    filled_orders = extract_filled_orders(example_input_df)
    output_path = get_example_data_output_path() 
    save_output_tables(
        filled_orders, output_path, 'filled_orders', csv_separator)


if __name__ == '__main__':
    main()
