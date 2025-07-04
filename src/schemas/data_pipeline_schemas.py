 
import pandera.pandas as pdr
from pandera.typing import Series as pdr_Series
from typing import Optional, Any
 
 
# SET DATAFRAME SCHEMAS
##############################

class LoadedTransformedBrokerOrdersForDashboard(pdr.DataFrameModel):

    # is there a way to check number of rows in dataframe?

    order_submit_time: pdr_Series[pdr.dtypes.DateTime]
    order_fill_cancel_time: pdr_Series[pdr.dtypes.DateTime] = pdr.Field(
        nullable=True)

    symbol: pdr_Series[str] = pdr.Field(
        str_length={'min_value': 1, 'max_value': 17}, nullable=False)

    price_spread: Optional[pdr_Series[float]] = pdr.Field(ge=0)

    buy_or_sell: pdr_Series[str] = pdr.Field(
        isin=[
            'buy', 'sell',
            'buy to open', 'sell to close',
            'buy to cover', 'sell short'], 
        nullable=False)
    buy_sell_unit: pdr_Series[int] = pdr.Field(isin=[-1, 1], nullable=False)

    shares_num_submit: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 1e5}, nullable=False)
    shares_num_fill: pdr_Series[float] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 1e5}, nullable=True)
    shares_num_not_fill: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 1e5}, nullable=True)

    stop_price: Optional[pdr_Series[float]] = pdr.Field(ge=0.01, nullable=True)
    limit_price: pdr_Series[float] = pdr.Field(
        in_range={'min_value': -1, 'max_value': 5000}, nullable=False)
    fill_price: pdr_Series[float] = pdr.Field(
        in_range={'min_value': 0.01, 'max_value': 5000}, nullable=True)

    simulator_or_real: pdr_Series[str] = pdr.Field(
        isin=['simulator', 'real'], nullable=False)

    commission_cost: pdr_Series[float] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 1000}, nullable=False)
    #contract_expiration_date: Optional[pdr_Series[pdr.dtypes.DateTime]]
    contract_expiration_date: pdr_Series[Any] = pdr.Field(nullable=True)

    class Config:
        strict = False
        ordered = True


class BrokerOrdersTagsNotesForDashboard(
    LoadedTransformedBrokerOrdersForDashboard):

    tags: pdr_Series[str] = pdr.Field(nullable=True)
    notes: pdr_Series[str] = pdr.Field(nullable=True)


class FilledOrdersForDashboard(BrokerOrdersTagsNotesForDashboard):

    share_num_change: pdr_Series[float] = pdr.Field(nullable=False)
    balance_change: pdr_Series[float] = pdr.Field(nullable=False)
    balance_change_commission: pdr_Series[float] = pdr.Field(nullable=False)
    share_num_held: pdr_Series[float] = pdr.Field(nullable=False)
    positions_idx: pdr_Series[int] = pdr.Field(nullable=False)

    class Config:
        coerce = True


class FilledOrdersByPositionChange(pdr.DataFrameModel):

    positions_idx: pdr_Series[int] = pdr.Field(nullable=False)

    symbol: pdr_Series[str] = pdr.Field(
        str_length={'min_value': 1, 'max_value': 17}, nullable=False)

    simulator_or_real: pdr_Series[str] = pdr.Field(
        isin=['simulator', 'real'], nullable=False)


    # columns for buy orders
    ##############################

    match_shares_num_fill_buy: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 1e5}, nullable=False)
    match_shares_num_fill_sell: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 1e5}, nullable=False)
    match_shares_num_fill_diff: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 1e5}, nullable=False)

    match_idx_buy: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 1e6}, nullable=False)
    match_idx_sell: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 1e6}, nullable=False)

    match_shares_num_fill: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 1e5}, nullable=False)

    order_submit_time_buy: pdr_Series[pdr.dtypes.DateTime]
    order_fill_cancel_time_buy: pdr_Series[pdr.dtypes.DateTime]

    shares_num_submit_buy: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 1e5}, nullable=False)

    limit_price_buy: pdr_Series[float] = pdr.Field(
        in_range={'min_value': -1, 'max_value': 5000}, nullable=False)
    fill_price_buy: pdr_Series[float] = pdr.Field(
        in_range={'min_value': 0.01, 'max_value': 5000}, nullable=True)

    order_status_buy: pdr_Series[str] = pdr.Field(nullable=True)
    order_route_buy: pdr_Series[str] = pdr.Field(nullable=True)
    order_duration_buy: pdr_Series[str] = pdr.Field(nullable=True)

    commission_cost_buy: pdr_Series[float] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 1000}, nullable=False)

    tags_buy: pdr_Series[str] = pdr.Field(nullable=True)
    notes_buy: pdr_Series[str] = pdr.Field(nullable=True)


    # columns for sell orders
    ##############################

    match_shares_num_fill_sell: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 1e5}, nullable=False)
    match_shares_num_fill_sell: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 1e5}, nullable=False)
    match_shares_num_fill_diff: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 1e5}, nullable=False)

    match_idx_sell: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 1e6}, nullable=False)
    match_idx_sell: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 1e6}, nullable=False)

    match_shares_num_fill: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 1e5}, nullable=False)

    order_submit_time_sell: pdr_Series[pdr.dtypes.DateTime]
    order_fill_cancel_time_sell: pdr_Series[pdr.dtypes.DateTime]

    shares_num_submit_sell: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 1e5}, nullable=False)

    limit_price_sell: pdr_Series[float] = pdr.Field(
        in_range={'min_value': -1, 'max_value': 5000}, nullable=False)
    fill_price_sell: pdr_Series[float] = pdr.Field(
        in_range={'min_value': 0.01, 'max_value': 5000}, nullable=True)

    order_status_sell: pdr_Series[str] = pdr.Field(nullable=True)
    order_route_sell: pdr_Series[str] = pdr.Field(nullable=True)
    order_duration_sell: pdr_Series[str] = pdr.Field(nullable=True)

    commission_cost_sell: pdr_Series[float] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 1000}, nullable=False)

    tags_sell: pdr_Series[str] = pdr.Field(nullable=True)
    notes_sell: pdr_Series[str] = pdr.Field(nullable=True)


    # columns for combined buy-sell data
    ##############################

    stock_or_option: pdr_Series[str] = pdr.Field(nullable=False)
    long_or_short: pdr_Series[str] = pdr.Field(nullable=False)
    bull_or_bear: pdr_Series[str] = pdr.Field(nullable=False)
    fill_price_change: pdr_Series[float] = pdr.Field(nullable=False)

    balance_change: pdr_Series[float] = pdr.Field(nullable=False)
    balance_change_commission: pdr_Series[float] = pdr.Field(nullable=False)

    spent_outflow: pdr_Series[float] = pdr.Field(nullable=False)
    spent_outflow_commission: pdr_Series[float] = pdr.Field(nullable=False)

    class Config:
        strict = True
        ordered = True
        unique = ['positions_idx', 'match_idx_buy', 'match_idx_sell']
        coerce = True


class FilledOrdersByOngoingPosition(pdr.DataFrameModel):

    positions_idx: pdr_Series[int] = pdr.Field(nullable=False)

    order_submit_time: pdr_Series[pdr.dtypes.DateTime]
    order_fill_cancel_time: pdr_Series[pdr.dtypes.DateTime]

    symbol: pdr_Series[str] = pdr.Field(
        str_length={'min_value': 1, 'max_value': 17}, nullable=False)

    num_buy_sell_orders: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 2, 'max_value': 500}, nullable=False)

    shares_num_fill: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 1e5}, nullable=True)

    order_status: pdr_Series[str] = pdr.Field(nullable=False)
    order_route: pdr_Series[str] = pdr.Field(nullable=True)
    order_duration: pdr_Series[str] = pdr.Field(nullable=True)

    simulator_or_real: pdr_Series[str] = pdr.Field(
        isin=['simulator', 'real'], nullable=False)

    commission_cost: pdr_Series[float] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 5000}, nullable=False)

    tags: pdr_Series[str] = pdr.Field(nullable=True)
    notes: pdr_Series[str] = pdr.Field(nullable=True)

    balance_change: pdr_Series[float] = pdr.Field(nullable=False)
    balance_change_commission: pdr_Series[float] = pdr.Field(nullable=False)

    spent_outflow: pdr_Series[float] = pdr.Field(nullable=False)
    spent_outflow_commission: pdr_Series[float] = pdr.Field(nullable=False)

    stock_or_option: pdr_Series[str] = pdr.Field(nullable=False)
    long_or_short: pdr_Series[str] = pdr.Field(nullable=False)
    bull_or_bear: pdr_Series[str] = pdr.Field(nullable=False)
    fill_price_change: pdr_Series[float] = pdr.Field(nullable=False)

    buy_mean_fill_price: pdr_Series[float] = pdr.Field(nullable=False)
    sell_mean_fill_price: pdr_Series[float] = pdr.Field(nullable=False)

    class Config:
        strict = True
        ordered = True
        coerce = True


class FilledOrdersBySymbolDay(pdr.DataFrameModel):

    positions_idx: pdr_Series[str] = pdr.Field(nullable=False)
    submit_date: pdr_Series[pdr.dtypes.DateTime]

    order_submit_time: pdr_Series[pdr.dtypes.DateTime]
    order_fill_cancel_time: pdr_Series[pdr.dtypes.DateTime]

    symbol: pdr_Series[str] = pdr.Field(nullable=False)

    num_buy_sell_orders: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 2, 'max_value': 500}, nullable=False)

    shares_num_fill: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 1e5}, nullable=True)

    order_status: pdr_Series[str] = pdr.Field(nullable=False)
    order_route: pdr_Series[str] = pdr.Field(nullable=False)
    order_duration: pdr_Series[str] = pdr.Field(nullable=False)

    simulator_or_real: pdr_Series[str] = pdr.Field(nullable=False)

    commission_cost: pdr_Series[float] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 5000}, nullable=False)

    tags: pdr_Series[str] = pdr.Field(nullable=True)
    notes: pdr_Series[str] = pdr.Field(nullable=True)

    balance_change: pdr_Series[float] = pdr.Field(nullable=False)
    balance_change_commission: pdr_Series[float] = pdr.Field(nullable=False)

    spent_outflow: pdr_Series[float] = pdr.Field(nullable=False)
    spent_outflow_commission: pdr_Series[float] = pdr.Field(nullable=False)

    stock_or_option: pdr_Series[str] = pdr.Field(nullable=False)
    long_or_short: pdr_Series[str] = pdr.Field(nullable=False)
    bull_or_bear: pdr_Series[str] = pdr.Field(nullable=False)
    fill_price_change: pdr_Series[float] = pdr.Field(nullable=False)

    buy_mean_fill_price: pdr_Series[float] = pdr.Field(nullable=False)
    sell_mean_fill_price: pdr_Series[float] = pdr.Field(nullable=False)

    class Config:
        strict = True
        ordered = False
        coerce = True


class PriceWeightedByShareNumberResult(pdr.DataFrameModel):

    positions_idx: pdr_Series[int] = pdr.Field(nullable=False)
    weighted_average_price: pdr_Series[float] = pdr.Field(nullable=False)

    class Config:
        strict = True
        ordered = True



