

import pandas as pd
#from hypothesis import given, settings, reproduce_failure

from ...src.data_pipeline.transform_filled_orders import (
    match_cumulative_sums_by_row,
    #match_buy_sell_orders,
    #
    # so small: is it worth testing?
    #list_of_strings_without_nans,
    #
    # these functions are covered by Pandera schema, but might should undergo 
    #   further testing 
    #convert_to_filled_orders_by_position_change,
    #convert_to_filled_orders_by_ongoing_position,
    #convert_to_filled_orders_by_symbol_day,
    )

#from ...schemas.data_pipeline_schemas import FilledOrdersForDashboard


def test_match_cumulative_sums_by_row_01():

    sequence01 = [100]
    sequence02 = [50, 50]

    result = match_cumulative_sums_by_row(sequence01, sequence02)

    colnames = ['series1', 'series2', 'diff', 'series1_idx', 'series2_idx']
    correct_result = pd.DataFrame(
        ((100, 50, 50, 0, 0),
         ( 50, 50,  0, 0, 1)), columns=colnames)

    assert (result == correct_result).all().all()


def test_match_cumulative_sums_by_row_02():

    sequence01 = [50, 50]
    sequence02 = [100]

    result = match_cumulative_sums_by_row(sequence01, sequence02)

    colnames = ['series1', 'series2', 'diff', 'series1_idx', 'series2_idx']
    correct_result = pd.DataFrame(
        ((50, 100, 50, 0, 0),
         (50,  50,  0, 1, 0)), columns=colnames)

    assert (result == correct_result).all().all()


def test_match_cumulative_sums_by_row_03():

    sequence01 = [80, 120]
    sequence02 = [150, 50]

    result = match_cumulative_sums_by_row(sequence01, sequence02)

    colnames = ['series1', 'series2', 'diff', 'series1_idx', 'series2_idx']
    correct_result = pd.DataFrame(
        (( 80, 150, 70, 0, 0),
         (120,  70, 50, 1, 0),
         ( 50,  50,  0, 1, 1)), columns=colnames)

    assert (result == correct_result).all().all()


def test_match_cumulative_sums_by_row_04():

    sequence01 = [150, 50]
    sequence02 = [80, 120]

    result = match_cumulative_sums_by_row(sequence01, sequence02)

    colnames = ['series1', 'series2', 'diff', 'series1_idx', 'series2_idx']
    correct_result = pd.DataFrame(
        ((150,  80, 70, 0, 0),
         ( 70, 120, 50, 0, 1),
         ( 50,  50,  0, 1, 1)), columns=colnames)

    assert (result == correct_result).all().all()


def test_match_cumulative_sums_by_row_05():

    sequence01 = [100, 40, 160, 400, 75, 125, 100]
    sequence02 = [300, 50,  15,  70, 35,  80, 200, 50, 200]

    result = match_cumulative_sums_by_row(sequence01, sequence02)

    colnames = ['series1', 'series2', 'diff', 'series1_idx', 'series2_idx']
    correct_result = pd.DataFrame(
        ((100, 300, 200, 0, 0),
         ( 40, 200, 160, 1, 0),
         (160, 160,   0, 2, 0),
         (400,  50, 350, 3, 1),
         (350,  15, 335, 3, 2),
         (335,  70, 265, 3, 3),
         (265,  35, 230, 3, 4),
         (230,  80, 150, 3, 5),
         (150, 200,  50, 3, 6),
         ( 75,  50,  25, 4, 6),
         ( 25,  50,  25, 4, 7),
         (125,  25, 100, 5, 7),
         (100, 200, 100, 5, 8),
         (100, 100,   0, 6, 8)), columns=colnames)

    assert (result == correct_result).all().all()


"""
@given(df=FilledOrdersForDashboard.to_schema().strategy())
@settings(print_blob=True)
@reproduce_failure('6.54.6', b'AXicY2RgYGQAAUYIZoTxcQJGvFxSFBG2BwAFYgAM') 
def test_convert_to_filled_orders_by_position_change_01(df: pd.DataFrame):

    print(df)
    breakpoint()
    if not df.empty:
        result = convert_to_filled_orders_by_position_change(df)
        assert len(result) == df['positions_idx'].nunique()
"""














