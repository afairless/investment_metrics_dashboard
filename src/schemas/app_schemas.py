 
import pandera as pdr
from pandera.typing import Series as pdr_Series

from ..schemas.data_pipeline_schemas import (
    FilledOrdersByPositionChange,
    FilledOrdersByOngoingPosition,
    FilledOrdersBySymbolDay)



"""
Columns added from function 'itemize_datetime_data' are same for all 3 derived 
    tables
Schema for these columns is repeated for each derived table; is there a 
    convenient way to avoid the repetition?
"""

class FilledOrdersByPositionChangeDatetime(FilledOrdersByPositionChange):

    min_date: pdr_Series[pdr.dtypes.DateTime]
    min_date_quarter: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 4}, nullable=False)
    min_date_month: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 12}, nullable=False)
    min_date_week_of_year: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 53}, nullable=False)
    min_date_day_of_week: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 6}, nullable=False)
    min_date_day_name: pdr_Series[str] = pdr.Field(
        isin=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 
            'Saturday'], nullable=False)
    min_date_hour_of_day: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 23}, nullable=False)
    # column contains datetime.time type, which produces error
    #min_date_time_of_day: pdr_Series[pdr.dtypes.DateTime]  
    min_date_time_of_day: pdr_Series[pdr.dtypes.Any]
    min_date_market_time: pdr_Series[str] = pdr.Field(
        isin=['premarket', 'market', 'postmarket'], nullable=False)

    max_date: pdr_Series[pdr.dtypes.DateTime]
    max_date_quarter: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 4}, nullable=False)
    max_date_month: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 12}, nullable=False)
    max_date_week_of_year: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 53}, nullable=False)
    max_date_day_of_week: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 6}, nullable=False)
    max_date_day_name: pdr_Series[str] = pdr.Field(
        isin=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 
            'Saturday'], nullable=False)
    max_date_hour_of_day: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 23}, nullable=False)
    # column contains datetime.time type, which produces error
    #max_date_time_of_day: pdr_Series[pdr.dtypes.DateTime]
    max_date_time_of_day: pdr_Series[pdr.dtypes.Any]
    max_date_market_time: pdr_Series[str] = pdr.Field(
        isin=['premarket', 'market', 'postmarket'], nullable=False)


class FilledOrdersByOngoingPositionDatetime(FilledOrdersByOngoingPosition):

    min_date: pdr_Series[pdr.dtypes.DateTime]
    min_date_quarter: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 4}, nullable=False)
    min_date_month: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 12}, nullable=False)
    min_date_week_of_year: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 53}, nullable=False)
    min_date_day_of_week: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 6}, nullable=False)
    min_date_day_name: pdr_Series[str] = pdr.Field(
        isin=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 
            'Saturday'], nullable=False)
    min_date_hour_of_day: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 23}, nullable=False)
    # column contains datetime.time type, which produces error
    #min_date_time_of_day: pdr_Series[pdr.dtypes.DateTime]
    min_date_time_of_day: pdr_Series[pdr.dtypes.Any]
    min_date_market_time: pdr_Series[str] = pdr.Field(
        isin=['premarket', 'market', 'postmarket'], nullable=False)

    max_date: pdr_Series[pdr.dtypes.DateTime]
    max_date_quarter: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 4}, nullable=False)
    max_date_month: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 12}, nullable=False)
    max_date_week_of_year: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 53}, nullable=False)
    max_date_day_of_week: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 6}, nullable=False)
    max_date_day_name: pdr_Series[str] = pdr.Field(
        isin=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 
            'Saturday'], nullable=False)
    max_date_hour_of_day: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 23}, nullable=False)
    # column contains datetime.time type, which produces error
    #max_date_time_of_day: pdr_Series[pdr.dtypes.DateTime]
    max_date_time_of_day: pdr_Series[pdr.dtypes.Any]
    max_date_market_time: pdr_Series[str] = pdr.Field(
        isin=['premarket', 'market', 'postmarket'], nullable=False)


class FilledOrdersBySymbolDayDatetime(FilledOrdersBySymbolDay):

    min_date: pdr_Series[pdr.dtypes.DateTime]
    min_date_quarter: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 4}, nullable=False)
    min_date_month: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 12}, nullable=False)
    min_date_week_of_year: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 53}, nullable=False)
    min_date_day_of_week: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 6}, nullable=False)
    min_date_day_name: pdr_Series[str] = pdr.Field(
        isin=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 
            'Saturday'], nullable=False)
    min_date_hour_of_day: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 23}, nullable=False)
    # column contains datetime.time type, which produces error
    #min_date_time_of_day: pdr_Series[pdr.dtypes.DateTime]
    min_date_time_of_day: pdr_Series[pdr.dtypes.Any]
    min_date_market_time: pdr_Series[str] = pdr.Field(
        isin=['premarket', 'market', 'postmarket'], nullable=False)

    max_date: pdr_Series[pdr.dtypes.DateTime]
    max_date_quarter: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 4}, nullable=False)
    max_date_month: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 12}, nullable=False)
    max_date_week_of_year: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 1, 'max_value': 53}, nullable=False)
    max_date_day_of_week: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 6}, nullable=False)
    max_date_day_name: pdr_Series[str] = pdr.Field(
        isin=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 
            'Saturday'], nullable=False)
    max_date_hour_of_day: pdr_Series[int] = pdr.Field(
        in_range={'min_value': 0, 'max_value': 23}, nullable=False)
    # column contains datetime.time type, which produces error
    #max_date_time_of_day: pdr_Series[pdr.dtypes.DateTime]
    max_date_time_of_day: pdr_Series[pdr.dtypes.Any]
    max_date_market_time: pdr_Series[str] = pdr.Field(
        isin=['premarket', 'market', 'postmarket'], nullable=False)

