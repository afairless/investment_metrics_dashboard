
"""
Framework for collapsible sidebar and sub-menus adapted from:

    https://github.com/facultyai/dash-bootstrap-components/blob/main/examples/python/templates/multi-page-apps/sidebar-with-submenus/sidebar.py

    https://github.com/facultyai/dash-bootstrap-components/blob/main/examples/python/templates/multi-page-apps/responsive-collapsible-sidebar/sidebar.py
"""

from math import trunc

import dash_bootstrap_components as dbc
from dash import dcc, html


def convert_hour_to_string(hour_decimal: float) -> str:

    hour = str(trunc(hour_decimal))

    # surely there is a less ugly way to do this
    minute = str(int(round((int(format(round(
        hour_decimal, 2), '.2f').split('.')[1]) / 100) * 60, 0)))

    if len(minute) == 1:
        return hour + ':0' + minute
    else:
        return hour + ':' + minute


buy_sell_date_choice_dict = {'label': 'Buy and Sell Dates', 'value': 1}
buy_sell_date_choice = buy_sell_date_choice_dict['value']
slider_style01 = {'padding': '20px 10px 25px 4px'}
slider_style02 = {'margin-left': '6px'}
submenu_subheading_style = {'font-weight': 'bold'}

# add scrollbar when sidebar menus expand
sidebar_style = {'overflow': 'scroll'}


##################################################
# SIDEBAR COMPONENTS
##################################################
# Sidebar menus allow the user to select/filter the view of the data in 
#   displayed plots and tables
##################################################


sidebar_header = dbc.Row([
    dbc.Col(html.H3('Load and Select Transaction Data', className='display-4')),
    dbc.Col([
        html.Button(
            id='navbar-toggle',
            # use the Bootstrap navbar-toggler classes to style
            children=html.Span(className='navbar-toggler-icon'),
            className='navbar-toggler',
            # the navbar-toggler classes don't set color
            style={
                'color': 'rgba(0,0,0,.5)',
                'border-color': 'rgba(0,0,0,.1)'}),
        html.Button(
            id='sidebar-toggle',
            children=html.Span(className='navbar-toggler-icon'),
            className='navbar-toggler',
            style={
                'color': 'rgba(0,0,0,.5)',
                'border-color': 'rgba(0,0,0,.1)' })],
        # the column containing the toggle will be only as wide as the
        # toggle, resulting in the toggle being right aligned
        width='auto',
        align='center')])


# Provide descriptions of how transactions are grouped into positions
##################################################

grouping_text_01 = (f'Group by position change: '
    'An investor holds a single position for as long as '
    'the same number of shares in a particular equity/stock are held. If '
    'the investor buys or sells any shares, the investor has started a '
    'new position.')
grouping_text_02 = (f'Group by ongoing position: '
    'An investor holds a single position for as long as '
    'at least one share in an equity/stock is held.')
grouping_text_03 = (f'Group by symbol-day: '
    'All positions that an investor holds in a single equity/stock on a '
    'single day count as a single position. Positions spanning multiple '
    'days are assigned to the date when the position was initiated.')
#grouping_text_04 = (f'Group by day: '
#    'All positions that an investor holds across multiple equities/stocks on a '
#    'single day count as a single position. Positions spanning multiple '
#    'days are assigned to the date when the position was initiated.')


grouping_description_01 = html.Div([
    dbc.Button(
        id='group-description-01-collapse',
        children='Description',
        className='mb-3',
        color='secondary',
        n_clicks=0),
    dbc.Collapse(
        id='group-description-01-text',
        children=dbc.Card(dbc.CardBody(grouping_text_01)),
        is_open=False)
    ])


grouping_description_02 = html.Div([
    dbc.Button(
        id='group-description-02-collapse',
        children='Description',
        className='mb-3',
        color='secondary',
        n_clicks=0),
    dbc.Collapse(
        id='group-description-02-text',
        children=dbc.Card(dbc.CardBody(grouping_text_02)),
        is_open=False)
    ])


grouping_description_03 = html.Div([
    dbc.Button(
        id='group-description-03-collapse',
        children='Description',
        className='mb-3',
        color='secondary',
        n_clicks=0),
    dbc.Collapse(
        id='group-description-03-text',
        children=dbc.Card(dbc.CardBody(grouping_text_03)),
        is_open=False)
    ])


# Submenus for selecting positions by various criteria
##################################################

submenu_01 = html.Div([
    dbc.Label(
        'Select simulated or real transactions', 
        style=submenu_subheading_style),
    dbc.Checklist( 
        id='real-simulator-both',
        options=[{'label': e, 'value': e} for e in ['Simulator', 'Real']],
        value=['Simulator', 'Real'], 
        inline=True), 
    html.H1(),
    dbc.Label(
        'Select stocks/equities or options', 
        style=submenu_subheading_style),
    dbc.Checklist( 
        id='stocks-options-both',
        options=[
            {'label': e, 'value': e} for e in ['Stocks/Equities', 'Options']],
        value=['Stocks/Equities', 'Options'], 
        inline=True), 
    html.H1(),
    dbc.Label(
        'Select long or short positions', 
        style=submenu_subheading_style),
    dbc.Checklist( 
        id='long-short-both',
        options=[
            {'label': e, 'value': e} for e in ['Long', 'Short']],
        value=['Long', 'Short'], 
        inline=True), 
    html.H1(),
    dbc.Label(
        'Select bullish or bearish positions', 
        style=submenu_subheading_style),
    dbc.Checklist( 
        id='bull-bear-both',
        options=[
            {'label': e, 'value': e} for e in ['Bull', 'Bear']],
        value=['Bull', 'Bear'], 
        inline=True), 
    ], className='mb-4')


submenu_02 = [
    html.H3(),
    dbc.Label('Should both buy and sell dates fall within selected '
        'time interval(s), or only the sell date?', 
        style=submenu_subheading_style),
    dbc.RadioItems(
        id='buy-sell-dates',
        options=[
            {'label': 'Sell Date', 'value': 0},
            buy_sell_date_choice_dict],
        value=1,
        inline=True),
    html.H1(),
    html.P('Select date interval', style=submenu_subheading_style),
    dcc.DatePickerRange(id='date-range'),
    html.H1(),
    html.Div([
        dbc.Label(
            'Select quarter during the year', style=submenu_subheading_style),
        dbc.Checklist(
            id='quarter-of-year',
            options=[
                {'label': '1st', 'value': 1}, 
                {'label': '2nd', 'value': 2}, 
                {'label': '3rd', 'value': 3}, 
                {'label': '4th', 'value': 4}],
            value=[1, 2, 3, 4],
            inline=True),
        ], className='mb-4'),
    html.H1(),
    html.Div([
        dbc.Label('Select month of the year', style=submenu_subheading_style),
        dbc.Checklist(
            id='month-of-year',
            options=[
                {'label': 'Jan', 'value': 1}, 
                {'label': 'Feb', 'value': 2}, 
                {'label': 'Mar', 'value': 3}, 
                {'label': 'Apr', 'value': 4},
                {'label': 'May', 'value': 5},
                {'label': 'Jun', 'value': 6},
                {'label': 'Jul', 'value': 7},
                {'label': 'Aug', 'value': 8},
                {'label': 'Sep', 'value': 9},
                {'label': 'Oct', 'value': 10},
                {'label': 'Nov', 'value': 11},
                {'label': 'Dec', 'value': 12}],
            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            inline=True),
        ], className='mb-4'),
    #html.P('Select week of the year'),
    # how make week of year cyclical in selection?
    html.H1(),
    html.Div([
        dbc.Label('Select day of the week', style=submenu_subheading_style),
        dbc.Checklist(
            id='day-of-week',
            options=[
                {'label': 'Mon', 'value': 0}, 
                {'label': 'Tue', 'value': 1}, 
                {'label': 'Wed', 'value': 2}, 
                {'label': 'Thu', 'value': 3},
                {'label': 'Fri', 'value': 4},
                {'label': 'Sat', 'value': 5},
                {'label': 'Sun', 'value': 6}],
            value=[0, 1, 2, 3, 4, 5, 6],
            inline=True),
        ], className='mb-4')]


submenu_03 = [
    html.Div(
        style=slider_style01,
        children=[
            dbc.Row([
                dbc.Col(html.P(
                    'Select hour of the day', style=submenu_subheading_style)),
                dbc.Col(html.Div(
                    id='hour-of-day-min', 
                    style={'color': 'lightblue'}), 
                    md=2),
                dbc.Col(html.Div(
                    id='hour-of-day-max', 
                    style={'color': 'lightblue'}), 
                    md=2)]),
            html.Div(
                style=slider_style02,
                children=dcc.RangeSlider(
                    id='hour-of-day',
                    min=0, max=24, step=0.25,
                    marks={
                        0:   convert_hour_to_string(0),
                        4:   convert_hour_to_string(4),
                        9.5: convert_hour_to_string(9.5),
                        12:  convert_hour_to_string(12),
                        16:  convert_hour_to_string(16),
                        20:  convert_hour_to_string(20),
                        24:  convert_hour_to_string(24)},
                    value=[0, 24],
                    allowCross=False))]),
    html.Div([
        dbc.Label('Select market time of the day', style=submenu_subheading_style),
        dbc.Checklist(
            id='market-time-of-day',
            inline=True),
        ], className='mb-4')]


submenu_04 = html.Div([
    dbc.Label('Select stock/equity', style=submenu_subheading_style),
    dbc.Checklist(
        id='all-stock-symbol',
        options=[{'label': 'All', 'value': 'All'}],
        value=['All'],
        inline=True),
    dbc.Checklist(
        id='stock-symbol',
        inline=True),
    ], className='mb-4')


submenu_05 = [
    html.Div(
        style=slider_style01,
        children=[
            dbc.Row([
                dbc.Col(html.P(
                    'Select Buy Fill Price ($)', 
                    style=submenu_subheading_style)),
                dbc.Col(html.Div(
                    id='fill-price-buy-min', 
                    style={'color': 'lightblue'}), 
                    md=2),
                dbc.Col(html.Div(
                    id='fill-price-buy-max', 
                    style={'color': 'lightblue'}), 
                    md=2)]),
            html.Div(
                style=slider_style02,
                children=dcc.RangeSlider(
                    id='fill-price-buy',
                    step=0.01,
                    marks={i: f'{10 ** i}' for i in range(-2, 10)},
                    allowCross=False))]),
    html.Div(
        style=slider_style01,
        children=[
            dbc.Row([
                dbc.Col(html.P(
                    'Select Sell Fill Price ($)', 
                    style=submenu_subheading_style)),
                dbc.Col(html.Div(
                    id='fill-price-sell-min', 
                    style={'color': 'lightblue'}), 
                    md=2),
                dbc.Col(html.Div(
                    id='fill-price-sell-max', 
                    style={'color': 'lightblue'}), 
                    md=2)]),
            html.Div(
                style=slider_style02,
                children=dcc.RangeSlider(
                    id='fill-price-sell',
                    step=0.01,
                    marks={i: f'{10 ** i}' for i in range(-2, 10)},
                    allowCross=False))]),
    html.Div(
        style=slider_style01,
        children=[
            dbc.Row([
                dbc.Col(html.P(
                    'Select Commission Cost (sum of buy & sell) ($)',
                    style=submenu_subheading_style)),
                dbc.Col(html.Div(
                    id='commission-buy-sell-min', 
                    style={'color': 'lightblue'}), 
                    md=2),
                dbc.Col(html.Div(
                    id='commission-buy-sell-max', 
                    style={'color': 'lightblue'}), 
                    md=2)]),
            html.Div(
                style=slider_style02,
                children=dcc.RangeSlider(
                    id='commission-buy-sell',
                    step=0.01,
                    marks={i: f'{10 ** i}' for i in range(-2, 10)},
                    allowCross=False))])]


submenu_06 = [
    html.Div(
        style=slider_style01,
        children=[
            dbc.Row([
                dbc.Col(html.P(
                    'Select Balance Change ($)',
                    style=submenu_subheading_style)),
                dbc.Col(html.Div(
                    id='balance-change-min', 
                    style={'color': 'lightblue'}), 
                    md=2),
                dbc.Col(html.Div(
                    id='balance-change-max', 
                    style={'color': 'lightblue'}), 
                    md=2)]),
            html.Div(
                style=slider_style02,
                children=dcc.RangeSlider(
                    id='balance-change',
                    step=1,
                    marks={0: '0'},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    allowCross=False))]),
    html.Div(
        style=slider_style01,
        children=[
            dbc.Row([
                dbc.Col(html.P(
                    'Select Balance Change including Commissions ($)',
                    style=submenu_subheading_style)),
                dbc.Col(html.Div(
                    id='balance-change-commission-min', 
                    style={'color': 'lightblue'}), 
                    md=2),
                dbc.Col(html.Div(
                    id='balance-change-commission-max', 
                    style={'color': 'lightblue'}), 
                    md=2)]),
            html.Div(
                style=slider_style02,
                children=dcc.RangeSlider(
                    id='balance-change-commission',
                    step=1,
                    marks={0: '0'},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    allowCross=False))])]


submenu_07 = [
    html.Div(
        style=slider_style01,
        children=[
            dbc.Row([
                dbc.Col(html.P(
                    'Select Number of Shares',
                    style=submenu_subheading_style)),
                dbc.Col(html.Div(
                    id='shares-num-fill-min', 
                    style={'color': 'lightblue'}), 
                    md=2),
                dbc.Col(html.Div(
                    id='shares-num-fill-max', 
                    style={'color': 'lightblue'}), 
                    md=2)]),
            html.Div(
                style=slider_style02,
                children=dcc.RangeSlider(
                    id='shares-num-fill',
                    marks={i: f'{10 ** i}' for i in range(0, 10)},
                    allowCross=False))])]


submenu_08 = [
    html.Div([
        dbc.Label('Select tags', style=submenu_subheading_style),
        dbc.Checklist(
            id='all-tags',
            options=[{'label': 'All', 'value': 'All'}],
            value=[],
            inline=True),
        dbc.Checklist(
            id='tags',
            inline=True),
        ], className='mb-4'),
    ]





sidebar = html.Div(
    id='sidebar',
    children=[
        sidebar_header,
        html.H1(),
        dbc.Row([
            dbc.Col(dbc.NavLink('Filled Orders', href='/position-change')), 
            dbc.Col(dcc.Upload(
                id='orders-upload', 
                children='Upload', 
                className='button'))]),
        html.H1(),
        html.Div(html.P(
            'Please choose how to group transactions together into positions:', 
            className='lead')),
        dbc.Row(dbc.Col(dbc.NavLink(
            'Position Change', href='/position-change'))),
        grouping_description_01,
        dbc.Row(dbc.Col(dbc.NavLink(
            'Ongoing Position', href='/ongoing-position'))),
        grouping_description_02,
        dbc.Row(dbc.Col(dbc.NavLink('Symbol-Day', href='/symbol-day'))),
        grouping_description_03,
        #dbc.Row(dbc.Col(dbc.NavLink('Day', href='/day'))),
        #grouping_description_04,
        html.H1(),
        dbc.Accordion([
            dbc.AccordionItem(
                title='Select by Simulated/Real Trades',
                children=submenu_01),
            dbc.AccordionItem(
                title='Select by Date/Day',
                children=submenu_02),
            dbc.AccordionItem(
                title='Select by Time',
                children=submenu_03),
            dbc.AccordionItem(
                title='Select by Stock/Equity Symbol',
                children=submenu_04),
            dbc.AccordionItem(
                title='Select by Price/Cost',
                children=submenu_05),
            dbc.AccordionItem(
                title='Select by Balance Change',
                children=submenu_06),
            dbc.AccordionItem(
                title='Select by Volume',
                children=submenu_07),
            dbc.AccordionItem(
                title='Select by Tag',
                children=submenu_08)],
            start_collapsed=True,
            always_open=False,
            ), 
        ],
    style=sidebar_style,
    )

