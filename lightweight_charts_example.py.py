import datetime as dt
import yfinance as yf
from lightweight_charts import Chart
import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_bar_data(symbol, timeframe):
    if timeframe in ('1m', '5m', '30m'):
        days = 7 if timeframe == '1m' else 60
        start_date = dt.datetime.now()-dt.timedelta(days=days)
    else:
        start_date = None

    chart.spinner(True)
    logger.error(f"Fetching data: {symbol}, {start_date}, {timeframe}")
    data = yf.download(symbol, start_date, interval=timeframe)

    logger.error(
        f"Fetched: LEN:{len(data)},\nINDEX: {data.index},\nCOLUMNS: {data.columns}")
    # Now columns are 'Open', 'High', etc.
    data = data.xs(symbol, axis=1, level=1)

    chart.spinner(False)

    if data.empty:
        return False
    chart.set(data)
    return True


def on_search(chart, searched_string):
    if get_bar_data(searched_string, chart.topbar['timeframe'].value):
        chart.topbar['symbol'].set(searched_string)


def on_timeframe_selection(chart):
    get_bar_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)


def on_markers_toggle(chart: Chart):
    logger.error(
        f"Markers Toggled: {chart.topbar['symbol'].value}, {chart.topbar['timeframe'].value}, {chart.topbar['markers'].value}")
    markers_on: bool = (chart.topbar['markers'].value or 'OFF') == 'ON'

    if markers_on:
        df = chart.candle_data

        shapes = ['circle', 'arrow_up', 'arrow_down', 'square']
        positions = ['above', 'below', 'inside']
        colors = ['red', 'blue', 'green', 'orange']

        num_markers = len(shapes) * len(positions) * len(colors)
        num_markers = max(num_markers, len(df))

        sampled = df.sample(n=num_markers, random_state=42)

        markers = []
        for shape in shapes:
            for pos in positions:
                for color in colors:
                    idx = np.random.choice(sampled.index)
                    text = f"{pos}, {color}, {shape}"
                    markers.append(
                        dict(
                            time=idx,  # or row.name
                            position=pos,
                            color=color,
                            shape=shape,
                            text=text
                        )
                    )

        chart.marker_list(markers)
        # markers = [
        #     dict(time=time1, position='aboveBar', color='red', shape='circle', text='A'),
        #     dict(time=time2, position='belowBar', color='blue', shape='arrowUp', text='B')
        # ]
        # chart.marker_list(markers)


if __name__ == '__main__':
    chart = Chart(
        toolbox=True, debug=False,
        maximize=True,
        title="Chart Title"
    )
    chart.legend(True)
    chart.events.search += on_search
    chart.topbar.textbox('symbol', 'NVDA', func=on_search)
    chart.topbar.switcher(
        'timeframe',
        ('1m', '5m', '30m', '1d', '1wk'),
        default='5m',
        func=on_timeframe_selection
    )

    chart.topbar.switcher(
        'markers',
        ('ON', 'OFF'),
        default='OFF',
        func=on_markers_toggle
    )
    get_bar_data('NVDA', '1h')

    chart.legend(visible=True, font_size=14)
    chart.crosshair(mode='normal', vert_color='#FFFFFF', vert_style='dotted',
                    horz_color='#FFFFFF', horz_style='dotted')

    chart.show(block=True)
