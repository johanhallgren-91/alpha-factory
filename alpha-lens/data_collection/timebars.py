import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import nest_asyncio
from typing import Union
import datetime as dt
import asyncio
#import pytz 
import os 


#nest_asyncio.apply()


def convert_to_ms(date_time: Union[dt.datetime, int]) -> int:
    return round(date_time.timestamp()*1000) if isinstance(date_time, dt.datetime) else int(date_time)


def start_exchange(exchange: str = 'binance', start_async: bool = False):
    ccxt_ = ccxt_async if start_async else ccxt
    exchange = getattr(ccxt_, exchange)({
        'apiKey': os.environ.get('API_KEY'), 
        'secret': os.environ.get('SECRET_KEY'), 
        'enableRateLimit': True}
    )
    if start_async: 
        asyncio.run(exchange.load_markets())
        asyncio.run(exchange.close())
    else:
        exchange.load_markets()
    return exchange


def get_timebars_async(
        ticker: str, start_datetime: Union[int, dt.datetime], end_datetime: Union[int, dt.datetime],
        timeframe: str, limit: int = 1000, timezone: str = 'UTC', exchange = None, 
    ) -> pd.DataFrame:
    """Asyncronicly downloads time bars from the exchange between start_datetime and current time for the specified ticker."""
    async def timebars_async(ticker:str, start_datetime:str, end_datetime, exchange, timeframe:str, limit:int, timezone: str) -> pd.DataFrame:
        timedelta = exchange.parse_timeframe(timeframe) * limit * 1000
        start_times = [start_datetime + timedelta * i for i in range(((end_datetime - start_datetime) // timedelta)+1)]
        corout = [
            asyncio.create_task(exchange.fetch_ohlcv(ticker, timeframe, start_time, limit))
            for start_time in start_times
        ]
        try:
            timebars = await asyncio.gather(*corout)
            timebars = format_timebars([item for sublist in timebars for item in sublist], timezone)
        except Exception as e:
            print(e)
            timebars = pd.DataFrame()
            
        await exchange.close()
        return timebars
    
    if exchange is None: exchange = start_exchange(start_async = True)
    start_datetime, end_datetime, = convert_to_ms(start_datetime), convert_to_ms(end_datetime)    
    return asyncio.run(timebars_async(ticker, start_datetime, end_datetime, exchange, timeframe, limit, timezone), debug = True)
    

def get_timebars(
        ticker: str, start_datetime: Union[int, dt.datetime], end_datetime: Union[int, dt.datetime],
        timeframe: str, limit: int = 1000, timezone: str = 'UTC', exchange = None, 
    ) -> pd.DataFrame:
    """Downloads time bars from the exchange between start_datetime and current time for the specified ticker."""
   
    if exchange is None: exchange = start_exchange(start_async = False)
    start_datetime, end_datetime, = convert_to_ms(start_datetime), convert_to_ms(end_datetime)
    timedelta = exchange.parse_timeframe(timeframe) * limit * 1000
    daydelta = exchange.parse_timeframe("1d") * 1000
    ohlcv = []

    while start_datetime < end_datetime:
        try:
            temp = exchange.fetch_ohlcv(ticker, timeframe, start_datetime, limit)
            ohlcv.extend(temp)

            if len(temp) == 0 and len(ohlcv) == 0:
                start_datetime += daydelta
            elif len(temp) == 0:
                start_datetime += timedelta
            else: 
                start_datetime = (ohlcv[-1][0] + 1)

        except Exception as e:
            print(e, ' ', ticker)
            return format_timebars(ohlcv, timezone) 

    return format_timebars(ohlcv, timezone)


def format_timebars(timebar: list, timezone: str = 'UTC') -> pd.DataFrame:
    """Formats the ohlcv list from ccxt as a dataframe"""
    bars = (
        pd.DataFrame(timebar, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
          .assign(datetime = lambda df_: 
                  df_['timestamp']
                    .pipe(pd.to_datetime, unit = 'ms', utc = True)
                    .apply(lambda x: x.tz_convert(timezone)))
          .drop_duplicates(subset = 'datetime')
          .set_index('datetime')
    )
    return bars

