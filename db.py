import datetime
import logging.config
from os import environ
from typing import Tuple, Dict

import psycopg2

import log
from model.FiatCryptoPrice import FiatCryptoPrice

logging.config.dictConfig(log.LOGGING_CONF)
logger = logging.getLogger("db")

logger.info("connecting to db")

try:
    conn = psycopg2.connect(database=environ.get("CRYPTO_DB"),
                            user=environ.get("CRYPTO_DB_USERNAME"),
                            password=environ.get("CRYPTO_DB_PASSWORD"),
                            host=environ.get("CRYPTO_DB_HOST"),
                            port="5432")
    logger.info("connected successfully to db")
except:
    logger.error("couldn't connect to db")


def get_fiat_crypto_prices() -> Tuple[FiatCryptoPrice]:
    return cursor_wrapper(get_fiat_crypto_prices_action)


def get_fiat_crypto_prices_action(cursor) -> Tuple[FiatCryptoPrice]:
    cursor.execute(" select fc.id, "
                   "fc.high,"
                   "fc.low,"
                   "fc.open,"
                   "fc.close,"
                   "fc.volume,"
                   "fc.date_time,"
                   "p.period,"
                   "f.code,"
                   "currency.code"
                   " from fc_market_price fc"
                   " inner join period p on fc.period_id = p.id"
                   " inner join fiat f on fc.fiat_id = f.id"
                   " inner join crypto_currency currency on fc.cc_id = currency.id"
                   # " order by fc.date_time asc;")
                   " order by fc.date_time desc limit 100")
    fiat_crypto_tuples = cursor.fetchall()

    logger.debug('fetched %d fiat crypto prices', len(fiat_crypto_tuples))

    return tuple(map(lambda fcp: FiatCryptoPrice(fcp), fiat_crypto_tuples))


def get_fiat_crypto_ta(price_id: int) -> Dict[str, float]:
    return cursor_wrapper(get_fiat_crypto_ta_action, price_id)


def get_fiat_crypto_ta_action(cursor, price_id: int) -> Dict[str, float]:
    cursor.execute("select ta.indicator, ta.value from fc_market_price_ta ta where ta.fc_market_price_id = %s",
                   (price_id,))
    ta_indicators = cursor.fetchall()

    logger.debug('fetched %d indicators for fiat crypto price id:%d', len(ta_indicators), price_id)

    return dict((indicator, value) for indicator, value in ta_indicators)


def get_sentiment(period: int, crypto_currency_code: str, date_time: datetime) -> float:
    return cursor_wrapper(get_sentiment_action, period, crypto_currency_code, date_time)


def get_sentiment_action(cursor, period: int, crypto_currency_code: str, date_time: datetime) -> float:
    cursor.execute("select sentiment from twitter_sentiment s"
                   " inner join crypto_currency cc on cc.id = s.crypto_currency_id "
                   " inner join period p on p.id = s.period "
                   " where p.period = %s and cc.code = %s and s.date_time = %s",
                   (period, crypto_currency_code, date_time))
    sentiment = cursor.fetchone()

    if sentiment:
        logger.debug('fetched sentiment:%s for period:%s, crypto:%s, date time:%s',
                     sentiment, period, crypto_currency_code, date_time)
        return sentiment[0]
    else:
        logger.debug('no sentiment for period:%d, crypto:%s, date time:%s, returning default value',
                     period, crypto_currency_code, date_time)
        return None


def cursor_wrapper(db_function, *args):
    cursor = None
    try:
        cursor = conn.cursor()
        return db_function(cursor, *args)
    finally:
        if cursor is not None:
            cursor.close()
