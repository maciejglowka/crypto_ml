import log
import logging.config
import pandas as pd
import db
from model.Indicator import Indicator
from model.PreviousPriceHolder import PreviousPriceHolder

logging.config.dictConfig(log.LOGGING_CONF)
logger = logging.getLogger("main")


def main():
    logger.info("starting ml evaluation")

    fiat_crypto_prices = db.get_fiat_crypto_prices()

    column_names = create_df_column_names()

    df = pd.DataFrame(columns=column_names)
    previous_prices = PreviousPriceHolder()

    for price in fiat_crypto_prices:
        df_dict = create_df_row_dict(previous_prices, price)
        row_df = pd.DataFrame(df_dict, index=[0])
        df = df.append(row_df)


def create_df_row_dict(previous_prices, price):
    ta_dict = db.get_fiat_crypto_ta(price.id)
    df_dict = create_df_ta_dict(ta_dict)
    logger.debug('ta_dict for price with id: %d is %s', price.id, df_dict)

    sentiment = db.get_sentiment(price.period, price.crypto_code, price.date_time)
    df_dict['sentiment'] = sentiment
    logger.debug('sentiment for price with id: %d is %s', price.id, sentiment)

    change = get_previous_price(previous_prices, price)
    previous_prices.insert(price.period, price.fiat_code, price.crypto_code, price.close)
    df_dict['change'] = change

    return df_dict


def get_previous_price(previous_prices, price):
    previous_price = previous_prices.get(price.period, price.fiat_code, price.crypto_code)
    if previous_price:
        return 1 if previous_price - price.close > 0 else 0
    else:
        return 0


def create_df_ta_dict(ta_dict):
    df_dict = {}
    for ind in Indicator:
        ind_value = ind.value
        if ind_value in ta_dict:
            df_dict[ind_value] = ta_dict[ind_value]
        else:
            df_dict[ind_value] = None
    return df_dict


def create_df_column_names():
    columns = [ind.value for ind in Indicator]
    columns.append('SENTIMENT')
    columns.append('CHANGE')

    return columns


if __name__ == '__main__':
    main()
