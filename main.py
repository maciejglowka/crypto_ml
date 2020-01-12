import logging.config
import time
import math

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import db
import log
import matplotlib.pyplot as plt
from model.Indicator import Indicator
from model.PreviousPriceHolder import PreviousPriceHolder
from joblib import dump, load

logging.config.dictConfig(log.LOGGING_CONF)
logger = logging.getLogger("main")


def main():
    crypto = 'BTC'
    fiat = 'EUR'
    period = 900
    sentiment_switch = False

    if sentiment_switch:
        filename = crypto + '_' + fiat + '_' + str(period)
    else:
        filename = crypto + '_' + fiat + '_' + str(period) + '_no_sentiment'

    logger.info("starting ml data fetch for crypto: %s, fiat: %s, period: %s", crypto, fiat, period)

    count = db.count_fiat_crypto_prices(crypto, fiat, period)
    logger.info('fetched %s prices', count)

    fiat_crypto_prices = db.get_fiat_crypto_prices(crypto, fiat, period)
    column_names = create_df_column_names()

    all_metrics_df = create_all_metrics_df(column_names, fiat_crypto_prices)

    logger.info('starting data preprocessing')
    pre_time = time.time()

    imputer = SimpleImputer(strategy="mean")
    model = ExtraTreesClassifier(n_estimators=100)
    # X = all_metrics_df.iloc[:, 0:28]
    # with sentiment
    if sentiment_switch:
        X = all_metrics_df.iloc[:, 0:29]
    else:
        X = all_metrics_df.iloc[:, 0:28]

    X_copy = X.copy()
    y = all_metrics_df.iloc[:, -1].astype('int32')

    imputer.fit(X)
    X = imputer.transform(X)
    dump(imputer, filename + '_imputer.joblib')

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    dump(scaler, filename + '_scaler.joblib')

    model.fit(X, y)

    logger.info('data preprocessing finished, time: %0.3f seconds', time.time() - pre_time)

    features = get_features_for_classifier(X_copy, model.feature_importances_, 14, filename + '.png')

    logger.info('selected features: %s', features)

    svc_classifier = SVC(kernel='rbf')
    C_range = [1, 10, 100, 1000]
    gamma_range = [1e-3, 1e-4]
    param_grid = dict(gamma=gamma_range, C=C_range)
    # df_model = pd.DataFrame(data=X, columns=column_names[0:28])
    # with sentiment
    if sentiment_switch:
        df_model = pd.DataFrame(data=X, columns=column_names[0:29])
    else:
        df_model = pd.DataFrame(data=X, columns=column_names[0:28])

    df_model_feature = df_model[features]
    train_count = math.floor(count * 0.9)
    test_count = count - train_count
    logger.info('split values: %s to %s', train_count, test_count)

    y_train_split = y.head(train_count)
    X_train_split = df_model_feature.iloc[:train_count, :]
    y_test_split = y.tail(test_count)
    X_test_split = df_model_feature.iloc[train_count:, :]

    # X_train, X_test, y_train, y_test = train_test_split(X_train_split, y_train_split, test_size=0.10)
    # X_train, X_test, y_train, y_test = train_test_split(df_model[features], y, test_size=0.10)

    clf = GridSearchCV(svc_classifier, param_grid, cv=5)

    logger.info('classifier fitting starting')
    fit_time = time.time()

    clf.fit(X_train_split, y_train_split)

    logger.info('classifier fitting finished, time: %0.3f seconds', time.time() - fit_time)

    logger.info("The best parameters are %s with a score of %0.2f", clf.best_params_, clf.best_score_)

    logger.info('starting predictions')
    predict_time = time.time()
    y_true, y_pred = y_test_split, clf.predict(X_test_split)
    logger.info('predictions ended, time: %0.3f seconds', time.time() - predict_time)

    logger.info(classification_report(y_true, y_pred))

    dump(clf, filename + '_clf.joblib')


def get_features_for_classifier(X_copy, features, n, filename):
    feature_importances = pd.Series(features, index=X_copy.columns)
    n_important_features = feature_importances.nlargest(n)

    feature_importances.plot(kind='barh', fontsize=5, rot=45)
    plt.savefig(filename)

    return n_important_features.index.tolist()


def create_all_metrics_df(column_names, fiat_crypto_prices):
    all_metrics_df = pd.DataFrame(columns=column_names)

    previous_prices = PreviousPriceHolder()
    for price in fiat_crypto_prices:
        df_dict = create_df_row_dict(previous_prices, price)
        row_df = pd.DataFrame(df_dict, index=[0])
        all_metrics_df = all_metrics_df.append(row_df, sort=False, ignore_index=True)
    return all_metrics_df


def create_df_row_dict(previous_prices, price):
    ta_dict = db.get_fiat_crypto_ta(price.id)
    df_dict = create_df_ta_dict(ta_dict)
    logger.debug('ta_dict for price with id: %d is %s', price.id, df_dict)

    sentiment = db.get_sentiment(price.period, price.crypto_code, price.date_time)
    df_dict['SENTIMENT'] = sentiment
    logger.debug('sentiment for price with id: %d is %s', price.id, sentiment)

    change = get_previous_price(previous_prices, price)
    df_dict['CHANGE'] = change

    previous_prices.insert(price.period, price.fiat_code, price.crypto_code, price.close)

    return df_dict


def get_previous_price(previous_prices, price):
    previous_price = previous_prices.get(price.period, price.fiat_code, price.crypto_code)
    if previous_price:
        return 0 if previous_price - price.close > 0 else 1
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
