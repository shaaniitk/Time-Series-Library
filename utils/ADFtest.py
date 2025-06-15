import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import ADF
from utils.logger import logger

def calculate_ADF(root_path,data_path):
    logger.info(f"Calculating ADF for all columns in {data_path}")
    try:
        df_raw = pd.read_csv(os.path.join(root_path,data_path))
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise
    cols = list(df_raw.columns)
    cols.remove('date')
    df_raw = df_raw[cols]
    adf_list = []
    for i in cols:
        df_data = df_raw[i]
        try:
            adf = adfuller(df_data, maxlag = 1)
        except Exception as e:
            logger.error(f"ADF test failed for column {i}: {e}")
            adf = None
        print(adf)
        adf_list.append(adf)
    return np.array(adf_list)

def calculate_target_ADF(root_path,data_path,target='OT'):
    logger.info(f"Calculating ADF for target(s) {target} in {data_path}")
    try:
        df_raw = pd.read_csv(os.path.join(root_path,data_path))
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise
    target_cols = target.split(',')
    df_raw = df_raw[target_cols]
    adf_list = []
    for i in target_cols:
        df_data = df_raw[i]
        try:
            adf = adfuller(df_data, maxlag = 1)
        except Exception as e:
            logger.error(f"ADF test failed for column {i}: {e}")
            adf = None
        adf_list.append(adf)
    return np.array(adf_list)

def archADF(root_path, data_path):
    logger.info(f"Calculating ARCH ADF for {data_path}")
    try:
        df = pd.read_csv(os.path.join(root_path,data_path))
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise
    cols = df.columns[1:]
    stats = 0
    for target_col in cols:
        series = df[target_col].values
        try:
            adf = ADF(series)
        except Exception as e:
            logger.error(f"ARCH ADF failed for column {target_col}: {e}")
            adf = None
        stat = adf.stat
        stats += stat
    return stats/len(cols)

def adf_test(series):
    logger.info("Running ADF test")
    result = adfuller(series)
    return result

if __name__ == '__main__':

    # * Exchange - result: -1.902402344564288 | report: -1.889
    ADFmetric = archADF(root_path="./dataset/exchange_rate/",data_path="exchange_rate.csv")
    print("Exchange ADF metric", ADFmetric)

    # * Illness - result: -5.33416661870624 | report: -5.406
    ADFmetric = archADF(root_path="./dataset/illness/",data_path="national_illness.csv") 
    print("Illness ADF metric", ADFmetric)

    # * ETTm2 - result: -5.663628743471695 | report: -6.225
    ADFmetric = archADF(root_path="./dataset/ETT-small/",data_path="ETTm2.csv")
    print("ETTm2 ADF metric", ADFmetric)

    # * Electricity - result: -8.44485821939281 | report: -8.483
    ADFmetric = archADF(root_path="./dataset/electricity/",data_path="electricity.csv")
    print("Electricity ADF metric", ADFmetric)

    # * Traffic - result: -15.020978067839014 | report: -15.046
    ADFmetric = archADF(root_path="./dataset/traffic/",data_path="traffic.csv")
    print("Traffic ADF metric", ADFmetric)

    # * Weather - result: -26.681433085204866 | report: -26.661
    ADFmetric = archADF(root_path="./dataset/weather/",data_path="weather.csv")
    print("Weather ADF metric", ADFmetric)


    # print(ADFmetric)

    # mean_ADFmetric = ADFmetric[:,0].mean()
    # print(mean_ADFmetric)