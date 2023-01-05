import kfp
from kfp.components import func_to_container_op, InputPath, OutputPath


def read_csv(
        data_url: str,
        data_csv_path: OutputPath(str)):
    """ reading csv data form url"""
    import pandas as pd
    data = pd.read_csv(data_url)
    data.to_csv(data_csv_path, index=None)


read_csv_op = func_to_container_op(func=read_csv,
                                   base_image='python:3.9',
                                   packages_to_install=['pandas'],
                                   output_component_file='../deployment_pipeline/read_csv.yaml')


def process_data(
        data_csv_path: InputPath(str),
        categorical_columns: list,
        processed_data_path: OutputPath(str),
        one_hot_enc_model_path: OutputPath(str),
        saved_model_path: InputPath(str) = None,
        training: bool = True):
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    import holidays
    import numpy as np
    import pickle

    # get data from upstream component
    data = pd.read_csv(data_csv_path)

    def process_dates(data):
        # dates preprocessing
        data["year"] = data.day_id.dt.year
        data["month"] = data.day_id.dt.month
        data["week"] = data.day_id.dt.isocalendar().week
        data["quarter"] = data.day_id.dt.quarter
        # #either a day in the weekly turnover belongs to a holiday
        data["is_holiday"] = is_holiday_week(data).astype(int)

    def one_hot_encoding(data, categorical_columns, training=True):
        """add one hot encoding of categorical columns"""
        if training:
            ohe = OneHotEncoder()
            one_hot_encoded_data = ohe.fit_transform(data[categorical_columns])
        else:
            with open(saved_model_path, 'rb') as f:
                ohe = pickle.load(f)
            one_hot_encoded_data = ohe.transform(data[categorical_columns])

        # save one_hot encoding model
        with open(one_hot_enc_model_path, "wb") as f:
            pickle.dump(ohe, f)

        one_hot_df = pd.DataFrame(one_hot_encoded_data.toarray(),
                                  columns=ohe.get_feature_names_out(),
                                  index=data.index)
        return one_hot_df

    def is_holiday_week(data):
        # get holiday dates in France from 2012 to 2017
        holidays_france = pd.DataFrame(
            holidays.France(years=range(2012, 2018)).keys(),
            dtype="datetime64[ns]",
            columns=["holiday_date"])

        # make a tuple of (year, week) key
        holidays_france["year"] = holidays_france["holiday_date"].dt.year
        holidays_france["week"] = holidays_france["holiday_date"].dt.isocalendar().week
        year_week_tuple = list(holidays_france[["year", "week"]].itertuples(index=False, name=None))

        # check each row in the data if it belongs to (year, week of the year) tuple
        series = pd.Series(list(zip(data.year, data.week)), index=data.index).isin(year_week_tuple)

        return series

    def add_turnover_lags(data, time_lag=4):
        """add historical data of the last time_lag year"""

        # add an empty columns to fill lags
        for i in range(time_lag):
            lag = np.empty(data.shape[0])
            lag[:] = np.nan
            data["turnover_N-{}".format(i + 1)] = lag

        # get the list of departments and stores
        business_units_list = data.but_num_business_unit.unique()
        department_list = data.dpt_num_department.unique()

        # ingest lags by store and by department
        for i in business_units_list:
            for j in department_list:
                for k in range(1, time_lag + 1):
                    lag_data = data.loc[
                        (data.but_num_business_unit == i) & (data.dpt_num_department == j), "turnover"].shift(
                        -52 * k)
                    if lag_data.shape != 0:
                        data.loc[lag_data.index, "turnover_N-{}".format(k)] = lag_data

    # set day_id adequate type
    data["day_id"] = pd.to_datetime(data["day_id"], infer_datetime_format=True)
    # sort data by day_id
    data.sort_values("day_id", ascending=False, inplace=True)

    # # add time lags
    add_turnover_lags(data, time_lag=4)

    if not training:
        data = data[data['turnover'].isna()]
        data.drop(columns=['turnover'], inplace=True)

    # process dates
    process_dates(data)

    one_hot_encoded_data = one_hot_encoding(data, categorical_columns, training)
    # drop old categorical columns
    data.drop(columns=categorical_columns, inplace=True)
    # concat with the one hot encoded ones
    data = pd.concat([data, one_hot_encoded_data], axis=1)
    data.to_csv(processed_data_path, index=False)


process_data_op = func_to_container_op(func=process_data, base_image='python:3.9',
                                       packages_to_install=['pandas', 'scikit-learn', 'numpy', 'holidays'],
                                       output_component_file='../deployment_pipeline/process_data.yaml')


def split_data(
        processed_data_path: InputPath(str),
        X_train_path: OutputPath(str),
        y_train_path: OutputPath(str),
        X_eval_path: OutputPath(str),
        y_eval_path: OutputPath(str)):
    import pandas as pd
    from datetime import datetime, date

    processed_data = pd.read_csv(processed_data_path)
    # set day_id adequate type
    processed_data["day_id"] = pd.to_datetime(processed_data["day_id"], infer_datetime_format=True)

    # train on all data except the last month
    train_idx = processed_data.day_id.dt.date <= datetime(year=2017, month=8, day=31).date()

    X = processed_data.drop(labels=['turnover', "day_id"], axis=1)
    y = processed_data['turnover']

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_eval, y_eval = X.loc[~train_idx], y.loc[~train_idx]

    X_train.to_csv(X_train_path)
    y_train.to_csv(y_train_path)
    X_eval.to_csv(X_eval_path)
    y_eval.to_csv(y_eval_path)


split_data_op = func_to_container_op(func=split_data, base_image='python:3.9',
                                     packages_to_install=['pandas', 'datetime'],
                                     output_component_file='../deployment_pipeline/split_data.yaml')


def train_model(
        x_train_path: InputPath(str),
        y_train_path: InputPath(str),
        model_path: OutputPath(str)):
    import xgboost as xgb
    import pandas as pd

    X_train = pd.read_csv(x_train_path, index_col=0)
    y_train = pd.read_csv(y_train_path, index_col=0)

    xgb_reg = xgb.XGBRegressor(n_estimators=1, n_jobs=-1, max_depth=20, verbosity=1, random_state=42)
    xgb_reg.fit(X_train.astype(float), y_train)
    xgb_reg.save_model(model_path)


train_model_op = func_to_container_op(func=train_model, base_image='python:3.9',
                                      packages_to_install=['pandas', 'xgboost', 'scikit-learn'],
                                      output_component_file='../deployment_pipeline/train_model.yaml')


def eval_trained_model(
        x_eval_path: InputPath(str),
        y_eval_path: InputPath(str),
        model_path: InputPath(str)):
    import xgboost as xgb
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    X_eval = pd.read_csv(x_eval_path, index_col=0)
    y_eval = pd.read_csv(y_eval_path, index_col=0)

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    pred = model.predict(X_eval.astype(float))
    print("MAE: ", mean_absolute_error(y_eval, pred))


eval_trained_model_op = func_to_container_op(func=eval_trained_model, base_image='python:3.9',
                                             packages_to_install=['pandas', 'xgboost', 'scikit-learn'],
                                             output_component_file='../deployment_pipeline/eval_trained_model.yaml')


def pre_process_serving_data(
        serving_data_path: InputPath(str),
        historical_data_path: InputPath(str),
        preprocessed_serving_data_path: OutputPath(str)):
    """get historical data from data_train"""
    import numpy as np
    import pandas as pd

    historical_data = pd.read_csv(historical_data_path)
    serving_data = pd.read_csv(serving_data_path)

    # add an empty turnover column
    empty_column = np.empty(serving_data.shape[0])
    empty_column[:] = np.nan
    serving_data["turnover"] = empty_column

    preprocessed_serving_data = pd.concat([historical_data, serving_data], axis=0, ignore_index=True)
    preprocessed_serving_data.to_csv(preprocessed_serving_data_path, index=False)


preprocess_serving_data_op = func_to_container_op(func=pre_process_serving_data, base_image='python:3.9',
                                                  packages_to_install=['pandas', 'numpy'],
                                                  output_component_file='../deployment_pipeline/preprocess_serving_data.yaml')


def make_predictions(
        x_test_path: InputPath(str),
        model_path: InputPath(str)):
    import xgboost as xgb
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    X_test = pd.read_csv(x_test_path)
    X_test = X_test.drop(columns=['day_id'])

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    pred = model.predict(X_test.astype(float))
    print(pred)
    # pred = pd.DataFrame(pred, index=X_test, columns=['turnover_prediction'])


make_prediction_op = func_to_container_op(func=make_predictions, base_image='python:3.9',
                                          packages_to_install=['pandas', 'xgboost', 'scikit-learn'],
                                          output_component_file='../deployment_pipeline/make_prediction.yaml')


def decathlon_deployment_pipeline(
        data_train_url,
        data_serve_url,
        categorical_columns,
):
    read_train_csv_task = read_csv_op(data_url=data_train_url)

    process_train_data_task = process_data_op(
        read_train_csv_task.outputs['data_csv'],
        categorical_columns=categorical_columns,
        training=True)

    split_data_task = split_data_op(
        process_train_data_task.outputs['processed_data'])

    train_model_task = train_model_op(
        split_data_task.outputs["x_train"],
        split_data_task.outputs["y_train"])

    eval_trained_model_op(
        split_data_task.outputs['x_eval'],
        split_data_task.outputs['y_eval'],
        train_model_task.outputs['model'])

    read_serving_csv_task = read_csv_op(data_url=data_serve_url)

    preprocess_serving_data_task = preprocess_serving_data_op(
        read_serving_csv_task.outputs['data_csv'],
        read_train_csv_task.outputs['data_csv'])

    process_serving_data_task = process_data_op(
        data_csv=preprocess_serving_data_task.outputs['preprocessed_serving_data'],
        categorical_columns=categorical_columns,
        saved_model=process_train_data_task.outputs['one_hot_enc_model'],
        training=False)

    # post_process_serving_data_op(preprocessed_serving_data=process_serving_data_task.outputs['processed_data'])
    make_prediction_op(
        process_serving_data_task.outputs['processed_data'],
        train_model_task.outputs['model'])


if __name__ == '__main__':
    arguments = {
        'data_train_url': 'https://github.com/seifou23i/decathlon_challenge/raw/master/data/inputs/train/train.csv',
        'data_serve_url': 'https://github.com/seifou23i/decathlon_challenge/raw/master/data/inputs/test/test.csv',
        'categorical_columns': ["dpt_num_department", "but_num_business_unit", "year", "month", "week", "quarter"]}

    kfp.Client(host="http://localhost:8080").create_run_from_pipeline_func(decathlon_deployment_pipeline,
                                                                           arguments=arguments)
