# Enerjisa Hackathon - Data Ingestion
# 18.04.2022
# Author: Ali Baris Bilen

import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import math

data_path = pathlib.Path('C:/Users/asus/Desktop/kaggle/enerjisa/')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)

# 1) power.csv

power_df = pd.read_csv(data_path / 'power.csv')

power_df.head(3)
power_df.describe()
power_df.dtypes

power_df.isnull().values.any() # dataset is NA free

power_df.plot(figsize=(8,5)) # looks bad

power_df_agg_daily = power_df.set_index(pd.DatetimeIndex(power_df['Timestamp']))
power_df_agg_daily = power_df_agg_daily.resample('D').sum()

power_df_agg_daily.plot(figsize = (16, 5)) # seasonality is obvious

power_df.rename({'Power(kW)':'power'}, axis = 1, inplace = True)
power_df['power'].describe()

power_df['power'] = round(power_df['power'], 4)

power_df[power_df['power'] < 0] # 957 intervals have negative power
power_df_agg_daily[power_df_agg_daily['Power(kW)'] < 0] # only 17 days have negative power

# Negative power values will be kept as is.

# Negative values of power result from the necessity to cover the electricity demand of the power plant
# equipment (the so-called source’s own needs). This results in a change in the direction of electricity flow.
# Source: 'Impact of the Wind Turbine on the Parameters of the Electricity Supply to an Agricultural Farm'.

power_df_agg_weekly = power_df.set_index(pd.DatetimeIndex(power_df['Timestamp']))
power_df_agg_weekly = power_df_agg_weekly.resample('W').sum()

power_df_agg_weekly.plot(figsize = (16, 5)) # seasonality is obvious

# Convert 'Timestamp' to datetime for later use
power_df['Timestamp'] = pd.to_datetime(power_df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

power_df.dtypes

# ---

# 2) features.csv

features_df = pd.read_csv(data_path / 'features.csv') # 77 columns including timestamp

features_df.head(3)
features_df.describe()
#features_df.dtypes

features_df.isnull().values.any() # dataset contains NA values

display(features_df.isnull().sum()) # at least timestamp is NA free.

# We should impute them appropriately.

# Most of the features that contain NA values are observations like temperature, pressure or load
# kind of things. I don't expect them to change drastically within 10-min time intervals.

# One option might be to impute them with latest non-NA observation.

features_df.sort_values(by=['Timestamp'], inplace=True, ascending=True)
features_df.tail()

features_df[pd.isnull(features_df["Gearbox_T1_High_Speed_Shaft_Temperature"])].shape[0] # 4349 rows.
features_df[pd.isnull(features_df["Gearbox_T1_High_Speed_Shaft_Temperature"])]
features_df[(features_df["Timestamp"] > "2019-01-15 23:30:00") & (features_df["Timestamp"] < "2019-01-16 08:00:00")]

features_df_imputed_1 = features_df.fillna(method='ffill', axis=0)

features_df_imputed_1[(features_df_imputed_1["Timestamp"] > "2019-01-15 23:30:00") & (features_df_imputed_1["Timestamp"] < "2019-01-16 08:00:00")] # check.

# Another option might be to use interpolate() from pandas.
features_df_imputed_2 = features_df.interpolate(method='quadratic', axis=0) # -> quadratic interpolation

features_df_imputed_2[(features_df_imputed_2["Timestamp"] > "2019-01-15 23:30:00") & (features_df_imputed_2["Timestamp"] < "2019-01-16 08:00:00")].head(20) # check.

# --> quadratic does not make sense.

features_df_imputed_3 = features_df.interpolate(method='linear', axis=0) # -> linear interpolation

features_df_imputed_3[(features_df_imputed_3["Timestamp"] > "2019-01-15 23:30:00") & (features_df_imputed_3["Timestamp"] < "2019-01-16 08:00:00")].head(20) # check.

# linear is better I guess. quadratic method exaggerates the changes in the series.

# Observe imputations visually 

features_df[(features_df["Timestamp"] > "2019-01-15 23:30:00") & (features_df["Timestamp"] < "2019-01-16 23:00:00")].tail(10)

# here is exactly where NA values meet non-NA values.

features_df_imputed_3[(features_df_imputed_3["Timestamp"] > "2019-01-15 23:30:00") & (features_df_imputed_3["Timestamp"] < "2019-01-16 23:00:00")].tail(10)

# linear interpolation seems to perform far better than fillna(method=ffill) or simply imputing with mean.
# it better captures the trend in features.

display(features_df_imputed_3.isnull().sum())
features_df_imputed_3.isnull().values.any() # dataset is now NA free.

# ---

# 3) sample_submission.csv

sample_submission_df = pd.read_csv(data_path / 'sample_submission.csv')

sample_submission_df # 17,532 days in total, to be forecast
sample_submission_df.dtypes

sample_submission_df.isnull().values.any()

# write cleaned data
power_df.to_csv(path_or_buf = data_path / 'power_cleaned.csv', index=False,
                header=True)
features_df_imputed_3.to_csv(path_or_buf = data_path / 'features_cleaned.csv',
                             index=False, header=True)

# end
