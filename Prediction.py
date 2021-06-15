# NYC Prediction

# Importing Libraries
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
df = pd.read_csv('train.csv', nrows=500_000, parse_dates=['pickup_datetime'])
print(df.head())

print(df.dtypes)

description = df.describe()

# Missing values
print(df.isnull().sum())
# drop null values
df = df.dropna(how = 'any', axis = 'rows')

# Noticed fare amount is negative, Drop values
df = df[df.fare_amount>0]

description = df.describe()




## Exploring the Fare

df['fare_amount'].describe()

plt.hist(df.fare_amount, bins=100)
plt.show()

# aparently we have a few too many rich people in cabs

df.fare_amount[df.fare_amount>100].count()

plt.hist(df.fare_amount[df.fare_amount<100], bins=50)
plt.show()


# Removing the outliers
df = df[df.fare_amount<100]

# I am not sure how to explore the location data
"""
Latitudes range from -90 to 90.
Longitudes range from -180 to 180.
"""

imputer = df[((-180>df.pickup_longitude) | (-180>df.dropoff_longitude)
             |(-90>df.pickup_latitude) | (-90>df.dropoff_latitude))]

print(imputer.count())

df = df.drop(imputer.index, axis=0)

imputer = df[((180<df.pickup_longitude) | (180<df.dropoff_longitude)
             |(90<df.pickup_latitude) | (90<df.dropoff_latitude))]

print(imputer.count())

df = df.drop(imputer.index, axis=0)

df = df[(df.pickup_longitude != df.dropoff_longitude) & (df.pickup_latitude != df.dropoff_latitude)]

"""
What is the area range of new york
VERY IMPORTANT
"""






## Exploring passenger data

df.passenger_count.describe()

# OwO. okay something is wrong, 208 passengers?

e = df[df.passenger_count == 208]

# the drop-off location and pickup locations are same while 208 passengers are
# riding who were charged 3 dollars...

df = df[df.passenger_count > 0]

df.passenger_count.describe()

description = df.describe()

# This looks much better

# Alright spliting the date time into useful sections and finding distance of travel
# I suppose i can use haversine formula

def haversine_distance(lat1, long1, lat2, long2):
        R = 6371  #radius of earth in kilometers
        phi1 = np.radians(df[lat1])
        phi2 = np.radians(df[lat2])
    
        delta_phi = np.radians(df[lat2]-df[lat1])
        delta_lambda = np.radians(df[long2]-df[long1])
    
        #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
        #c = 2 * atan2( √a, √(1−a) )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
        #d = R*c
        d = (R * c) #in kilometers
        df['H_Distance'] = d
        return d
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')


# now for the date time

df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['date'] = df['pickup_datetime'].dt.day
df['day'] = df['pickup_datetime'].dt.dayofweek
df['hour'] = df['pickup_datetime'].dt.hour

description = df.describe()
# Does the number of passengers affect the fare?

plt.scatter(df.passenger_count, df.fare_amount, alpha=0.01)
plt.xlabel('Numebr of passengers')
plt.ylabel('Amount paid')
plt.show()

sns.boxplot(x='passenger_count', y='fare_amount', data=df)

mean = []
for i in range(1,7):
    mean.append(df.fare_amount[df.passenger_count==i].mean())

plt.plot(range(1,7), mean)



# Does the Year of travel affect the fare?
# changing price over the years?

plt.scatter(df.year, df.fare_amount, alpha=0.1)
plt.xlabel('Year')
plt.ylabel('Amount paid')
plt.show()


'''
No real general trend, 09to11 there us a decrease
but a overall increase form 2011 to 2015 with 2015 having the highest numbers of outliers
'''

e = df[df.year == 2015]
ed = e.describe()

e2 = df[df.year == 2011]
e2d = e2.describe()

'''
it sure is costlier
'''


# Does the month of travel affet the fare?

plt.scatter(df.month, df.fare_amount)
plt.xlabel('month')
plt.ylabel('Amount paid')
plt.show()

# nope

# Does the date or day?

plt.scatter(df.date, df.fare_amount)
plt.xlabel('date')
plt.ylabel('Amount paid')
plt.show()

sns.boxplot(x='date', y='fare_amount', data=df)


mean = []
for i in range(1,32):
    mean.append(df.fare_amount[df.date==i].mean())
plt.plot(range(1,32), mean)


plt.scatter(df.day, df.fare_amount, alpha=0.01)
plt.xlabel('day of week')
plt.ylabel('Amount paid')
plt.show()

sns.boxplot(x='day', y='fare_amount', data=df)
            

e = df[df.day == 1]
ed = e.describe()

e2 = df[df.day == 6]
e2d = e2.describe()

'''
The weekends might have people who travel farther
update: no.
'''

# What about hour of day

plt.scatter(df.hour[df.day>1], df.fare_amount[df.day>1])
plt.xlabel('Hour')
plt.ylabel('Amount paid')
plt.show()

sns.boxplot(x='hour', y='fare_amount', data=df)

mean = []
for i in range(0, 24):
    mean.append(df.fare_amount[df.hour==i].mean())
plt.plot(range(0,24), mean)

'''
Nights and early mornings are costlier
'''



# Is there base and additional fare?
# The fare must be linearly increasing with distance, so in y = mx + c what is the c?

df.H_Distance[df.H_Distance<100].hist(bins=100)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()

# Most people travel less than 10 20 Kms

df.H_Distance[df.H_Distance>30].count()

imputer = df.loc[(df.H_Distance>30)]
df = df.drop(imputer.index, axis=0)



# What is the minimum (non outlying) base fare?

e = df[['H_Distance', 'fare_amount']][df.H_Distance>0].sort_values('fare_amount', ascending=True)
ed  = e.head(100)

ed = e.head(1000).mean()

# The average base fare is about 2.5$




# While working through that data, many entries were close to 0 distance. Lets fix that

e = df[['H_Distance', 'fare_amount']][df.H_Distance<1]

# hmm... Now that we know the base fare, lets create a threshold and impute bad data

imputer = df.loc[((df.H_Distance<0.3) & (df.fare_amount>5))]
ed = imputer[['H_Distance', 'fare_amount']].head(100)

df = df.drop(imputer.index, axis=0)

# Removing more bad data
imputer = df.loc[(df.fare_amount<2.5)]
imputer.count()

df = df.drop(imputer.index, axis=0)

description = df.describe()


e = df[df.fare_amount == df.fare_amount.max()]

df = df[((df.pickup_longitude < 0) & (df.dropoff_longitude < 0))]

imputer = df[((df.pickup_latitude < 39) | (df.dropoff_latitude < 39)
              |(df.pickup_latitude > 42) | (df.dropoff_latitude > 42))]

df = df.drop(imputer.index, axis=0)


e = df.head(100)



















































"""
CLEANING ONLY SECTION
"""

print(df.isnull().sum())
df = df.dropna(how = 'any', axis = 'rows')

df = df[df.fare_amount>0]

df = df[df.fare_amount<100]

imputer = df[((-180>df.pickup_longitude) | (-180>df.dropoff_longitude)
             |(-90>df.pickup_latitude) | (-90>df.dropoff_latitude))]

print(imputer.count())

df = df.drop(imputer.index, axis=0)

imputer = df[((180<df.pickup_longitude) | (180<df.dropoff_longitude)
             |(90<df.pickup_latitude) | (90<df.dropoff_latitude))]

print(imputer.count())

df = df.drop(imputer.index, axis=0)

df = df[(df.pickup_longitude != df.dropoff_longitude) & (df.pickup_latitude != df.dropoff_latitude)]

df = df[df.passenger_count > 0]







def haversine_distance(lat1, long1, lat2, long2):
        R = 6371  #radius of earth in kilometers
        phi1 = np.radians(df[lat1])
        phi2 = np.radians(df[lat2])
    
        delta_phi = np.radians(df[lat2]-df[lat1])
        delta_lambda = np.radians(df[long2]-df[long1])
    
        #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
        #c = 2 * atan2( √a, √(1−a) )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
        #d = R*c
        d = (R * c) #in kilometers
        df['H_Distance'] = d
        return d
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')


# now for the date time

df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['date'] = df['pickup_datetime'].dt.day
df['day'] = df['pickup_datetime'].dt.dayofweek
df['hour'] = df['pickup_datetime'].dt.hour







imputer = df.loc[(df.H_Distance>30)]
df = df.drop(imputer.index, axis=0)

imputer = df.loc[((df.H_Distance<0.3) & (df.fare_amount>5))]
df = df.drop(imputer.index, axis=0)

imputer = df.loc[(df.fare_amount<2.5)]
imputer.count()

df = df.drop(imputer.index, axis=0)

df = df[((df.pickup_longitude < 0) & (df.dropoff_longitude < 0))]

imputer = df[((df.pickup_latitude < 39) | (df.dropoff_latitude < 39)
              |(df.pickup_latitude > 42) | (df.dropoff_latitude > 42))]

df = df.drop(imputer.index, axis=0)








"""
Adding the diff between long lat columns
"""

df['diff_long'] = abs(df['pickup_longitude'] - df['dropoff_longitude'])
df['diff_lat'] = abs(df['pickup_latitude'] - df['dropoff_latitude'])



"""

Adding aditional information

"""




import re

def extract_dateinfo(df, date_col, drop=True, time=False, 
                     start_ref = pd.datetime(1900, 1, 1),
                     extra_attr = False):
    """
    Extract Date (and time) Information from a DataFrame
    Adapted from: https://github.com/fastai/fastai/blob/master/fastai/structured.py
    """
    df = df.copy()
    
    # Extract the field
    fld = df[date_col]
    
    # Check the time
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    # Convert to datetime if not already
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[date_col] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    

    # Prefix for new columns
    pre = re.sub('[Dd]ate', '', date_col)
    pre = re.sub('[Tt]ime', '', pre)
    
    # Basic attributes
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Days_in_month', 'is_leap_year']
    
    # Additional attributes
    if extra_attr:
        attr = attr + ['Is_month_end', 'Is_month_start', 'Is_quarter_end', 
                       'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    
    # If time is specified, extract time information
    if time: 
        attr = attr + ['Hour', 'Minute', 'Second']
        
    # Iterate through each attribute
    for n in attr: 
        df[pre + n] = getattr(fld.dt, n.lower())
        
    # Calculate days in year
    df[pre + 'Days_in_year'] = df[pre + 'is_leap_year'] + 365
        
    if time:
        # Add fractional time of day (0 - 1) units of day
        df[pre + 'frac_day'] = ((df[pre + 'Hour']) + (df[pre + 'Minute'] / 60) + (df[pre + 'Second'] / 60 / 60)) / 24
        
        # Add fractional time of week (0 - 1) units of week
        df[pre + 'frac_week'] = (df[pre + 'Dayofweek'] + df[pre + 'frac_day']) / 7
    
        # Add fractional time of month (0 - 1) units of month
        df[pre + 'frac_month'] = (df[pre + 'Day'] + (df[pre + 'frac_day'])) / (df[pre + 'Days_in_month'] +  1)
        
        # Add fractional time of year (0 - 1) units of year
        df[pre + 'frac_year'] = (df[pre + 'Dayofyear'] + df[pre + 'frac_day']) / (df[pre + 'Days_in_year'] + 1)
        
    # Add seconds since start of reference
    df[pre + 'Elapsed'] = (fld - start_ref).dt.total_seconds()
    
    if drop: 
        df = df.drop(date_col, axis=1)
        
    return df


ext_df = extract_dateinfo(df, 'pickup_datetime', drop = False, 
                         time = True, start_ref = df['pickup_datetime'].min())

















# Define distance
def dist(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude):
    pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude = map(np.radians, [pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude])
    dlon = dropoff_longitude - pickup_longitude
    dlat = dropoff_latitude - pickup_latitude
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_latitude) * np.cos(dropoff_latitude) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6367 * c
    return distance


















def transform(data):
    # Distances to nearby airports, city center and other counties
    # By reporting distances to these points, the model can somewhat triangulate other locations of interest
    
    # city center
    nyc = (-74.0060, 40.7128)
    
    # county
    Nassau = (-73.5594, 40.6546)
    Suffolk = (-72.6151, 40.9849)
    Westchester = (-73.7949, 41.1220)
    Rockland = (-73.9830, 41.1489)
    Dutchess = (-73.7478, 41.7784)
    Orange = (-74.3118, 41.3912)
    Putnam = (-73.7949, 41.4351) 

    # airport
    jfk = (-73.7781, 40.6413)
    ewr = (-74.1745, 40.6895)
    lgr = (-73.8740, 40.7769)
    
    
    # county
    data['pickup_distance_to_center'] = dist(nyc[0], nyc[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_center'] = dist(nyc[0], nyc[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Nassau'] = dist(Nassau[0], Nassau[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Nassau'] = dist(Nassau[0], Nassau[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Suffolk'] = dist(Suffolk[0], Suffolk[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Suffolk'] = dist(Suffolk[0], Suffolk[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Westchester'] = dist(Westchester[0], Westchester[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Westchester'] = dist(Westchester[0], Westchester[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Rockland'] = dist(Rockland[0], Rockland[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Rockland'] = dist(Rockland[0], Rockland[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Dutchess'] = dist(Dutchess[0], Dutchess[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Dutchess'] = dist(Dutchess[0], Dutchess[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Orange'] = dist(Orange[0], Orange[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Orange'] = dist(Orange[0], Orange[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Putnam'] = dist(Putnam[0], Putnam[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Putnam'] = dist(Putnam[0], Putnam[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    # airports
    data['pickup_distance_to_jfk'] = dist(jfk[0], jfk[1],
                                         data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_jfk'] = dist(jfk[0], jfk[1],
                                           data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_ewr'] = dist(ewr[0], ewr[1], 
                                          data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_ewr'] = dist(ewr[0], ewr[1],
                                           data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_lgr'] = dist(lgr[0], lgr[1],
                                          data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_lgr'] = dist(lgr[0], lgr[1],
                                           data['dropoff_longitude'], data['dropoff_latitude'])
    
    # point distance
    data['distance'] = dist(data['pickup_longitude'], data['pickup_latitude'],
                            data['dropoff_longitude'], data['dropoff_latitude'])
    
    return data



ext_df = transform(ext_df)












def final_convert(df):
    # There is a 50-cent MTA State Surcharge for all trips that end in New York City or 
    # Nassau, Suffolk, Westchester, Rockland, Dutchess, Orange or Putnam Counties.
    # The following two variables can be merged into one.
    # The following only considers trips that starts in city center and ends in nearby counties,
    # while the opposite direction could also be considered counties
    df['county_dropoff_1'] = np.where((df['pickup_distance_to_center'] <= 5) &
                                     ((df['dropoff_distance_to_Nassau'] <= 21.3) |
                                      (df['dropoff_distance_to_Westchester'] <= 22.4)), 1, 0)
    
    df['county_dropoff_2'] = np.where((df['pickup_distance_to_center'] <= 5) &                  
                                     ((df['dropoff_distance_to_Suffolk'] <= 48.7) |           
                                      (df['dropoff_distance_to_Rockland'] <= 14.1) |
                                      (df['dropoff_distance_to_Dutchess'] <= 28.7) |
                                      (df['dropoff_distance_to_Orange'] <= 29) |
                                      (df['dropoff_distance_to_Putnam'] <= 15.7)), 1, 0)
    
    # There is a daily 50-cent surcharge from 8pm to 6am.
    df['night_hour'] = np.where((df['hour'] >= 20) |
                                (df['hour'] <= 6) , 1, 0)
    
    # There is a $1 surcharge from 4pm to 8pm on weekdays, excluding holidays.
    df['peak_hour'] = np.where((df['hour'] >= 16) &
                                (df['hour'] <= 20) & 
                                (df['day'] >=0) &
                                (df['day'] <=4) , 1, 0)
    
    # This is a flat fare of $52 plus tolls, the 50-cent MTA State Surcharge, the 30-cent Improvement Surcharge, 
    # to/from JFK and any location in Manhattan:
    df['to_from_jfk'] = np.where(((df['pickup_distance_to_jfk'] <= 2) & (df['dropoff_distance_to_center'] <= 5)) | 
                                 ((df['pickup_distance_to_center'] <= 5) & (df['dropoff_distance_to_jfk'] <= 2)) ,1, 0)

    # There is a $4.50 rush hour surcharge (4 PM to 8 PM weekdays, excluding legal holidays). o/from JFK and any location in Manhattan:
    df['jfk_rush_hour'] = np.where((df['to_from_jfk'] == 1) & 
                                   (df['hour'] >= 16) &
                                   (df['hour'] <= 20) ,1, 0)
    
    # There is a $17.50 Newark Surcharge to Newark Airport:
    df['ewr'] = np.where((df['pickup_distance_to_center'] <= 5) &
                         (df['dropoff_distance_to_ewr'] <= 1) ,1, 0)
    
    return df




ext_df = final_convert(ext_df)






"""
ONLY CLEANING SECTION
"""
















corr = df.corr()
sns.heatmap(corr)
plt.show()


corr = ext_df.corr()
sns.heatmap(corr)
plt.show()

relevant_cols = corr.fare_amount[abs(corr['fare_amount'])>0.1].index

cols = list(relevant_cols)
cols.remove('fare_amount')

y = df.iloc[:,1]
X = ext_df[relevant_cols]
X = X.iloc[:, 1:]




e = X.head()
e2 = df.head()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .25)


from sklearn.metrics import mean_squared_error, explained_variance_score as var_score, r2_score

def plot_prediction_analysis(y, y_pred, figsize=(10,4), title=''):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].scatter(y, y_pred)
    mn = min(np.min(y), np.min(y_pred))
    mx = max(np.max(y), np.max(y_pred))
    axs[0].plot([mn, mx], [mn, mx], c='red')
    axs[0].set_xlabel('$y$')
    axs[0].set_ylabel('$\hat{y}$')
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    evs = var_score(y, y_pred)
    axs[0].set_title('rmse = {:.2f}, evs = {:.2f}'.format(rmse, evs))
    
    axs[1].hist(y-y_pred, bins=50)
    avg = np.mean(y-y_pred)
    std = np.std(y-y_pred)
    axs[1].set_xlabel('$y - \hat{y}$')
    axs[1].set_title('Histrogram prediction error, $\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(avg, std))
    
    if title!='':
        fig.suptitle(title)





"""
# Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import explained_variance_score as var_score, r2_score

print(var_score(y_test, y_pred))
print(r2_score(y_test, y_pred))

plot_prediction_analysis(y_test, y_pred, title='Decision-Tree')





# Simple Linear Regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_lin = Pipeline((
        ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    ))
model_lin.fit(X_train, y_train)

y_train_pred = model_lin.predict(X_train)
plot_prediction_analysis(y_train, y_train_pred, title='Linear Model - Trainingset')

y_test_pred = model_lin.predict(X_test)
plot_prediction_analysis(y_test, y_test_pred, title='Linear Model - Testset')
"""






"""
Hyper Parameter Tuning
"""


from sklearn.model_selection import RandomizedSearchCV

# Hyperparameter grid
param_grid = {
    'n_estimators': np.linspace(10, 100).astype(int),
    'max_depth': [None] + list(np.linspace(5, 30).astype(int)),
    'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
    'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}








# Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()




rs = RandomizedSearchCV(regressor, param_grid, n_jobs = -1, 
                        scoring = 'neg_root_mean_squared_error', cv = 3, 
                        n_iter = 100, verbose = 1)



rs.fit(X_train, y_train)

model = rs.best_estimator_
"""
RandomForestRegressor(bootstrap=False, max_depth=23, max_features=0.5, max_leaf_nodes=40,
                      min_samples_split=10, n_estimators=96, n_jobs=-1)
"""
model.n_jobs = -1


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(var_score(y_test, y_pred))
print(r2_score(y_test, y_pred))

plot_prediction_analysis(y_test, y_pred, title='Random Forest')

import lightgbm as lgbm

params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': -1,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': 23,
        'subsample': 0.8,
        'bagging_fraction' : 0.99,
        'max_bin' : 5000,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'num_rounds':50000
    }

def LGBMmodel(X_train,X_test,y_train,y_test,params):
    matrix_train = lgbm.Dataset(X_train, y_train)
    matrix_test = lgbm.Dataset(X_test, y_test)
    model=lgbm.train(params=params,
                    train_set=matrix_train,
                    num_boost_round=100000, 
                    early_stopping_rounds=500,
                    verbose_eval=100,
                    valid_sets=matrix_test)
    return model



model = LGBMmodel(X_train,X_test,y_train,y_test,params)

y_pred = model.predict(X_test, num_iteration = model.best_iteration)


print(var_score(y_test, y_pred))
print(r2_score(y_test, y_pred))

plot_prediction_analysis(y_test, y_pred, title='Random Forest lgbm')








df = pd.read_csv("test.csv", parse_dates=['pickup_datetime'])
key = df.key

haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['date'] = df['pickup_datetime'].dt.day
df['day'] = df['pickup_datetime'].dt.dayofweek
df['hour'] = df['pickup_datetime'].dt.hour
df['diff_long'] = abs(df['pickup_longitude'] - df['dropoff_longitude'])
df['diff_lat'] = abs(df['pickup_latitude'] - df['dropoff_latitude'])
ext_df = extract_dateinfo(df, 'pickup_datetime', drop = False, 
                         time = True, start_ref = df['pickup_datetime'].min())
ext_df = transform(ext_df)
ext_df = final_convert(ext_df)




test = ext_df[cols]

prediction = model.predict(test, num_iteration = model.best_iteration)
prediction = prediction.reshape(-1 , 1)
key = key.values.reshape(-1, 1)
answer = np.append(arr=key, values=prediction, axis=1)

answer = pd.DataFrame(answer, columns=['key', 'fare_amount'])

answer.to_csv('Prediction.csv', index= None)


"""
df_train = np.concatenate((X_train, y_train.reshape((-1,1))),axis = 1 )
df_test = np.concatenate((X_test,y_test.reshape((-1,1))),axis = 1)

df_train = pd.DataFrame(df_train)
df_test = pd.DataFrame(df_test)

df_test.columns = ['pickup_datetime', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count', 'H_Distance', 'year', 'month', 'date', 'day',
       'hour','fare_amount']

df_train.columns = ['pickup_datetime', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count', 'H_Distance', 'year', 'month', 'date', 'day',
       'hour','fare_amount']

df_test.to_csv('df_test.csv',index = None)
df_train.to_csv('df_train.csv',index = None)
"""