# -*- coding: utf-8 -*-
"""

@author: Paolo
"""
## IMPORT APPROPRIATE MODELS ##
import numpy as np
import pandas as pd
# import plotly.express as px
import matplotlib.pyplot as plt
from statistics import mode
from sklearn import preprocessing, ensemble, linear_model, model_selection, metrics



## READ IN DATA ##
dataset = pd.read_excel("free_throws.xlsx")

## DATA CLEANING ##
## Downcast in order to save memory
def downcast(df):
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i,t in enumerate(types):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif t == np.object:
            if cols[i] == 'date':
                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
            else:
                df[cols[i]] = df[cols[i]].astype('category')
    return df

dataset = downcast(dataset)

## Check for mising values
print(dataset.apply(lambda x: sum(x.isnull()), axis=0))




## 1) DATA ANALYSIS ##
## Plot free throw % per season for each period for ALL players
dataset.groupby(["period", "season"]).mean()["shot_made"].unstack(0).plot.barh()

## Plot free throw count per season for each period for ALL players
dataset.groupby(["period", "season"]).count()["shot_made"].unstack(0).plot.barh()

## Plot average number of free throws made in each period over all seasons for ALL players
dataset_period = dataset.groupby("period").mean()
dataset_period["shot_made"] = dataset_period["shot_made"]*100
dataset_period["shot_made"].plot()
plt.xlabel("Period (above 4 is overtime)")
plt.ylabel("Average free throw %")


## List all players present in dataset
players = list(dataset["player"].unique())

## Define player, can change to an actual name of type str
name = players[100] ## can pick from available players

#name = input("Please input name of player to analyse ") ## can manually input name

## Check seasons that a player is active
active_seasons = dataset.loc[dataset["player"] == name, "season"].unique()
final_season = active_seasons[-1]
total_free_throws_attempts = dataset.loc[dataset["player"] == name, "shot_made"].value_counts().sum()
total_free_throws_made = dataset.loc[dataset["player"] == name, "shot_made"].value_counts()[1]

## all-time free throw percentage
all_time_free_throw_pct = 100*round(total_free_throws_made/total_free_throws_attempts, 3)
print(name + " all time free throw % = " + str(all_time_free_throw_pct))

## seasonal free throw percentage
dataset_seasonal = dataset.loc[dataset["player"] == name, :]
free_throw_made = dict.fromkeys(active_seasons, 0)
free_throw_pct = dict.fromkeys(active_seasons, 0)

for season in active_seasons:
    made_ft = dataset_seasonal.loc[(dataset_seasonal["shot_made"] == 1) & (dataset_seasonal["season"] == season), "shot_made"].count()
    free_throw_made[season] = made_ft
    free_throw_pct[season] = 100*round(made_ft/dataset_seasonal.loc[dataset_seasonal["season"] == season, "shot_made"].count(), 3)
  
print(name + " seasonal free made = " + str(free_throw_made))
print(name + " seasonal free throw % = " + str(free_throw_pct))

plt.figure()
x_made, y_made = zip(*free_throw_made.items())
plt.plot(x_made, y_made)
plt.xlabel("Season")
plt.ylabel("Free throws made")
plt.title(name + " seasonal free throws made")

plt.figure()
x_pct, y_pct = zip(*free_throw_pct.items())
plt.plot(x_pct, y_pct)
plt.xlabel("Season")
plt.ylabel("Free throw %")
plt.title(name + " seasonal free throw %")







## 3) MACHINE LEARNING ##
## Reset dataframe index
dataset_seasonal = dataset_seasonal.reset_index()

## Drop columns that won't be useful as features
dataset_seasonal = dataset_seasonal.drop(columns = ["game_id", "play", "player"])

## Define team player plays for
team = dict.fromkeys(active_seasons)
for season in active_seasons:
    team_season = []
    for i in dataset_seasonal.loc[dataset_seasonal["season"] == season].index.values:
        team_season.append(dataset_seasonal.loc[i, "game"][0:3])
        team_season.append(dataset_seasonal.loc[i, "game"][-3:])
    team[season] = mode(team_season) ## assume that the most occuring team name is the team

dataset_seasonal["team"] = dataset_seasonal["season"].map(team)

## Define two new features
dataset_seasonal["winning"] = 0 # -1 is losing, 1 is winning and 0 is drawing
dataset_seasonal["score_diff"] = 0 # difference between the current team of player and the opponent

## Processing the winning and score_diff features
for i in dataset_seasonal.index.values:
    if dataset_seasonal.loc[i, "game"][0:3] == dataset_seasonal.loc[i, "team"]:
        dataset_seasonal.loc[i, "Home_game"] = -1 ## IE THEY ARE AWAY
    else:
        dataset_seasonal.loc[i, "Home_game"] = 1 ## IE THEY ARE AT HOME
    
    away_score = [int(s) for s in dataset_seasonal.loc[i, "score"].split() if s.isdigit()][0]
    home_score = [int(s) for s in dataset_seasonal.loc[i, "score"].split() if s.isdigit()][1]
    if (away_score > home_score):
        if (dataset_seasonal.loc[i, "Home_game"] == -1):
            dataset_seasonal.loc[i, "winning"] = 1
            dataset_seasonal.loc[i, "score_diff"] = away_score - home_score
        elif (dataset_seasonal.loc[i, "Home_game"] == 1):
            dataset_seasonal.loc[i, "winning"] = -1
            dataset_seasonal.loc[i, "score_diff"] = home_score - away_score
    elif (away_score < home_score):
        if (dataset_seasonal.loc[i, "Home_game"] == -1):
            dataset_seasonal.loc[i, "winning"] = -1
            dataset_seasonal.loc[i, "score_diff"] = away_score - home_score
        elif (dataset_seasonal.loc[i, "Home_game"] == 1):
            dataset_seasonal.loc[i, "winning"] = 1
            dataset_seasonal.loc[i, "score_diff"] = home_score - away_score
    
## Drop features which are no longer needed and are incompatible with model
dataset_seasonal = dataset_seasonal.drop(columns = ["end_result", "game", "team", "score"])

## Use one hot encoding for playoffs category as it is currently incompatible with model
encoder = preprocessing.OneHotEncoder()
dataset_old_category = dataset_seasonal[["playoffs"]]
dataset_seasonal = dataset_seasonal.drop(columns = ["playoffs"])
dataset_category = encoder.fit_transform(dataset_old_category)
dataset_category = pd.DataFrame(dataset_category.toarray())
dataset_category.columns = encoder.get_feature_names()
dataset_seasonal = pd.concat([dataset_seasonal, dataset_category], axis = 1)
dataset_seasonal = dataset_seasonal.drop(columns = ['index'])

## Change time column to appropriate format for model input
def change_time_format(df):
    time = df["time"].hour * 3600 + df["time"].minute * 60 + df["time"].second
    return time

dataset_seasonal["time"] = dataset_seasonal.apply(change_time_format, axis = 'columns')

## Pick season to predict
season_number = 2
season_to_predict = active_seasons[season_number]
season_to_train = active_seasons[0:season_number] ## this increases depending on how far in the future we would want to predict

## Split data in training and testing sets
x_train = dataset_seasonal.loc[dataset_seasonal["season"].isin(season_to_train), :]
x_train = x_train.drop(columns = ["season", "shot_made"])
y_train = dataset_seasonal.loc[dataset_seasonal["season"].isin(season_to_train), "shot_made"]

x_test = dataset_seasonal.loc[dataset_seasonal["season"].isin(season_to_predict), :]
x_test = x_test.drop(columns = ["season", "shot_made"])
y_test = dataset_seasonal.loc[dataset_seasonal["season"].isin(season_to_predict), "shot_made"]

## Logistic model run
logit_model = linear_model.LogisticRegression(max_iter = 5000)
scores = model_selection.cross_val_score(logit_model, x_train, y_train, cv = 5)
print("Cross-validated logistic model scores:", scores) ## get an idea of how generalised model is

## Fit logistic model to training data
logit_model.fit(x_train, y_train)

## Evaluate model on test data
print("Logistic model test score:", logit_model.score(x_test, y_test))
y_pred0 = logit_model.predict(x_test)
metric_score_logit_model = metrics.classification_report(y_test, y_pred0)


## RF model run
parameters = {'max_depth': [5, 10, 20, 50, 100], 'n_estimators': [5, 10, 20, 50, 100, 200]}
rf_model = ensemble.RandomForestClassifier()
rf_model_gridsearch = model_selection.GridSearchCV(rf_model, parameters)
scores2 = model_selection.cross_val_score(rf_model_gridsearch, x_train, y_train, cv = 5)
print("Cross-validated RF model scores:", scores2)

## Fit model to training data
rf_model_gridsearch.fit(x_train, y_train)

## Evaluate model on test data
print("RF model test score:", rf_model_gridsearch.score(x_test, y_test))
y_pred2 = rf_model_gridsearch.predict(x_test)
metric_score_rf_model = metrics.classification_report(y_test, y_pred2)


## CONCLUDE ##
y_previous = dataset_seasonal.loc[dataset_seasonal["season"] == active_seasons[season_number - 1], "shot_made"]
if sum(y_pred2) > sum(y_previous):
    print("Model predicts that previous season's free throw count will be beat.")
elif sum(y_pred2) < sum(y_previous):
    print("Model predicts that previous season's free throw count will NOT be beat.")
else:
    print("Model predicts that the previous season's free throw count will be the same this season.")