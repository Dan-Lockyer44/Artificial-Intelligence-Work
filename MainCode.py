import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('vgsales.csv') ##Used for reading in the .csv file into the program
le = preprocessing.LabelEncoder() ##Used to start the label encoder
data['Publisher'] = le.fit_transform(data['Publisher'].astype(str))
data['Genre'] = le.fit_transform(data['Genre'])
data['Platform'] = le.fit_transform(data['Platform'])

data = data.drop('Rank', axis=1) ##Drop certain columns
data = data.drop('Name', axis=1)
data = data.drop('NA_Sales', axis=1)
data = data.drop('JP_Sales', axis=1)
data = data.drop('Other_Sales', axis=1)
data = data[data['Global_Sales'] > 0.3] ##Only include data above 0.3
data = data.replace(to_replace='N/A', value="") ##Used to find any N/A section of data and replace the N/A with "NaN" aka NULL
data = data.dropna() ##Used to remove all NULL data frames

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) ##Used to produce a graph to show the number of sales per year
ax.hist(data['Year'], bins=7)
plt.title('Number of releases per year')
plt.xlabel('Game Release Year')
plt.ylabel('Amount of Releases')
plt.show()


fig = plt.figure() ##used to plot a graph using the actual data in the data set
ax = fig.add_subplot(1,1,1)
plt.title('Actual Sales')
plt.xlabel('Global_Sales')
plt.ylabel('EU_Sales')
ax.scatter(data['Global_Sales'],data['EU_Sales'])
plt.show()

print(data)
features = data.drop('EU_Sales', axis=1) ##Used to drop the EU_Sales column
target = data.EU_Sales ##Used to select a target for the data

features_train, features_test, target_train, target_test = train_test_split(features,
target, test_size = 0.33, random_state = 10)

model = RandomForestRegressor()
model.fit(features_train, target_train)
target_pred = model.predict(features_test)

print(r2_score(target_test, target_pred) * 100)

fig = plt.figure() #Used to plot a scatter graph using the predicted data
ax = fig.add_subplot(1,1,1)
plt.title('Predicted EU_Sales')
ax.scatter(target_test, target_pred)
plt.show()
