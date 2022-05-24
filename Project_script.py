# -*- coding: utf-8 -*-
"""
Created on Wed May 18 22:15:54 2022
@author: rdfla
"""

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import scipy.stats as stats
import time
import seaborn as sns
import yfinance as yf

# Kaggle datasets
# https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks

# Loading in the datasets from Kaggle and create Dataframes
stocks = pd.read_csv(r"Datasets\sp500_stocks.csv", parse_dates=['Date'],index_col='Date')
companies = pd.read_csv(r"Datasets\sp500_companies.csv")
index = pd.read_csv(r"Datasets\sp500_index.csv", parse_dates=['Date'], index_col='Date')

# Web scraping S&P500 information from Wikipedia & create a DataFrame
# get the response in the form of html
wikiurl="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
table_class="wikitable sortable jquery-tablesorter"
response=requests.get(wikiurl)
# parse data from the html into a beautifulsoup object
soup = BeautifulSoup(response.text, 'html.parser')
wikitable=soup.find('table',{'class':"wikitable"})
wiki_companies=pd.read_html(str(wikitable))
# convert list to dataframe
comps=pd.DataFrame(wiki_companies[0])

# Check the first 5 lines of each of the dataframes
# comps is effectively a subset of companies
print(stocks.head())
print(companies.head())
print(comps.head())
print(index.head())

# Index: Review & plot of S&P500 
print(index.describe())
print(index.isna().any())
# Plot Index time series
plt.plot(index)
plt.title("S&P500 Index (2012-2022)")
plt.xlabel("Year")
plt.ylabel("Price")
plt.show()
# Index daily returns
index['Daily return'] = index.squeeze().pct_change()
# Calculate mean, std, and generate a normal distribution
print(len(index))
av_ret = index['Daily return'].mean()
std_ret = index['Daily return'].std()
x = np.linspace(av_ret - 4*std_ret, av_ret + 4*std_ret, 2500)
# Overlay plots
plt.hist(index['Daily return'],color='red',bins=600,alpha=0.5)
plt.plot(x, stats.norm.pdf(x, av_ret, std_ret), color='blue')
plt.title("S&P 500: Histogram of daily returns",fontsize=20)
plt.show()
# Plot index mean and std dev
stats = index['Daily return'].resample("M").agg(['mean', 'std'])
stats.plot()
plt.title("S&P 500: mean and std dev",fontsize=20)
plt.show();

# Use Yahoo Finance package to download VIX data from Yahoo finance API
vix = yf.Ticker("^VIX")
# get historical market data
vix_hist = vix.history(start="2012-05-17", end="2022-05-16")
# Plot index & VIX with x2 y-axis
fig,ax = plt.subplots()
ax.plot(index['S&P500'], color="blue")
ax.set_xlabel("Year",fontsize=14)
ax.set_ylabel("Index",color="blue",fontsize=14)
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
ax2.plot(vix_hist['Close'],color="red",alpha=0.5)
ax2.set_ylabel("VIX",color="red",fontsize=14)
ax2.set_title("(1): S&P500 and VIX comparison",fontsize=20)
plt.show()
#Alternative plot - small multiples
fig,ax = plt.subplots(2,1)
ax[0].plot(index['S&P500'], color="blue")
ax[1].plot(vix_hist['Close'],color="red")
ax[0].set_ylabel("S&P500",color="b",fontsize=14)
ax[1].set_ylabel("VIX",color="r",fontsize=14)
ax[1].set_xlabel("Year",fontsize=14)
ax[0].set_title("(2): S&P500 and VIX comparison",fontsize=20)
plt.show()

# ANALYSIS OF STOCKS
# drop blank rows
stocks = stocks.dropna()
# Calculate daily returns of all stocks using own FOR loop with IF statement
stocks_v2 = []
start_time = time.time()
for i in range(0,len(stocks)):
    if i == 0:
        stocks_v2.append(0.00)
    elif stocks['Symbol'][i] == stocks['Symbol'][i-1]:
        stocks_v2.append((stocks['Close'][i]/stocks['Close'][i-1]) - 1)
    else:
        stocks_v2.append(0.00)
print("--- %s seconds ---" % (time.time() - start_time))
# Adding daily returns into stocks df
stocks['Daily Returns'] = stocks_v2
# Creating stocks_v3 with Date and daily returns
stocks_v3 = pd.read_csv(r"Datasets\sp500_stocks.csv")
stocks_v3 = stocks_v3.dropna()
stocks_v3['Daily Ret'] = stocks_v2

# Create top10 companies by Marketcap in the S&P500
top10 = companies.sort_values('Marketcap',ascending=False).head(10)
# Merge top10 and stock data
t10_stocks = stocks_v3.merge(top10, how='inner', left_on='Symbol',right_on='Symbol')

# Create pivot of the daily returns and then produce correlation matrix
df_pivot = t10_stocks.pivot('Date','Symbol','Daily Ret').reset_index()
corr_df = df_pivot.corr(method='pearson')
# Create heatmap using the correlation matrix
sns.heatmap(corr_df,annot=True)
plt.xticks(rotation=45)
plt.title('Daily Return Correlations',fontsize=20)
    
# Average sector returns
all_comps = companies.sort_values('Marketcap',ascending=False)
sector = stocks_v3.merge(all_comps, how='inner', left_on='Symbol',right_on='Symbol')
sector_av = sector['Daily Ret'].groupby(sector['Sector']).mean()
# Turn into a Dataframe and multiply by 100 in order to make %
sector_av = pd.DataFrame(sector_av*100)
sector_av.plot.bar(legend=False,color='g')
plt.title("Average daily return by sector",fontsize=20)
plt.xlabel("Sector",fontsize=14)
plt.ylabel("Average daily return (%)",fontsize=14)
plt.show()

# Custom function: Create plot of a stock price
def stock_plot(symbol):
    stock = stocks.loc[stocks['Symbol'] == str(symbol)]
    stock['date']=stock.index
    plt.plot(stock['date'],stock['Close'])
    plt.title(str(symbol) + " stock price (2012-2022)",fontsize=18)
    plt.xlabel("Year",fontsize=14)
    plt.ylabel("Price ($)",fontsize=14)
    return plt.show()
# Use the function to create the plot of Apple, Google, Microsoft, Amazon                       
stock_plot("AAPL")                       
stock_plot("GOOG")                       
stock_plot("MSFT")                       
stock_plot("AMZN")                         

# Focus on a single stock - AAPL
apple = stocks.loc[stocks['Symbol'] == "AAPL"]
apple['date']=apple.index
# Calculate moving average for 10, 50 and 100 days
ma50 = apple['Close'].rolling(50).mean()
ma100 = apple['Close'].rolling(100).mean()
ma200 = apple['Close'].rolling(200).mean()
plt.plot(apple['Close'],label="Close")
plt.plot(ma50,label="50-day")
plt.plot(ma100,label="100-day")
plt.plot(ma200,label="200-day")
plt.xlabel("Year")
plt.ylabel("Price ($)")
plt.legend(loc=2)
plt.title("AAPL: Stock price moving averages")
plt.show()

# ---------------------------------------------------------------
# Machine Learning: LSTM ----------------------------------------
# Credit: https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

ticker = yf.Ticker("AAPL")
# get historical market data
df = ticker.history(start="2012-01-01", end="2022-02-16")
df

plt.figure(figsize=(16,6))
plt.title('Close Price History',fontsize=20)
plt.plot(df['Close'])
plt.xlabel('Year', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# Create a new dataframe with only the 'Close' column 
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on i.e. 95% of dataset
training_data_len = int(np.ceil( len(dataset) * .95 ))

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model',fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')
plt.show()