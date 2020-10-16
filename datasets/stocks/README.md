# DJIA 30 Stock Time Series

**Historical stock data for DJIA 30 companies 2006-2018**

Downloaded from [Kaggle](https://www.kaggle.com/szrlee/stock-time-series-20050101-to-20171231) on October 16, 2020.

## Content

Contains 13 years of daily stock data from 2006-01-01 to 2018-01-01 from the Dow Jones Industrial Average index companies. The files `all_stocks.csv` and the subset `all_stocks_2017.csv` contain all records while the files `[SYM].csv` contain the records for an individual company by their stock ticker name.

All the files have the following columns:

- Date - in format: yy-mm-dd
- Open - price of the stock at market open (this is NYSE data so all in USD)
- High - Highest price reached in the day
- Low - Lowest price reached in the day
- Close - price of the stock at market close
- Volume - Number of shares traded
- Name - the stock's ticker name
