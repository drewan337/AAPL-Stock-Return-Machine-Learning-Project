# Final project: Stock Prediction

Given the recent stock market history, you will predict the return for ticker AAPL for the day ahead.

# Objectives

You will demonstrate your strengths in several areas, especially
- Exploratory Data Analysis
- Feature Engineering
- Creating, testing and evaluating models
- Error analysis
and Prediction **using Neural Networks**.

Your objective will be to predict next period price returns of a stock given past price data.

# Objectives

You will demonstrate your strengths in several areas, especially
- Exploratory Data Analysis
- Feature Engineering
- Creating, testing and evaluating models
- Error analysis
and Prediction **using Neural Networks**.

Your objective will be to predict next period price returns of a stock given past price data.

We will evaluate your model using *holdout data*
- identical in format to the training data
- but beginning on a date strictly after the last date in the training data

You will write a function
- that takes an example from the holdout data
- and predicts the date $t'$ return of AAPL
- where $t'$  is in the *last* 200 days of the holdout data's date range

This same function should be used on your training/validation/test data as well.
- to make a prediction for each date $t$ in the data set.

We will run this function, one date $t'$ at a time, for each of the last 200 days of the holdout data's date range.
- when predicting for date $t'$, you need not use your predictions for any date $t'' \lt t'$ 
- your prediction on date $t'$ should depend only on actual past data, not any prior prediction

Naturally, your function, when called to predict for date t, may only use
data available up to and including date (t'-1)

# The data

As explained in the Final Project Overview:
- you will be given a data directory for training (the training data)
    - supplied as a compressed archive file in the course's Resource tab in NYU Classes
    - unpacking this file will yield directory `./Data/train`

The data directory will contain one file per ticker, with a history going back many years of fields including
- Price: Close, Adj Close, Open, High, Low
- Volume

"return" means the percent change in the **Close** feature
- you can choose to define return based on the **Adj Close** feature
    - but you should explain your choice
    - in the rest of the instructions: the word "Close" will refer to whichever choice you made
