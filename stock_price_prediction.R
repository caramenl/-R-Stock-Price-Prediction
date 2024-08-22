
required_packages <- c('quantmod', 'tseries', 'timeSeries', 'forecast', 'ggplot2', 
                       'caret', 'keras', 'rtweet', 'tidyverse', 'quantstrat')

new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)


library(quantmod)
library(tseries)
library(timeSeries)
library(forecast)
library(ggplot2)
library(caret)
library(keras)
library(rtweet)
library(dplyr)  
library(quantstrat)


getSymbols('AAPL', from = '2019-01-01', to = '2021-01-01')
View(AAPL)


chartSeries(AAPL, subset = 'last 6 months', type = 'auto')
addBBands()
addMACD()
addRSI(n = 14)
addSMA(n = 50)
addSMA(n = 200)


Open_prices <- AAPL[,1]
High_prices <- AAPL[,2]
Low_prices <- AAPL[,3]
Close_prices <- AAPL[,4]
Volume_prices <- AAPL[,5]
Adjusted_prices <- AAPL[,6]


par(mfrow = c(2,3))
plot(Open_prices, main = 'Opening Price of Stocks (Over a given period)')
plot(High_prices, main = 'Highest Price of Stocks (Over a given period)')
plot(Low_prices, main = 'Lowest Price of Stocks (Over a given period)')
plot(Close_prices, main = 'Closing Price of Stocks (Over a given period)')
plot(Volume_prices, main = 'Volume of Stocks (Over a given period)')
plot(Adjusted_prices, main = 'Adjusted Price of Stocks (Over a given period)')


Predic_Price <- Adjusted_prices
par(mfrow = c(1,2))
Acf(Predic_Price, main = 'ACF for differenced Series')
Pacf(Predic_Price, main = 'PACF for differenced Series', col = '#cc0000')

print(adf.test(Predic_Price))

return_AAPL <- 100 * diff(log(Predic_Price))
AAPL_return_train <- return_AAPL[1:(0.9 * length(return_AAPL))]
AAPL_return_test <- return_AAPL[(0.9 * length(return_AAPL) + 1):length(return_AAPL)]

# ARIMA Model Fitting
fit <- auto.arima(AAPL_return_train, seasonal = FALSE)
summary(fit)

# Making Predictions
preds <- predict(fit, n.ahead = (length(return_AAPL) - (0.9 * length(return_AAPL))))$pred

# Forecasting
test_forecast <- forecast(fit, h = 15)
par(mfrow = c(1,1))
plot(test_forecast, main = "ARIMA Forecast for Apple Stock")


accuracy(preds, AAPL_return_test)

# Sentiment Analysis from Twitter
tweets <- search_tweets("Apple OR AAPL", n = 1000, lang = "en")
tweets_clean <- tweets %>%
  mutate(text = gsub("[^[:alnum:] ]", "", text)) %>%
  unnest_tokens(word, text)

sentiment <- tweets_clean %>%
  inner_join(get_sentiments("bing")) %>%
  count(sentiment, sort = TRUE) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)


ggplot(data = as.data.frame(AAPL), aes(x = index(AAPL), y = AAPL.Close)) +
  geom_line() + geom_smooth() + labs(title = "Apple Stock Price", x = "Date", y = "Close Price")


AAPL$Lag1 <- lag(AAPL$AAPL.Adjusted, 1)
AAPL$MA10 <- rollapply(AAPL$AAPL.Adjusted, width = 10, FUN = mean, by = 1, fill = NA)


timesteps <- 1  # Example value
features <- 1  # Example value

model <- keras_model_sequential() %>%
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(timesteps, features)) %>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam'
)


train_data <- as.matrix(AAPL_return_train)
dim(train_data) <- c(nrow(train_data), 1, ncol(train_data))

# Train the LSTM Model
history <- model %>% fit(
  train_data,
  epochs = 100,
  batch_size = 32
)


lstm_preds <- model %>% predict(train_data)
plot(lstm_preds, type = 'l', main = "LSTM Predictions for Apple Stock")

# Cross-Validation using ARIMA
train_control <- trainControl(method="cv", number=10)
model <- train(x=AAPL_return_train, y=AAPL_return_test, method="arima", trControl=train_control)










