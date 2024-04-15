# ARIMA+GARCH Trading Strategy on the ETF SPY

## Strategy Overview

# The idea of this strategy is as below:

# For each day use differenced logarithmic returns of ETF SPY for the previous days to fit an optimal ARIMA and GARCH model
# Use the combined model to make a prediction for the next day returns
# If the prediction is positive, go long the stock and if negative, short the stock at day's close
# If the prediction is the same as the previous day then do nothing

# In this sheet we will work with a window of 252 days, but this is parameter that can be optimised in order to improve performance or reduce drawdown.

# Note: The backtest is doesnot take comission or slippage into account, hence the performance achieved in a real trading system would be lower than what you see here.**

## Strategy Implementation

import pandas as pd
import numpy as np

import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW
import scipy.stats as scs
from arch import arch_model
import yfinance as yf
import time
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl

#Parameters

startDateStr = '2021-12-31'
endDateStr = datetime.now().date()

instrumentIds = ['SPY']

data = yf.download(instrumentIds,start=startDateStr,end=endDateStr,
                       auto_adjust=True,actions='inline',progress=True, period="1d")['Close']

# log returns
lrets = np.log(data/data.shift(1)).fillna(0)
lrets.index = pd.DatetimeIndex(lrets.index.values,freq=lrets.index.inferred_freq)

#I create a function to be able to find the best ARIMA model according to the criteria of
#lowest AIC and parameter estimation method innovations_mle

def _get_best_model(TS, max_order):
    best_aic = np.inf
    best_order = None
    best_arima_mdl = None

    pq_rng = range(1,max_order)
    for i in pq_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(TS, order=(i,0,j)).fit(
                        method='innovations_mle'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, 0, j)
                        best_arima_mdl = tmp_mdl
                except: continue

    #now that we have our ARIMA fit, we feed this to GARCH model
    p_ = best_order[0]
    o_ = best_order[1]
    q_ = best_order[2]

    am = arch_model(best_arima_mdl.resid, p=p_, o=o_, q=q_, dist='StudentsT')
    garch_model = am.fit(update_freq=5, disp='off')


    # print('aic: %6.5f | order: %s'%(best_aic, best_order))
    return best_aic, best_order, best_arima_mdl, garch_model


#I create function to graph

def tsplot(y, lags=None, figsize=(15, 10), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return


class Strategy:

    def __init__(self, name, BuyAndHold_logreturns, windowLength, max_order, periodicity):
        self.name = name + "_" + str(windowLength) + "_"  + str(max_order) + "_"  + str(periodicity)
        self.BuyAndHold_logreturns = BuyAndHold_logreturns
        self.windowLength = windowLength
        self.max_order = max_order
        # self.returns = log_returns
        self.periodicity = periodicity

        # periodicity = 1 dayly
        # periodicity = 5 weekly
        # periodicity = 22 montly
        # periodicity = 252 annual

    def outputs(self):

        # foreLength = len(self.BuyAndHold_logreturns) - self.windowLength
        #Variable de comparación para poder comparar todas las estrategias con
        #el forecasteo de los mismos días para poder decidir cual es mejor.
        #Como la de mayor duración es 252 usaré este valor

        maxwindowstrategy = 252

        foreLength = min(len(self.BuyAndHold_logreturns) - self.windowLength, len(self.BuyAndHold_logreturns) - maxwindowstrategy)

        self.foreLength = foreLength
        start_time = time.time()

        signal = 0*self.BuyAndHold_logreturns[-foreLength:]

        predictions = {'order_p': [0]*len(signal),
                    'order_q': [0]*len(signal),
                    'predicted_mu': [0]*len(signal),
                'predicted_et': [0]*len(signal),
                'prediction': [0]*len(signal)}
        predictions = pd.DataFrame(predictions, index= signal.index)

        for d in range(0+maxwindowstrategy-self.windowLength,foreLength):
            # create a rolling window by selecting the values between 1+d and k+d of S&P500 returns
            TS = self.BuyAndHold_logreturns[(1+d):(self.windowLength+d)]
            # Find the best ARIMA fit (we set differencing to 0 since we've already differenced the series once)
            res_tup = _get_best_model(TS, self.max_order)

            # Forecast 1 day horizon

            # Use ARIMA to predict mu
            predicted_mu = res_tup[2].predict(n_periods=1)[0]

            # Use GARCH to predict the residual
            garch_forecast = res_tup[3].forecast(horizon=1, start=None, align='origin')
            predicted_et = garch_forecast.mean['h.1'].iloc[-1]

            # Combine both models' output: yt = mu + et
            prediction = predicted_mu + predicted_et

            predictions['order_p'].iloc[d] = res_tup[1][0]
            predictions['order_q'].iloc[d] = res_tup[1][2]
            predictions['predicted_mu'].iloc[d] = predicted_mu
            predictions['predicted_et'].iloc[d] = predicted_et
            predictions['prediction'].iloc[d] = prediction

            signal.iloc[d] = np.sign(prediction)

        log_returns = signal*self.BuyAndHold_logreturns[-foreLength:]

        self.returns = log_returns

        # Finaliza el temporizador
        end_time = time.time()

        # Calcula el tiempo de ejecución
        execution_time = end_time - start_time
        self.execution_time = execution_time


        # Imprime el tiempo de ejecución
        print("Tiempo de ejecución:", execution_time, "segundos")
        print("Tiempo de ejecución:", execution_time/60, "minutos")

        return self.returns, signal, predictions, execution_time/60


    def summary(self, Include_BuyAndHold = False):

        summary_BuyAndHold = {'execution time (minutes)': 0,
                        'window length': len(self.BuyAndHold_logreturns),
                        'forecast length': 0,
                        'annual mean return': [np.exp(self.BuyAndHold_logreturns[-self.foreLength:].mean() * 252 / self.periodicity) -1],
                       'annual mean log return': [self.BuyAndHold_logreturns[-self.foreLength:].mean() * 252 / self.periodicity],
                       'annual std': [self.BuyAndHold_logreturns[-self.foreLength:].std() * (252 / self.periodicity)**0.5],
                       'sharpe ratio': [self.BuyAndHold_logreturns[-self.foreLength:].mean() * 252 / self.periodicity / (self.BuyAndHold_logreturns[-self.foreLength:].std() * (252 / self.periodicity)**0.5)],
                       'p(return)<2.5%': [np.exp(DescrStatsW(self.BuyAndHold_logreturns[-self.foreLength:]).tconfint_mean(alpha=1-0.95)[0]* 252 / self.periodicity)-1],
                       'p(return)>97.5%': [np.exp(DescrStatsW(self.BuyAndHold_logreturns[-self.foreLength:]).tconfint_mean(alpha=1-0.95)[1]* 252 / self.periodicity)-1],
                       'win ratio': [(self.BuyAndHold_logreturns[-self.foreLength:] > 0).sum() / len(self.BuyAndHold_logreturns[-self.foreLength:])],
                       'ratio risk/benefit': [-self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] > 0].sum() / self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] < 0].sum()],
                       'wt: quantity': [len(self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] > 0])],
                       'wt: annual mean return': [np.exp(self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] > 0].mean() * 252 / self.periodicity) -1],
                       'wt: annual mean log return': [(self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] > 0].mean() * 252 / self.periodicity)],
                       'wt: annual std': [self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] > 0].std() * (252 / self.periodicity)**0.5],
                       'wt: sharpe ratio': [self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] > 0].mean() * 252 / self.periodicity / (self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] > 0].std() * (252 / self.periodicity)**0.5)],
                       'lt: quantity': [len(self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] <= 0])],
                       'lt: annual mean return': [np.exp(self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] <= 0].mean() * 252 / self.periodicity) -1],
                       'lt: annual mean log return': [(self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] <= 0].mean() * 252 / self.periodicity)],
                       'lt: annual std': [self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] <= 0].std() * (252 / self.periodicity)**0.5],
                       'lt: sharpe ratio': [self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] <= 0].mean() * 252 / self.periodicity / (self.BuyAndHold_logreturns[-self.foreLength:][self.BuyAndHold_logreturns[-self.foreLength:] <= 0].std() * (252 / self.periodicity)**0.5)]
                       }

        summary_strategy = {'execution time (minutes)': self.execution_time/60,
                        'window length': self.windowLength,
                        'forecast length': self.foreLength,
                        'annual mean return': [np.exp(self.returns.mean() * 252 / self.periodicity) -1],
                       'annual mean log return': [self.returns.mean() * 252 / self.periodicity],
                       'annual std': [self.returns.std() * (252 / self.periodicity)**0.5],
                       'sharpe ratio': [self.returns.mean() * 252 / self.periodicity / (self.returns.std() * (252 / self.periodicity)**0.5)],
                       'p(return)<2.5%': [np.exp(DescrStatsW(self.returns).tconfint_mean(alpha=1-0.95)[0]* 252 / self.periodicity)-1],
                       'p(return)>97.5%': [np.exp(DescrStatsW(self.returns).tconfint_mean(alpha=1-0.95)[1]* 252 / self.periodicity)-1],
                       'win ratio': [(self.returns > 0).sum() / len(self.returns)],
                       'ratio risk/benefit': [-self.returns[self.returns > 0].sum() / self.returns[self.returns < 0].sum()],
                       'wt: quantity': [len(self.returns[self.returns > 0])],
                       'wt: annual mean return': [np.exp(self.returns[self.returns > 0].mean() * 252 / self.periodicity) -1],
                       'wt: annual mean log return': [(self.returns[self.returns > 0].mean() * 252 / self.periodicity)],
                       'wt: annual std': [self.returns[self.returns > 0].std() * (252 / self.periodicity)**0.5],
                       'wt: sharpe ratio': [self.returns[self.returns > 0].mean() * 252 / self.periodicity / (self.returns[self.returns > 0].std() * (252 / self.periodicity)**0.5)],
                       'lt: quantity': [len(self.returns[self.returns <= 0])],
                       'lt: annual mean return': [np.exp(self.returns[self.returns <= 0].mean() * 252 / self.periodicity) -1],
                       'lt: annual mean log return': [(self.returns[self.returns <= 0].mean() * 252 / self.periodicity)],
                       'lt: annual std': [self.returns[self.returns <= 0].std() * (252 / self.periodicity)**0.5],
                       'lt: sharpe ratio': [self.returns[self.returns <= 0].mean() * 252 / self.periodicity / (self.returns[self.returns <= 0].std() * (252 / self.periodicity)**0.5)]
                       }

        summary_BuyAndHold = pd.DataFrame(summary_BuyAndHold)
        summary_BuyAndHold = summary_BuyAndHold.rename(index={0:'Buy&Hold'})

        summary_strategy = pd.DataFrame(summary_strategy)
        summary_strategy = summary_strategy.rename(index={0:self.name})

        if Include_BuyAndHold == True:
            summary = pd.concat([summary_BuyAndHold, summary_strategy], ignore_index=False)
        else: summary = summary_strategy

        return summary

# At this stage we need to loop through every day in the trading data and fit an
# appropriate ARIMA and GARCH model to the rolling window of length.
# Given that we try 32 separate ARIMA fits and fit a GARCH model, for each day,
# the indicator can take a long time to generate.

#Testing the Strategy

#Window = 252 y Max order 3

ArimaGarch_200_3 = Strategy('ArimaGarch', lrets, 200, 3, 1)
ArimaGarch_outputs_200_3 = ArimaGarch_200_3.outputs()

ArimaGarch_200_3.summary(True)

# Graph of the results

tsplot(ArimaGarch_outputs_200_3[0])

#STRATEGY RESULTS

# Now that we have generated our signals, we need to compare its performance to "Buy & Hold".

BuyAndHold_Ret = np.exp(lrets[-(len(lrets)-252):].cumsum())
Strategy_Ret = np.exp(ArimaGarch_outputs_200_3[0].cumsum())

plt.plot(Strategy_Ret.index, BuyAndHold_Ret, label='Buy and Hold')
plt.plot(Strategy_Ret.index, Strategy_Ret, label='Strategy')

plt.legend()
plt.show()
