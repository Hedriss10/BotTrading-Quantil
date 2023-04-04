import os 
import numpy as np
import pandas as pd 
import backtrader as bt

from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from data.make_dataset import load_csv



class ModelBotTrading_Quantil:
    def logit(self, df, langs):
        for i in range(1 , langs+1):
            df['Lang_'+str(i)] = df['ret'].shift(i)
        return ['Lang_'+str(i) for i in range(1, langs+1)]
    

    
class firstStrategy(bt.Strategy):
    
    def __init__(self):
        # initializing rsi, slow and fast sma
        self.rsi = bt.indicators.RSI(self.data.close, period=21)
        self.fast_sma = bt.indicators.SMA(self.data.close, period=50)
        self.slow_sma = bt.indicators.SMA(self.data.close, period=100)
        self.crossup = bt.ind.CrossUp(self.fast_sma, self.slow_sma)

    def next(self):
        if not self.position:
            if self.rsi > 30 and self.fast_sma > self.slow_sma:  # when rsi > 30 and fast_sma cuts slow_sma
                self.buy(size=100)  # buying 100 quantities 
        else:
            if self.rsi < 70:  # when rsi is below 70 line
                self.sell(size=100)  # selling 100 quantities



if __name__ == "__main__":
    
    def start_software():
            run = ModelBotTrading_Quantil()
            df = pd.read_csv(load_csv())
            df['ret'] = df.close.pct_change()
            run.logit(df, 2)
            df['direction'] = np.where(df.ret > 0, 1, 0)
            features = run.logit(df, 3)
            df.dropna(inplace=True)
            X = df[features]
            y = df['direction']
            model =  LogisticRegression(class_weight='balanced')
            model.fit(X, y)
            df['prediction_LR'] = model.predict(X)
            df['start'] = df['prediction_LR'] * df.ret
            validation = (df[['start', 'ret']] + 1).cumprod() -1
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
            model.fit(X_train, y_train)
            X_test['prediction_LR'] = model.predict(X_test)
            X_test['ret'] = df.ret[X_test.index[0]:]
            X_test['start'] = X_test['prediction_LR'] * X_test['ret']
            validation_test = (X_test[['start', 'ret']] + 1).cumprod() - 1
            
            

    def BotTrading():
        #Start in cash money 
        startcash = 10000
        cerebro = bt.Cerebro() # class backtradder
        cerebro.addstrategy(firstStrategy) # adding strategy 
        # Get HDFCBANK data from Yahoo Finance.

        data = bt.feeds.YahooFinanceCSVData(
            dataname="E:\\Projetos\\BotTrading-Quantil\\BotTrading\\src\\6A1.csv",
            fromdate=datetime(2020,11,1),
            todate =datetime(2021,11,1))

        cerebro.adddata(data)
        cerebro.broker.setcommission(commission=0.002)
        cerebro.run()
        portvalue = cerebro.broker.getvalue()
        pnl = portvalue - startcash
        # Printing out the final result
        print('Final Portfolio Value: ${}'.format(portvalue))
        print('P/L: ${}'.format(pnl))
            

            
    start_software()    
    BotTrading()