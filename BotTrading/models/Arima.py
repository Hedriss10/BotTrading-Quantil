import os
import pandas as pd 
import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller
# Carregando o conjunto de dados em um DataFrame do pandas
# dataset = [os.path.join("base", f) for f in os.listdir("base") if f.endswith('.csv')] 

dataset = 'E:\\Projetos\\BotTrading-Quantil\\BotTrading\\models\\DOL1.csv'


df = pd.read_csv(dataset, parse_dates=['time'])
df['time'] = pd.to_datetime(df['time'])
df = df.fillna(0)

def testa_adfuller(x):
    pvalue = adfuller(x)[1]
    
    if pvalue < 0.05:
        print('A série é estacionária')
    else:
        print('A série é não estacionária')

testa_adfuller(df['close'])

df['close_diff'] = df['close'].diff(periods=1)


p = 1
d = 0
q = 1

# Ajustar o modelo ARIMA na coluna close_diff
modelo_arima = sm.tsa.ARIMA(df['close'].diff(periods=1).dropna(), order=(p, d, q))
modelo_arima_fit = modelo_arima.fit()

# Fazer as previsões na coluna close_diff
previsao_diff = modelo_arima_fit.forecast(steps=30)[0]

# Aplicar a operação inversa para obter as previsões para a coluna close original
ultima_data = df.index[-1]
proximas_datas = pd.date_range(start=ultima_data, periods=len(previsao_diff)+1, freq='D')[1:]
previsao = df['close'].iloc[-1] + previsao_diff.cumsum()
df_previsoes = pd.DataFrame({'Data':proximas_datas, 'close':previsao})
df_previsoes.set_index('Data', inplace=True)

print(df_previsoes)



# # Definindo a ordem do modelo ARIMA (p, d, q)
# p = 1
# d = 0
# q = 1

# modelo_arima = sm.tsa.ARIMA(df[''], order=(2,1,2))
# modelo_arima_fit = modelo_arima.fit()
# previsao = modelo_arima_fit.forecast(steps=30)[0]



# # Adicionando as previsões ao seu DataFrame
# ultima_data = df.index[-1]
# proximas_datas = pd.date_range(start=ultima_data, periods=len(previsao)+1, freq='D')[1:]
# df_previsoes = pd.DataFrame({'data':proximas_datas, 'a':previsao})
# df_previsoes.set_index('Data', inplace=True)

# df_com_previsao = pd.concat([df, df_previsoes])
# print(df_com_previsao)