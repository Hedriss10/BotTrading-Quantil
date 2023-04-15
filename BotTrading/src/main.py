import os 
import pandas as pd
import numpy as np 
import pickle

from datetime import timedelta, datetime
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight


#Iteração para preços relativos
def lagit(df, langs):
  for i in range(1, langs+1):
    df['Lang_'+str(i)] = df['ret'].shift(i)
  return ['Lang_'+str(i) for i in range(1, langs+1)]


def request_data():

    folder = [os.path.join("base", f) for f in os.listdir("base") if f.endswith('.csv')]
    for file_path in (folder):
        print('*'*50)
        print(f'Ativo: {file_path}')            
        df = pd.read_csv(file_path, parse_dates=['time'])
        df = df.loc[(df['time'] >= start_date) & (df['time'] <= end_date)]
        predicotors = ['open' , 'high', 'low', 'close', 'Volume', 'Volume MA']
        # Cria uma nova coluna "Direction", que indica se o preço subiu ou desceu em relação ao dia anterior
        df['direction'] = (df['close'] > df['close'].shift(1)).astype(int)

        df['ret'] = df.close.pct_change()
        lagit(df, 2)
        df['direction'] = np.where(df.ret > 0, 1, 0)
        features = lagit(df, 3)
        df.dropna(inplace=True)
        X = df[features]
        y = df['direction']
        model =  LogisticRegression(class_weight='balanced')
        model.fit(X, y)
        df['prediction_LR'] = model.predict(X)
        df['start'] = df['prediction_LR'] * df.ret
        validation = (df[['start', 'ret']] + 1).cumprod() -1
        X_train, X_test, y_train, y_test = train_test_split(df[predicotors], df['direction'], test_size=0.3, random_state=0)
        model = DecisionTreeClassifier(random_state=0)
        model.fit(X_train, y_train)
        
        # Avalia o desempenho do modelo nos dados de teste 
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        # Faz a previsão da direção futura dos preços para o último dia dos dados
        last_day = df.tail(1)[predicotors]
        prediction = model.predict(last_day)
        # print(f'Análise de variação do preço: {validation}')
        print(f'Direção do mercado calculada se for 0 é shel e 1 buy: {df.direction.iloc[0]}')
        print(f"Previsão para o proxímo dia: {'up' if prediction == 1 else 'down'}")
        print(f'Data Análisada: {end_date}')
        print('*'*50)
        # print(df.tail())
        
        with open("modelo.pkl", 'wb') as f:
          pickle.dump(model, f)
        
            
if __name__ == "__main__":
    #Data padronizada porfavor não mecha
    start_date = '2010-01-01'
    #Escolha a data para a previsão
    end_date = '2023-02-08'
    request_data()
