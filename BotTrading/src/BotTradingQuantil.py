import os 
import pandas as pd
import numpy as np 

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight






def lagit(df, langs):
  for i in range(1, langs+1):
    df['Lang_'+str(i)] = df['ret'].shift(i)
  return ['Lang_'+str(i) for i in range(1, langs+1)]



def request_data():

    folder = [os.path.join("base", f) for f in os.listdir("base") if f.endswith('.csv')]
    
    for file_path in folder:
        print(f'Ativo: {file_path}')            
        df = pd.read_csv(file_path, parse_dates=['time'])
        # Filtra as linhas que se encontram dentro do período desejado
        df = df.loc[(df['time'] >= inicio) & (df['time'] <= fim)]
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
        print(validation)
        
        
        # X_train, X_test, y_train, y_test = train_test_split(df[predicotors], df['Direction'], test_size=0.3, random_state=0)

        # model = DecisionTreeClassifier(random_state=0)
        # model.fit(X_train, y_train)
        

        # # Avalia o desempenho do modelo nos dados de teste
        # y_pred = model.predict(X_test)
        # print(classification_report(y_test, y_pred))
        # print(confusion_matrix(y_test, y_pred))

        # # Faz a previsão da direção futura dos preços para o último dia dos dados
        # last_day = df.tail(1)[predicotors]
        # prediction = model.predict(last_day)
        # print(f"Previsão para o último dia dos dados: {'up' if prediction == 1 else 'down'}")





if __name__ == "__main__":
    #Selecionando a data para o posicionamento
    inicio = '2004-01-01'
    fim = '2015-12-31'
    request_data()