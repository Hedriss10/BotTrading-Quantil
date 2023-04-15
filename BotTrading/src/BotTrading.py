import os
import pandas as pd 
import numpy as np 
from datetime import timedelta, datetime
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight

class RequestDados:
    
    def __init__(self) -> None:
        pass
    
    def RequestDataFynance(self):
        folder = [os.path.join("base", f) for f in os.listdir("base") if f.endswith('.csv')]
        return folder
        
    def lagit(self, df, langs):
        for i in range(1, langs+1):
            df['Lang_'+str(i)] = df['ret'].shift(i)
        return ['Lang_'+str(i) for i in range(1, langs+1)]

    def Analyzer(self, end_date):
        start_date = '2003-01-02'
        dt_lst = []
        for file_path in self.RequestDataFynance():
            df = pd.read_csv(file_path, parse_dates=['time'])
            df = df.loc[(df['time'] >= start_date) & (df['time'] <= end_date)]
            predicotors = ['open' , 'high', 'low', 'close', 'Volume', 'Volume MA']
            df['direction'] = (df['close'] > df['close'].shift(1)).astype(int)
            df['ret'] = df.close.pct_change()
            features = self.lagit(df, 3)
            df.dropna(inplace=True)
            X = df[features]
            y = df['direction']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(class_weight='balanced')
            model.fit(X_train,y_train)
            df['prediction_LR'] = model.predict(X)
            df['start'] = df['prediction_LR'] * df.ret
            validation = (df[['start', 'ret']] + 1).cumprod() -1 
            sinal = None
            
            if df['direction'].iloc[-1] == 0 & df['prediction_LR'].iloc[-1] == 0:
                sinal = "down"
            
            elif df['direction'].iloc[-1] == 1 & df['prediction_LR'].iloc[-1] == 1:
                sinal = "up"
            
            else:
                sinal = "Not identified"
            
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
            next_date_obj =  end_date_obj + timedelta(days=1)
            end_date_str = end_date_obj.strftime('%Y-%m-%d')
            next_date_str = next_date_obj.strftime('%Y-%m-%d')
            
            
            
            dt = pd.DataFrame({
                'Ativo': [os.path.basename(file_path)],
                'Sinal': [sinal],
                'Data' : [next_date_str]
            })
            dt_lst.append(dt)
        result = pd.concat(dt_lst)        
        result.to_excel("Resultado\\ResultadoAnalise.xlsx", index=False)
        folder = 'Resultado'
        if not os.path.exists(folder):
            os.makedirs("Resultado")

if __name__ == "__main__":
    #Selecione a data ele irá analisar o próximo dia
    end_date = '2023-01-15'
    a = RequestDados()
    a.Analyzer(end_date)
