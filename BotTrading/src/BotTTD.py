import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Carregando os dados
df = pd.read_csv('E:\\Projetos\\BotTrading-Quantil\\BotTrading\\src\\6A1.csv')

# Criando a coluna de sinal
df['Signal'] = np.nan
df.loc[0, 'Signal'] = 1
df.loc[1:, 'Signal'] = np.where(df['close'][1:].values > df['close'][:-1].values, 1, -1)


# Criando a coluna de retorno
df['Return'] = df['close'].pct_change()
df = df.dropna()

# Criando a coluna de target
df['target'] = np.where(df['Signal'].shift(-1) == 1, 1, 0)

# Separando os dados em treino e teste
X = df[['open', 'high', 'low', 'close']].values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo de regressão logística
clf = LogisticRegression()

# Treinando o modelo
clf.fit(X_train, y_train)

# Fazendo as previsões
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# Avaliando a acurácia do modelo
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)


# Separando as features e o target
X = df.drop('target', axis=1)
y = df['target']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciando o modelo de classificação
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Treinando o modelo
clf.fit(X_train, y_train)

# Realizando as previsões
y_pred = clf.predict(X_test)

# Calculando a acurácia
acc = accuracy_score(y_test, y_pred)

print("Acurácia: {:.2f}%".format(acc*100))

print(f"Acurácia no conjunto de treino: {train_accuracy}")
print(f"Acurácia no conjunto de teste: {test_accuracy}")
