import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Carrega os dados do arquivo CSV
df = pd.read_csv('E:\Projetos\BotTrading-Quantil\BotTrading\src\CL1.csv')

# Define o período que você quer analisar
inicio = '2010-01-01'
fim = '2015-12-31'

# Filtra as linhas que se encontram dentro do período desejado
df = df.loc[(df['time'] >= inicio) & (df['time'] <= fim)]


# Define as colunas que serão usadas para prever a direção futura
predictors = ['open', 'high', 'low', 'close', 'Volume', 'Volume MA']

# Cria uma nova coluna "Direction", que indica se o preço subiu ou desceu em relação ao dia anterior
df['Direction'] = (df['close'] > df['close'].shift(1)).astype(int)

# Remove as linhas que contêm valores ausentes
df.dropna(inplace=True)

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(df[predictors], df['Direction'], test_size=0.3, random_state=0)

# Treina o modelo de árvore de decisão
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

# Avalia o desempenho do modelo nos dados de teste
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Faz a previsão da direção futura dos preços para o último dia dos dados
last_day = df.tail(1)[predictors]
prediction = model.predict(last_day)
print(f"Previsão para o último dia dos dados: {'up' if prediction == 1 else 'down'}")
