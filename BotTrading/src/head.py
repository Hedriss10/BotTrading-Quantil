import pickle

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight


with open('modelo.pkl', 'rb') as f:
    start_date = '2003-01-02'
    model = pickle.load(f)  
    print(model)