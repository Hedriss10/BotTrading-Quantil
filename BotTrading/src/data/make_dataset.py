import os 
import sys


# csv_files = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.csv')]


sys.path.append("..")

x = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
a = [os.path.join(x, f) for f in os.listdir(x) if f.endswith('.csv')]


def base_csv():
    filer = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    folder = os.path.join(x, 'data')
    return folder 




def load_csv():
    file = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arq = os.path.join(file, '6A1.csv')
    return arq 
    
    