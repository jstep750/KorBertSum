import os
from os import listdir
from os.path import isfile, join, isdir
import pandas as pd
from newspaper import Article
from tqdm import tqdm

def context(x):
    try:
        article = Article(x, language='ko')
        article.download()
        article.parse()
        return article.text
    except:
        return None
    
def add_context_to_df(df):
    contexts = []
    for i, link in enumerate(tqdm(df['url'])):
        cont = context(link)
        
        if(cont):
            contexts.append(cont)
        else:
            contexts.append('delete this')
    df['raw'] = contexts
    df = df.drop(df[df['raw'] == 'delete this'].index)
    return df

mypath = './'
onlydir = set([f for f in listdir(mypath)]) - set([f for f in listdir(mypath) if isfile(join(mypath, f))])
onlydir = [f for f in onlydir if f[0] in ['b','k']]
print(onlydir)

for dirs in onlydir:
    mypath = dirs
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = [f for f in onlyfiles if f.split('.')[1] == 'pkl' and 'raw' not in f.split('.')[0]]
    print('-------------', mypath, '-------------')
    print(onlyfiles)
    for file in onlyfiles:
        if(file.split('.')[1] == 'pkl' and 'raw' not in file.split('.')[0]):
            name = file.split('.')[0]
            df = pd.read_pickle(f'./{mypath}/{name}.pkl')
            print(name, len(df))
            if not os.path.exists(f'./{mypath}/raw'):
                os.makedirs(f'./{mypath}/raw')
            df = add_context_to_df(df)
            df.to_pickle(f'./{mypath}/raw/{name}_raw.pkl')
            df.to_csv(f'./{mypath}/raw/{name}_raw.csv')