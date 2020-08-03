import pandas as pd

"""
    Pega a base da dos e reduz ela a partir de atributos
"""
def get_atributos():

    # Carrega a base dados
    cl1 = pd.read_csv('Classe1.csv')
    cl2 = pd.read_csv('Classe2.csv')

    #Transposta da base dados
    cl1 = cl1.T
    cl2 = cl2.T

    # Rotular os dados
    cl1['labels'] = [1 for _ in range(0,cl1.shape[0])]
    cl2['labels'] = [2 for _ in range(0,cl2.shape[0])]

    # Armazena as duas classes em um unico dataframe
    base = cl1
    for i in range(0,cl2.shape[0]):
        base = base.append(cl2.iloc[i])
        base.reset_index(drop=True,inplace=True)

    # Reduz a base dados em atributos(caracteristicas definidas)
    atributos = pd.DataFrame({})
    atributos['media']    = base.iloc[:,:-1].mean(axis='columns')
    atributos['max']      = base.iloc[:,:-1].max(axis='columns')
    atributos['min']      = base.iloc[:,:-1].min(axis='columns')
    atributos['kurtosis'] = base.iloc[:,:-1].kurtosis(axis='columns')
    atributos['mediana']  = base.iloc[:,:-1].median(axis='columns')
    
    # Normlização zscore
    atributos = (atributos - atributos.mean())/atributos.var()

    # Rotulação dos dados
    atributos['labels']   = base['labels']

    # Retorna a base de dados   
    return atributos