import pandas as pd

"""
            Take the base of the dos and reduce it from attributes
"""
def get_features():

    # Load datastet
    cl1 = pd.read_csv('Classe1.csv')
    cl2 = pd.read_csv('Classe2.csv')

    #Transpose datasets
    cl1 = cl1.T
    cl2 = cl2.T

    # Label it 
    cl1['labels'] = [1 for _ in range(0,cl1.shape[0])]
    cl2['labels'] = [2 for _ in range(0,cl2.shape[0])]

    #Store both datasets in one dataframe
    base = cl1
    for i in range(0,cl2.shape[0]):
        base = base.append(cl2.iloc[i])
        base.reset_index(drop=True,inplace=True)

    # Reduce the dataset, feature extraction 
    features = pd.DataFrame({})
    features['media']    = base.iloc[:,:-1].mean(axis='columns')
    features['max']      = base.iloc[:,:-1].max(axis='columns')
    features['min']      = base.iloc[:,:-1].min(axis='columns')
    features['kurtosis'] = base.iloc[:,:-1].kurtosis(axis='columns')
    features['mediana']  = base.iloc[:,:-1].median(axis='columns')
    
    # zscore normalization
    features = (features - features.mean())/features.var()

    # Label it features dataset
    features['labels']   = base['labels']
 
    return features