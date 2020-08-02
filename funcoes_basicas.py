import numpy as np
import pandas as pd

# Calcula o ponto medio entre os pontos
def ponto_medio(x,y):
  return sum(x,y)/2

# Calcula a distancia eucliana entre os pontos
def dist_euclidiana(x,y):
  return np.sqrt(sum(pow(x-y,2)))

# Rotula os pontos e classifica os vizinhos mais proximos
def mode_labels(df,y_teste,k,j):
    label_mode = df.sort_values('distance')[:k]
    label_signal = int(label_mode['label'].mode().values[0])

    sample = pd.DataFrame(y_teste.iloc[j]).T
    sample['labels'] = label_signal

    return sample

# Calcula a acuracia dos resutlados
def acuracia(y_pred,y_teste):
  return  (sum(y_pred['labels'].values == y_teste['labels'].values)/y_pred.shape[0])*100


# Calcula os centroides da distribuicao de pontos de cada rotulo
def calcula_centroides(X):
  lb_1         = pd.DataFrame(columns = X.columns[:-1])
  lb_2         = pd.DataFrame(columns = X.columns[:-1])
  pts_medio_1  = pd.DataFrame(columns = X.columns[:])
  pts_medio_2  = pd.DataFrame(columns = X.columns[:])
  centroides_1 = pd.DataFrame(columns = X.columns[:])
  centroides_2 = pd.DataFrame(columns = X.columns[:])
  centroides   = pd.DataFrame(columns = X.columns[:])

  for i in range(0,X.shape[0]):
    if X['labels'].values[i] == 1:
      lb_1 = lb_1.append(X.iloc[i,:-1],ignore_index=True)
    elif X['labels'].values[i] == 2:
      lb_2 = lb_2.append(X.iloc[i,:-1],ignore_index=True)
  
  lb_1.reset_index(drop=True,inplace=True)
  lb_2.reset_index(drop=True,inplace=True)
  
  for i in range(0,lb_1.shape[0]):
    aux = 0
    if i == int(lb_1.shape[0])-1:
      aux = ponto_medio(lb_1.iloc[i],lb_1.iloc[0])
      pts_medio_1 = pts_medio_1.append({'media':aux[0],
                                        'max':aux[1],
                                        'min':aux[2],
                                        'kurtosis':aux[3],
                                        'mediana':aux[4]}
                                        ,ignore_index=True)
    else:
      aux = ponto_medio(lb_1.iloc[i],lb_1.iloc[i+1])
      pts_medio_1 = pts_medio_1.append({'media':aux[0],
                                        'max':aux[1],
                                        'min':aux[2],
                                        'kurtosis':aux[3],
                                        'mediana':aux[4]}
                                        ,ignore_index=True)

  for i in range(0,lb_2.shape[0]):
    aux = 0
    if i == int(lb_2.shape[0])-1:
      aux = ponto_medio(lb_2.iloc[i],lb_2.iloc[0])
      pts_medio_2 = pts_medio_2.append({'media':aux[0],
                                        'max':aux[1],
                                        'min':aux[2],
                                        'kurtosis':aux[3],
                                        'mediana':aux[4]}
                                        ,ignore_index=True)
    else:
      aux = ponto_medio(lb_2.iloc[i],lb_2.iloc[i+1])
      pts_medio_2 = pts_medio_2.append({'media':aux[0],
                                        'max':aux[1],
                                        'min':aux[2],
                                        'kurtosis':aux[3],
                                        'mediana':aux[4]}
                                        ,ignore_index=True)

  for i in range(0,int(pts_medio_1.shape[0]/2)):
    aux = 0 
    aux = ponto_medio(pts_medio_1.iloc[i],pts_medio_1.iloc[i+int(pts_medio_1.shape[0]/2)])
    centroides_1 = centroides_1.append({'media':aux[0],
                                        'max':aux[1],
                                        'min':aux[2],
                                        'kurtosis':aux[3],
                                        'mediana':aux[4]}
                                        ,ignore_index=True)

  for i in range(0,int(pts_medio_2.shape[0]/2)):
    aux = 0
    aux = ponto_medio(pts_medio_2.iloc[i],pts_medio_2.iloc[i+int(pts_medio_2.shape[0]/2)])
    centroides_2 = centroides_2.append({'media':aux[0],
                                        'max':aux[1],
                                        'min':aux[2],
                                        'kurtosis':aux[3],
                                        'mediana':aux[4]}
                                        ,ignore_index=True)
                                 
    
    centroides = centroides.append(centroides_1.mean(),ignore_index=True)
    centroides = centroides.append(centroides_2.mean(),ignore_index=True)
    centroides.reset_index(drop=True,inplace=True)
    centroides['labels'] = [1,2]

    return centroides