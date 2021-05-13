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
  lb_1         = []
  lb_2         = []
  centroides   = pd.DataFrame(columns = X.columns[:-1])

  # Carregar cada ponto para o seu rótulo
  for i in range(0,X.shape[0]):
    if X['labels'].values[i] == 1:
      lb_1.append(X.iloc[i,:-1])
    else:
      lb_2.append(X.iloc[i,:-1])

  # Armazena os pontos dos centroides
  centroides = centroides.append(sum(lb_1)/len(lb_1),ignore_index=True)
  centroides = centroides.append(sum(lb_2)/len(lb_2),ignore_index=True)
  centroides.reset_index(drop=True,inplace=True)
  centroides['labels'] = [1,2]

  return centroides