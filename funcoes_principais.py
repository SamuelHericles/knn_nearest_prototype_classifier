"""
                    PRIMEIRO TRABALHO DE RECONHECIMENTO DE PADRÕES
                            UNIVERSIDADE FEDERAL DO CEARÁ
                            SAMUEL HERICLES SOUZA SILVEIRA
                                    SOBRAL - 2020

- Implementar os classificadores k-NN e Nearet Prototype Classifier
  para classificar o banco de dados fornecidos
- No algoritmo k-NN, você deverá testar vários valores de k e usar
  o valor que fornecer a melhor taxa de acerto
- Usar validação cruzada K-fold com K=10
- O classificador possui 2 classes: sinal de ECG e sinal de áudio
- Cada uma das 2 classes possui 50 sinais, cada sinal com duração de 500 pontos
- Você deve escolher os atributos que achar mais convenientes 
- Deve-se usar pelo menos 5 atributos.

"""

"""
          Imports de bibliotecas necessáricas
"""

# Biblioteca para tratar vetores e matrizes
import numpy as np 

# Biblioteca para tratar a base de dados ou dataframe
import pandas as pd

# Bbilioteca que gera número aleatórios
import random

# Biblioteca que criei para funcoes básicas para os dois algoritmos
import funcoes_basicas as fb

# Biblioteca que criei para tratar a base de dados de acordo com o trabalho
import obtencao_de_atributos as obt

# Base de dados tratada
atributos = obt.get_atributos()

"""
          Algoritmo K-Nearest-Neighborhood

1.Carregar os dados

2.Inicialize K no número de vizinhos escolhido

3. Para cada exemplo nos dados

3.1 Calcule a distância entre o exemplo da consulta e o exemplo atual dos dados.

3.2 Adicione a distância e o índice do exemplo a uma coleção ordenada

4. Classifique a coleção ordenada de distâncias e índices do menor para o maior
   (em ordem crescente) pelas distâncias

5. Escolha as primeiras K entradas da coleção classificada

6. Obtenha os rótulos das entradas K selecionadas

7. Retornar o moda dos rótulos K

"""


"""
  @param  X_treino    dados de treino com os rótulos(y_treino)
  @param  y_teste     dados de teste sem os rótulos
  @param  n_vizinhos  n vizinhos mais próximos escolhidos

  @return y_repd      retorna os rótulos dos y_teste para comparar com os reais
"""
def KNN(X_treino,y_teste,n_vizinhos):
  y_pred = pd.DataFrame({})

  # Calcular a distancia euclidiana para cada ponto do sinal de treino com o de cada teste
  for j in range(0,y_teste.shape[0]):
    df_aux = pd.DataFrame({'distance':[],'label':[]})  
    for i in range(0,X_treino.shape[0]):

      # Armazenar em um dataframe para tratamento posterior
      df_aux = df_aux.append({'distance':fb.dist_euclidiana(X_treino.iloc[i][:-1],y_teste.iloc[j])
                      ,'label':X_treino['labels'].iloc[i]}
                      ,ignore_index=True)

    # Pegar os n vizinhos mais próximo e rotular o ponto de teste a partir da moda dos vizinhos                  
    y_pred = y_pred.append( fb.mode_labels(df_aux,y_teste,n_vizinhos,j))

  return y_pred

"""
          Nearest Prototype Classifier

1. Carregar os dados

2. Dividir os dados de treino pelos rótulos

3. Calcular os pontos médios de cada rótulo

4. Calcular os pontos médios dos pontos médios de cada rótulo para obter o centroides

5. A partir dos centróides de cada rótulo calcular as distancias de cada ponto de teste

6. Cada ponto raquear os mais próximos

7. rótular os pontos de testes com os rótulos dos centroides mais próximos

"""

"""
  @param X dados de treino(possui os rótulos)
  @param y dados de teste(não possui os rótulos)

  @return rótulos dos pontos
"""
def Nearest_prototype_classifier(X,y):
  
  # Calcula os centróides dos rótulos
  centroides = fb.calcula_centroides(X)

  # Calcula as distancias dos rótulos
  y_pred     = pd.DataFrame({'labels':[]})
  for i in range(0, y.shape[0]):
    df_aux = pd.DataFrame({'Distance':[],'labels':[]})
    for j in range(0,centroides.shape[0]):
      #print(f'{y.iloc[i,:-1]}-{centroides.iloc[j,:-1]}')
      df_aux = df_aux.append({'Distance': fb.dist_euclidiana(y.iloc[i,:-1],
                                                             centroides.iloc[j,:-1]),
                              'labels'  : centroides.iloc[j,-1]},
                              ignore_index=True)    
    df_aux.sort_values('Distance',inplace=True)

    # Armazenza os rótulos dos pontos
    y_pred = y_pred.append({'labels':df_aux.iloc[0,1]}
                            ,ignore_index=True)
    y_pred.reset_index(drop=True,inplace=True)
    
  return y_pred

"""
                                        K-fold

Fornece índices de treinamento / teste para dividir dados em conjuntos de treinamento / 
teste. Divida o conjunto de dados em k dobras consecutivas        
"""

"""
  @param k             número de divisoes de trenio/teste
  @param atributos     base dados fornecida
  @param metodo        funçao de classificacao desejada
  @param n_vizinhos    numeros de vizinhos mair próximos, caso seja 0 não usado o knn

  @return               vazio, mas imprime na tela a acurácia média dos folds
"""
def kfold(k,atributos,metodo,n_vizinhos=0):

  # Carrega os indice da base da dados
  indices = [i for i in range(0,atributos.shape[0])]
  acuracias = []

  # Divide os dados para treino e teste a partir escolha aleatória
  for _ in range(0,int(len(indices)/k)):
    itreino = []
    iteste  = []

    # Escolhe os indices dos sinais de teste aleatoriamente
    values =  sorted([int(random.randint(0,99)) for _ in range(0,k)])

    # Os indices que nao sao pra teste vao para o treino
    for i in indices:
        if i in values:
          iteste.append(i)
        else:
          itreino.append(i)
    X = atributos.iloc[itreino,:]
    y = atributos.iloc[iteste,:]
    X.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    
    # Escolha do metodo de classificacao
    if n_vizinhos == 0:
      y_pred = metodo(X,y)
    else:
      y_pred = metodo(X,y.iloc[:,:-1],n_vizinhos)

    # Armazena cada acuracia dos folds  
    acuracias.append(fb.acuracia(y,y_pred))
  
  # Exibe a acuracia media
  print(f'Acurácia média do {str(metodo.__name__)}: {np.mean(acuracias)}%')
  

kfold(10,atributos,Nearest_prototype_classifier)
# for i in [2,3,4,5,6,7,8,9,10,12,13,25]:
#   print(f'{i} folds temos:')
#   kfold(i,atributos,Nearest_prototype_classifier)
#   print('-'*50)

# Conforme requerido verificar o melhor resultado a partir dos parâmetros
# for n_vizinhos in [2,3,4,5,6,10]:
#   for folds in [4,6,8,10]:
#     print(f'{n_vizinhos} vizinhos com {folds} folds')
#     kfold(folds,atributos,KNN,n_vizinhos)
#     print('-'*50)
