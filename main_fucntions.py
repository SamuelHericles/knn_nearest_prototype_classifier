"""
                     FIRST STANDARD RECOGNITION WORK
                      FEDERAL UNIVERSITY OF CEARA
                     SAMUEL HERICLES SOUZA SILVEIRA
                             SOBRAL - 2020

- Implement the k-NN and Nearet Prototype Classifier classifiers
   to sort the database provided
- In the k-NN algorithm, you must test several k values and use
   the value that provides the best hit rate
- Use K-fold cross validation with K = 10
- The classifier has 2 classes: ECG signal and audio signal
- Each of the 2 classes has 50 signals, each signal lasting 500 points
- You must choose the attributes that you find most convenient
- At least 5 attributes must be used.

"""

"""
                                Imports needed
"""

# Library to treat vectors and matrices
import numpy as np 

# Library to handle the database or dataframe
import pandas as pd

# Library that generates random numbers
import random

# Library I created for basic functions for both algorithms
import basic_functions as fb

# Library I created to handle the database according to the work
import get_features as obt

# Database treated
atributos = obt.get_features()

"""
                                   K-Nearest-Neighborhood Algorithm

   1. Load the data;
   2.Start K at the chosen number of neighbors;
   3. For each example in the data;
   3.1 Calculate the distance between the query example and the current data example;
   3.2 Add the distance and index of the example to an ordered collection;
   4. Sort the ordered collection of distances and indices from smallest to largest
     (in ascending order) by distances;
   5. Choose the first K entries from the classified collection;
   6. Obtain the labels of the selected K entries;
   7. Return the fashion of the K labels.
"""


def KNN(X_treino,y_teste,n_vizinhos):

  """
     @param X_treino training data with labels (y_treino)
     @param y_teste test data without labels
     @param n_vizinhos n closest neighbors chosen

     @return y_repd returns the labels of the y_test to compare with the real ones
  """
  y_pred = pd.DataFrame({})

  # Calculate the Euclidean distance for each point of the training signal with that of each test
  for j in range(0,y_teste.shape[0]):
    df_aux = pd.DataFrame({'distance':[],'label':[]})  
    for i in range(0,X_treino.shape[0]):

      # Store on a dataframe for further processing
      df_aux = df_aux.append({'distance':fb.dist_euclidiana(X_treino.iloc[i][:-1],y_teste.iloc[j])
                      ,'label':X_treino['labels'].iloc[i]}
                      ,ignore_index=True)

    # Take the nearest n neighbors and label the test point in the fashion of the neighbors
    y_pred = y_pred.append( fb.mode_labels(df_aux,y_teste,n_vizinhos,j))

  return y_pred

"""
                                  Nearest Prototype Classifier

   1. Load the data
   2. Divide training data by labels
   3. Calculate the midpoints of each label
   4. Calculate the midpoints of the midpoints of each label to obtain the centroides
   5. From the centroid of each label, calculate the distances from each test point
   6. Each point to rake the closest
   7. label the test points with the labels of the nearest centroid

"""


def Nearest_prototype_classifier(X,y):
  """
     @param X training data (has the labels)
     @param y test data (does not have the labels)

     @return point labels
"""  

  # Calculates label centroids
  centroides = fb.calcula_centroides(X)

  # Calculates label distances
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

    # Store the point labels
    y_pred = y_pred.append({'labels':df_aux.iloc[0,1]}
                            ,ignore_index=True)
    y_pred.reset_index(drop=True,inplace=True)
    
  return y_pred

"""
                                         K-fold

  Provides training / testing indices for dividing data into training sets /
  test. Divide the data set into k consecutive folds
"""

def kfold(k,atributos,metodo,n_vizinhos=0):
  """
     @param k number of sled / test divisions
     @param attributes database provided
     @param method desired classification function
     @param n_vizinhos numbers of nearest neighbors, if 0 is not used the knn

     @return empty, but prints average fold accuracy on the screen
  """

  # Divide data for training and testing from random choice
  indices = [i for i in range(0,atributos.shape[0])]
  acuracias = []

  # Divide data for training and testing from random choice
  for _ in range(0,int(len(indices)/k)):
    itreino = []
    iteste  = []

    # Choose the test signal indices at random
    values =  sorted([int(random.randint(0,99)) for _ in range(0,k)])

    # The indices that are not for testing go to training
    for i in indices:
        if i in values:
          iteste.append(i)
        else:
          itreino.append(i)
    X = atributos.iloc[itreino,:]
    y = atributos.iloc[iteste,:]
    X.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    
    # Choice of classification method
    if n_vizinhos == 0:
      y_pred = metodo(X,y)
    else:
      y_pred = metodo(X,y.iloc[:,:-1],n_vizinhos)

    # Stores every accuracy of folds
    acuracias.append(fb.acuracia(y,y_pred))
  
  # Displays average accuracy
  print(f'Mean accuracy {str(metodo.__name__)}: {np.mean(acuracias)}%')
  

"""
                        Test for Nearest Prototype Classifier
"""

print('*'*80)
print(' '*20,'Test for Nearest_prototype_classifier')
print('*'*80)
kfold(10,atributos,Nearest_prototype_classifier)

# If you wanted to find the best kfold for the Nearest, just uncomfortable
# for i in [2,3,4,5,6,7,8,9,10,12,13,25]:
#   print(f'{i} folds temos:')
#   kfold(i,atributos,Nearest_prototype_classifier)
#   print('-'*50)

print('\n\n')

"""
                                Test for KNN
"""
print('*'*80)
print(' '*30,'Test for KNN')
print('*'*80)

# As required to check the best result from the parameters
for n_vizinhos in [2,3,4,5,6,10]:
    print(f'{n_vizinhos} neighbors 10 folds')
    kfold(10,atributos,KNN,n_vizinhos)
    print('-'*50)
