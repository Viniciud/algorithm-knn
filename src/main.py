import numpy as np
from utils import runKNN

k_values = [1, 3, 5, 7]

path = "C:/Users/144951/Desktop/python/algorithm-knn/dataBase/spambase.data"

for k in k_values:
    matrix, precision_rate, error_rate = runKNN(k, path)
    print('------------------------')
    print("Valor de k =", k)
    print('')
    print('Acertou {} % | Errou {} %: '.format(precision_rate, error_rate))
    print('')
    print(precision_rate)
    print(error_rate)
    print('')
    print('Matriz de confusão:')
    print(np.array(matrix))
    print('------------------------')
