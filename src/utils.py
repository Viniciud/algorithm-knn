import csv
import random

from knn import Data
from knn import Knn


def fileReader(filename):
    print('LENDO ARQUIVO...')
    categories = []
    data = []
    file = csv.reader(open(filename, "r"))

    for dataRow in file:
        category = dataRow[-1]

        if len(dataRow) != 0:
            params = []
            for i in range(len(dataRow)-1):
                params.append(float(dataRow[i]))
            instanceData = Data(params=params, category=category)
            data.append(instanceData)

            if not category in categories:
                categories.append(category)

    return (data, categories)


def dataSetCreator(path, percentage=0.2):
    dataset, categories = fileReader(path)
    data_for_training = []
    data_for_test = []

    print('CRIANDO DATASET...')
    for data in dataset:
        if random.random() < percentage:
            data_for_test.append(data)
        else:
            data_for_training.append(data)

    return (data_for_test, data_for_training, categories)


def runKNN(k, path):

    data_for_test, data_for_training, categories = dataSetCreator(path=path)

    data_for_predict = []

    knn = Knn(data_for_training)

    confusionMatrix = [[0 for i in range(len(categories))]
              for j in range(len(categories))]

    print('REALIZANDO PREVISÕES...')
    for data in data_for_test:
        result = knn.doPredict(data, k)

        category_index = categories.index(data.category)
        result_index = categories.index(result)

        confusionMatrix[result_index][category_index] += 1

        data_for_predict.append(
            Data(params=data.params, category=result))

    precision_rate, error_rate = __detectAccuracy(confusionMatrix, len(data_for_test))

    return (confusionMatrix, precision_rate, error_rate)

def __detectAccuracy(matrix, length):
    print('VERIFICANDO PRECISÃO...')
    precision_predict_count = 0

    for index in range(len(matrix)):
        precision_predict_count += matrix[index][index]

    precision_rate = (precision_predict_count*100) / length
    error_rate = 100 - precision_rate

    return (precision_rate, error_rate)