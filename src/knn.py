class Knn:
    def __init__(self, data):
        self.data = data

    def __distanceList(self, neigh):
        neighbours = []

        for el in self.data:
            neighbours.append((el, el.euclidianDistance(neigh)))

        return sorted(neighbours, key=lambda el: el[1])

    def categorizate(self, categories):
        categorization = None

        for category in categories.keys():
            if categorization is None:
                categorization = category
            elif categories[category] > categories[categorization]:
                categorization = category
        return categorization

    def doPredict(self, el, k):

        categories = {}
        neighbours = self.__distanceList(el)

        for i in range(k):
            data, _ = neighbours[i]
            if data.category in categories:
                categories[data.category] += 1
            else:
                categories[data.category] = 1

        return self.categorizate(categories)

class Data:
    def __init__(self, params, category):
        self.params = params
        self.category = category

    def euclidianDistance(self, otherValue):
        res = 0
        for i in range(len(self.params)):
            res += (self.params[i] - otherValue.params[i]) ** 2

        return res