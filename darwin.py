from __future__ import print_function

import math

class Darwin(object):
    def __init__(self, population, serializer, evaluator, breeder):
        self.population = population
        self.serializer = serializer
        self.evaluator = evaluator
        self.breeder = breeder

        self.history = {}

    def evaluate(self, entropy):
        results = []
        for member in self.population:
            serialized = self.serializer(member)
            score = self.history.get(serialized)
            if score is None:
                score = self.evaluator(member, entropy)
                self.history[serialized] = (member, score)
        results.append((member, score))
        return sorted(results, reverse=True, key=lambda e: e[1])

    def repopulate(self, survival_rate, results, entropy):
        survivor_count = int(math.ceil(len(results) * survival_rate))
        survivors = results[:survivor_count]

        new_population = []
        while len(new_population) < len(self.population):
            parent = entropy.choice(survivors)[0]
            offspring = self.breeder(parent, entropy)
            new_population.append(offspring)
            print(len(new_population))
        self.population = new_population

    def best(self):
        return max(self.history.iteritems(), key=(lambda entry: entry[1][0]))[0]