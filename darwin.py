from __future__ import print_function

import gc
import math

def descending_score(results):
    return sorted(results, reverse=True, key=lambda e: e[1])

class Darwin(object):
    def __init__(self, population, serializer, evaluator, breeder):
        self.population = population
        self.serializer = serializer
        self.evaluator = evaluator
        self.breeder = breeder

        self.history = {}

    def evaluate(self, entropy):
        results = []
        for index, member in enumerate(self.population):
            print("Evaluating", index)
            serialized = self.serializer(member)
            score = self.history.get(serialized)
            if score is None:
                score = self.evaluator(member, entropy)
                gc.collect()
                self.history[serialized] = (member, score)
            else:
                score = score[1]
            print("Score:", score)
            results.append((member, score))
        return descending_score(results)
    
    def repopulate(self, survival_rate, from_history, results, entropy):
        survivor_count = int(math.ceil(len(results) * survival_rate))
        survivors = results[:survivor_count]
        
        best_score = survivors[0][1]
        better = []
        for entry in self.history.values():
            if (entry[1] > best_score):
                better.append(entry)
        better = descending_score(better)
        
        survivors.extend(better[0:from_history])

        new_population = []
        while len(new_population) < len(self.population):
            parents = [
                entropy.choice(survivors)[0],
                entropy.choice(survivors)[0]
            ]
            offspring = self.breeder(parents, entropy)
            new_population.append(offspring)
        self.population = new_population

    def best(self):
        return max(self.history.iteritems(), key=(lambda entry: entry[1][1]))[1]