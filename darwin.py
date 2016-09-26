from __future__ import print_function

import gc
import math

def descending_score(results):
    """Sort results by second element in reverse."""
    return sorted(results, reverse=True, key=lambda e: e[1])

class Darwin(object):
    """Evolves a population."""

    def __init__(self, serializer, evaluator, breeder):
        """Configure the evolver.
        Keyword Arguments:
        serializer -- function to convert a member of the population to it's canonical representation
        evaluator -- function to score a member of the population given a source of entropy
        breeder -- function to take one or two population members and produce an offspring
        """
        self.population = []
        self.serializer = serializer
        self.evaluator = evaluator
        self.breeder = breeder

        self.history = {}

    def is_clone(self, parents, offspring):
        """Check if the offspring is an exact match for one of it's parents"""
        offspring = self.serializer(offspring)
        for parent in parents:
            if offspring == self.serializer(parent):
                return True
        return False

    def init_population(self, prototypes, population_target, include_prototypes, breed_options, entropy):
        """Given a set of prototype members, fill out the population."""
        new_population = []
        if include_prototypes:
            new_population = list(prototypes)
            prototypes = new_population
            
        while len(new_population) < population_target:
            parents = [
                entropy.choice(prototypes),
                entropy.choice(prototypes)
            ]
            offspring = self.breeder(parents, breed_options, entropy)
            if self.is_clone(parents, offspring):
                print("Offspring is clone.")
            else:
                new_population.append(offspring)
        self.population = new_population
        return self.population

    def evaluate(self, entropy):
        """Evaluate the entire population."""
        self.results = []
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
            self.results.append((member, score))
        return self.sorted_results()

    def sorted_results(self):
        """Get the last evaluation results ordered by descending score."""
        return descending_score(self.results)

    def load_history(self, results):
        """Restore history results."""
        for member, score in results:
            self.history[self.serializer(member)] = (member, score)

    def repopulate(self, population_target, survival_rate, from_history, results, options, entropy):
        """Cull the population based on the survival_rate and then refill up to the target."""
        survivor_count = int(math.ceil(len(results) * survival_rate))
        survivors = results[:survivor_count]
        
        best_score = survivors[0][1]
        better = []
        for entry in self.history.values():
            if (entry[1] > best_score):
                better.append(entry)
        better = descending_score(better)
        
        survivors.extend(better[0:from_history])
        
        if not population_target:
            population_target = len(population)

        self.init_population([m for m, s in survivors], population_target, False, options, entropy)
        return self.population

    def best(self):
        """Get the best all time result."""
        return max(self.history.items(), key=(lambda entry: entry[1][1]))[1]
