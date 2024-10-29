import numpy as np
from abc import ABC, abstractmethod
from Parameters import Parameters
from Entity import Entity
from Evaluator import evaluate
import random


class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    @abstractmethod
    def solve(self, problem, pop0, *args, **kwargs):
        """
        A method that solves the given problem for given initial solutions.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """
        ...


class Genetic_Algorithm(Solver):

    def __init__(self, parameters: Parameters, evaluator):
        self.parameters = parameters
        self.evaluator = evaluator

    def single_point_crossing(self, entity1, entity2):
        # starting from 1, because corssing that starts with first element does not make sense ??????
        crossing_idx = random.randint(1, len(entity1.genome) - 1)

        genome1 = entity1.genome
        genome2 = entity2.genome

        # Perform crossover
        entity1.genome = np.concatenate(
            (genome1[:crossing_idx], genome2[crossing_idx:])
        )
        entity2.genome = np.concatenate(
            (genome2[:crossing_idx], genome1[crossing_idx:])
        )

    def selection(self, population):
        sum_of_population_values = sum([x.value for x in population])

        for entity in population:
            probability_of_selection = entity.value / sum_of_population_values

        # selekcja ruletkowa

    def crossover(self, population):
        # single-point-crossing
        entities_to_cross = copy(population)
        population.clear()
        while len(entities_to_cross) >= 1:
            entity1, entity2 = np.random.choice(
                entities_to_cross, size=2, replace=False
            )  # we do not want to cross the same entity with itself
            if self.parameters.probability_of_crossover > random.random():
                single_point_crossing(entity1, entity2)
            population.add(entity1)
            population.add(entity2)

    def mutate(self, population):

        for entity in population:
            for i in range(len(entity.genome)):
                if self.parameters.probability_of_mutation > random.random():
                    entity.genome[i] = not entity.genome[i]

    def find_the_best_entity(self, population):
        return np.max(population)

    def get_parameters(self):
        return self.parameters

    def solve(self, problem, initial_solutions):
        # should we copy population ??
        population = initial_solutions

        result = find_the_best_entity(population)

        for i in range(self.parameters.number_of_generations):
            selection(population)
            crossover(population)
            mutate(population)

            # find the best
            best_of_population = find_the_best_entity(population)
            if result < best_of_population:
                result = best_of_population

            # succession


def main():

    pm = Parameters(1000, 0.1, 0.1)
    gm = Genetic_Algorithm(pm, evaluator=evaluate)

    # pop = np.array(
    #     [
    #         Entity(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1),
    #         Entity(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1),
    #         Entity(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1),
    #         Entity(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1),
    #     ]
    # )

    # result = gm.single_point_crossing(pop[0], pop[1])
    # print("a")


main()
