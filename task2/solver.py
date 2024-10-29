import numpy as np
from abc import ABC, abstractmethod
from Parameters import Parameters
from Entity import Entity
from problem import problem1
import random
import copy


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

    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    def single_point_crossing(self, entity1, entity2):
        # @TODO starting from 1, because corssing that starts with first element does not make sense ??????
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
        # roulete selection
        # @TODO tak jest git ?
        sum_of_population_values = sum([x.value for x in population])

        new_population = np.array()
        for entity in population:
            probability_of_selection = (
                entity.value / sum_of_population_values
            )  # value in range from 0 to 1
            if probability_of_selection > random.random():
                new_population.append(entity)

        while len(new_population) != len(population):
            entity = np.random.choice(population)
            if probability_of_selection > random.random():
                new_population.append(entity)

        return new_population

        # @TODO FIX - new_population powinno byc zwracane, dupa
        # to samo jest w crossover, tylko robimy kopie
        # rozwiązanie kopiujemy population gidzes, czyscimy population i jedzeimy

    def crossover(self, population, problem):
        # single-point-crossing
        entities_to_cross = copy.copy(population)
        population.clear()
        while len(entities_to_cross) >= 1:
            entity1, entity2 = np.random.choice(
                entities_to_cross, size=2, replace=False
            )  # we do not want to cross the same entity with itself
            if self.parameters.probability_of_crossover > random.random():
                single_point_crossing(entity1, entity2)
                entity1.value = problem.evaluate(entity1.genome)
                entity2.value = problem.evaluate(entity2.genome)
            population.add(entity1)
            population.add(entity2)
            entities_to_cross.remove(entity1)
            entities_to_cross.remove(entity2)

        # @TODO fix - population size would decrease here, we need to add last element if left

    def mutate(self, population):

        for entity in population:
            for i in range(len(entity.genome)):
                if self.parameters.probability_of_mutation > random.random():
                    entity.genome[i] = not entity.genome[i]

    def find_the_best_entity(self, population):
        return np.max(population)

    def get_parameters(self):
        return self.parameters

    def get_missing_population_members(self, population, problem, wanted_size):
        members_to_add = []
        while len(population) + len(members_to_add) < wanted_size:
            entity_value = np.random.choice([True, False], size=problem.dimension)
            members_to_add.append(Entity(entity_value, problem.evaluate(entity_value)))
        return np.array(members_to_add)

    def solve(self, problem, initial_solutions=None):
        if initial_solutions is None:
            initial_solutions = np.array([], dtype=bool)
        if len(initial_solutions) != self.parameters.population_size:
            initial_solutions = np.concatenate(
                initial_solutions,
                self.get_missing_population_members(
                    initial_solutions, problem, self.parameters.population_size
                ),
            )

        # should we copy population ??
        population = initial_solutions

        result = self.find_the_best_entity(population)

        for i in range(self.parameters.number_of_generations):
            S = self.selection(population)
            self.crossover(S, problem)
            self.mutate(S, problem)

            # find the best
            best_of_population = self.find_the_best_entity(population)
            if result < best_of_population:
                result = best_of_population

            # succession
            population = S


def main():
    print("start..")

    pm = Parameters(10, 0.1, 0.1, 10)
    gm = Genetic_Algorithm(pm)
    gm.solve(problem1)

    print("finished")
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
