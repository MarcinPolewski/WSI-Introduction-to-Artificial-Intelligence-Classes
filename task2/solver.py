import numpy as np
from abc import ABC, abstractmethod
from Parameters import Parameters
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

    def single_point_crossing(self, genome1, genome2):
        # @TODO starting from 1, because corssing that starts with first element does not make sense ??????
        crossing_idx = random.randint(1, len(genome1) - 1)

        # Perform crossover
        genome1 = np.concatenate((genome1[:crossing_idx], genome2[crossing_idx:]))
        genome2 = np.concatenate((genome2[:crossing_idx], genome1[crossing_idx:]))

    def selection(self, population, scores):
        # roulete selection

        sum_of_population_values = np.sum(scores)
        population_size = len(population)
        new_population = np.empty((len(population), len(population[0])), dtype=bool)
        # new_population = np.empty((population_size), dtype=np.ndarray)
        new_population_idx = 0

        while new_population_idx != population_size:
            random_value = random.random()  # value from 0 to 1
            sum_of_prevoius_probabilities = 0
            idx = 0
            while (
                sum_of_prevoius_probabilities < random_value and idx < population_size
            ):
                sum_of_prevoius_probabilities += scores[idx] / sum_of_population_values
                idx += 1

            # now idx is the chosen value
            new_population[new_population_idx] = population[idx - 1]
            new_population_idx += 1

        return new_population

    def crossover(self, population):
        entities_to_cross = list(population)

        new_population = np.empty((len(population), len(population[0])), dtype=bool)
        # new_population = np.empty((len(population)), dtype=np.ndarray)
        new_population_idx = 0

        while len(entities_to_cross) >= 1:
            idx1, idx2 = random.sample(range(len(entities_to_cross)), 2)
            genome1 = population[idx1]
            genome2 = population[idx2]

            if self.parameters.probability_of_crossover > random.random():
                crossing_idx = random.randint(1, len(genome1) - 1)
                # Perform crossover
                genome1 = np.concatenate(
                    (genome1[:crossing_idx], genome2[crossing_idx:])
                )
                genome2 = np.concatenate(
                    (genome2[:crossing_idx], genome1[crossing_idx:])
                )

            new_population[new_population_idx] = genome1
            new_population_idx += 1
            new_population[new_population_idx] = genome2
            new_population_idx += 1

            entities_to_cross.pop(max(idx1, idx2))  # Remove the higher index first
            entities_to_cross.pop(min(idx1, idx2))  # Remove the lower index

            # entities_to_cross = [
            #     x
            #     for x in entities_to_cross
            #     if (x is not population[idx1] and x is not population[idx2])
            # ]
            # print()

            # entities_to_cross = list(
            #     filter(
            #         lambda x: (x is not population[idx1])
            #         and (x is not population[idx2]),
            #         entities_to_cross,
            #     )  # @TODO FIX this - some times it only deletes one element
            # )

        if len(entities_to_cross) == 1:
            new_population[new_population_idx] = entities_to_cross[0]

        return new_population

    def mutate(self, population):

        for genome in population:
            for i in range(len(genome)):
                if self.parameters.probability_of_mutation > random.random():
                    genome[i] = not genome[i]

    def find_the_best_entity(self, population, scores):
        best_score_idx = 0
        population_size = len(population)
        for i in range(1, population_size):
            if scores[i] > scores[best_score_idx]:
                best_score_idx = i

        return population[i], scores[i]

    def get_parameters(self):
        return self.parameters

    def get_missing_population_members(self, problem, how_many):
        members_to_add = []
        for _ in range(how_many):
            members_to_add.append(
                np.random.choice([True, False], size=problem.dimension)
            )
        return np.array(members_to_add)

    def solve(self, problem, initial_solutions=None):
        if initial_solutions is None:
            initial_solutions = self.get_missing_population_members(
                problem, self.parameters.population_size
            )

        elif len(initial_solutions) != self.parameters.population_size:
            initial_solutions = np.concatenate(
                (
                    initial_solutions,
                    self.get_missing_population_members(
                        problem,
                        self.parameters.population_size - len(initial_solutions),
                    ),
                )
            )

        # should we copy population ??
        population = initial_solutions
        scores = problem.evaluate(population)

        result, result_score = self.find_the_best_entity(population, scores)

        for i in range(self.parameters.number_of_generations):
            S = self.selection(population, scores)
            C = self.crossover(S)
            self.mutate(C)
            scores = problem.evaluate(C)

            # find the best
            best_of_population, best_of_population_score = self.find_the_best_entity(
                population, scores
            )
            # print(best_of_population_score)

            if result_score < best_of_population_score:
                result = copy.copy(best_of_population)
                result_score = copy.copy(best_of_population_score)
                # print("better found^")

            # succession
            population = C

        return (result, result_score)


def main():
    print("start..")

    # pm = Parameters(1000, 0.8, 0.02, 100)
    # gm = Genetic_Algorithm(pm)
    # found_value, score = gm.solve(problem1)
    # print(found_value, score)

    gm = Genetic_Algorithm(Parameters(1000, 0.8, 0.02, 100))
    for number_of_iterations in {100, 500, 1000}:
        for probability_of_cross in np.arange(0.1, 0.5, 0.1):
            for probability_of_mutation in {0.005, 0.01, 0.02, 0.05, 0.1}:
                for size_of_population in {50, 100, 500, 1000, 5000}:
                    sum_of_scores = 0
                    for _ in range(10):
                        gm.parameters = Parameters(
                            number_of_iterations,
                            probability_of_cross,
                            probability_of_mutation,
                            size_of_population,
                        )
                        found_value, score = gm.solve(problem1)
                        sum_of_scores += score

                    avg_score = sum_of_scores / 10

                    print(
                        number_of_iterations,
                        probability_of_cross,
                        probability_of_mutation,
                        size_of_population,
                        avg_score,
                    )
    print("finished")


main()
