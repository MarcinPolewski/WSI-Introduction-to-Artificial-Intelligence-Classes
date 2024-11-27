from dataclasses import dataclass


@dataclass
class Parameters:
    number_of_generations: int
    probability_of_mutation: float
    probability_of_crossover: float
    population_size: int
