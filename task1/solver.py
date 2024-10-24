from abc import ABC, abstractmethod
import numpy as np
import math
from dataclasses import dataclass
import time
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import copy


class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    @abstractmethod
    def solve(self, problem, x0, *args, **kwargs):
        """
        A method that solves the given problem for given initial solution.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        here function is the problem - mozemy podac odrazu gradient
        albo pythonową funckje, ktorą podajemy - tak lepiej

        my podajemy gradient do funxkji np w inicje albo w solve

        synPy - biblioteka do obliczania gradientów

        Jeśli gradient bedzie się dziwnie zachowywał, to warto zerknąc na wzory

        autograd - z pytorcha sam liczy

        """
        ...


@dataclass
class Hiperparameters:
    step: float
    number_of_iteration: int
    proximity: np.array
    find_minimum: bool


@dataclass
class SolverStep:
    x: np.array
    y: float
    gradient: float


class SolverStepLogger:
    def __init__(self):
        self.entries = []

    def add(self, solverStep):
        self.entries.append(solverStep)


class GradientSolver(Solver):

    def __init__(self, hiperparameters=None, solver_step_logger=None):
        self.hiperparameters = hiperparameters
        self.solver_step_logger = solver_step_logger

    def solve(self, function, function_gradient, x0, *args, **kwargs):

        x = copy.copy(x0)
        i = 0
        gradient_value = function_gradient(x)

        # iterate while legnth of gradient vector is greate than provimity or number of iterations has been reached
        self.solver_step_logger.add(
            SolverStep(
                copy.copy(x), function(x), copy.copy(np.linalg.norm(gradient_value))
            )
        )
        while (
            i < self.hiperparameters.number_of_iteration
            and np.linalg.norm(gradient_value) > self.hiperparameters.proximity
        ):

            deltaX = gradient_value * self.hiperparameters.step
            if self.hiperparameters.find_minimum:
                deltaX *= -1
            x += deltaX
            i += 1
            gradient_value = function_gradient(x)
            self.solver_step_logger.add(
                SolverStep(
                    copy.copy(x), function(x), copy.copy(np.linalg.norm(gradient_value))
                )
            )

        return x

    def get_parameters(self):

        return self.hiperparameters


def exp(x):
    return (math.e) ** x


def exampleFunction1(x):
    return 0.5 * (x**4) + x


def gradientOfFunction1(x):
    return 2 * (x**3) + 1


# int this funciton x is a 2d vector


def exampleFunction2(x):
    x1 = x[0]
    x2 = x[1]

    return (
        1
        - 0.6 * exp(-(x1**2) - ((x2 + 1) ** 2))
        - 0.4 * exp(-((x1 - 1.75) ** 2) - ((x2 + 2) ** 2))
    )


def gradientOfFunciton2(x):
    x1 = x[0]
    x2 = x[1]

    gradientXValue = 1.2 * x1 * exp(-(x1**2) - ((x2 + 1) ** 2)) + 0.8 * (
        x1 - 1.75
    ) * exp(-((x1 - 1.75) ** 2) - ((x2 + 2) ** 2))
    gradientYValue = 1.2 * (x2 + 1) * exp(-(x1**2) - ((x2 + 1) ** 2)) + 0.8 * (
        x2 + 2
    ) * exp(-((x1 - 1.75) ** 2) - ((x2 + 2) ** 2))

    # tak czy na odwróc ??????
    return np.array([gradientXValue, gradientYValue])


class experimentRunner:
    def __init__(self):
        self.step_logger = SolverStepLogger()
        self.solver = GradientSolver(solver_step_logger=self.step_logger)

    def run_experiment(self, hiperparameters, start_x, function, gradient):
        self.solver.hiperparameters = hiperparameters
        self.step_logger.entries.clear()

        result = self.solver.solve(function, gradient, start_x)

        steps = np.array([entry.x for entry in self.step_logger.entries])
        values = np.array([entry.y for entry in self.step_logger.entries])

        if steps.size != 0:
            if steps[0].size == 1:
                # plot function
                x = np.linspace(-2, 2, 100)
                plt.plot(x, function(x), color="red")

                # plot path of gradient
                for i in range(values.size):
                    plt.scatter(
                        steps[i][0], values[i][0]
                    )  # values[i][0] because np array is reutrned from one dimentional function
            elif steps[0].size == 2:
                # plot function
                ax = plt.axes(projection="3d")
                x = np.linspace(-10, 10, 100)
                y = np.linspace(-10, 10, 100)
                x, y = np.meshgrid(x, y)
                z = function(np.array([x, y]))
                ax.plot_surface(x, y, z, cmap="viridis", alpha=0.5)

                # plot path of gradient
                for i in range(0, values.size, 10):
                    ax.scatter(steps[i][0], steps[i][1], values[i], color="red", s=10)
                    print(steps[i][0], steps[i][1], values[i])

        plt.show()

    def start_experiments(self):
        # hiperparameters = Hiperparameters(0.01,  1000.0, True)
        # start_x = np.array([0.0])
        # self.run_experiment(
        #     hiperparameters, start_x, exampleFunction1, gradientOfFunction1
        # )

        hiperparameters = Hiperparameters(0.0001, 10000, 0.001, True)
        start_x = np.array([2.0, -2.0])
        self.run_experiment(
            hiperparameters, start_x, exampleFunction2, gradientOfFunciton2
        )


def main():
    runner = experimentRunner()
    runner.start_experiments()


if __name__ == "__main__":
    main()
