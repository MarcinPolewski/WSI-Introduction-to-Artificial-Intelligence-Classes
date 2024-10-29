import numpy as np


class problem1:

    dimension = 200

    @staticmethod
    def evaluate(x):
        temperature = np.zeros_like(x[..., 0]) + 1500
        obj_temperature = np.zeros_like(x[..., 0])

        temperatures = [temperature]
        for t, v in enumerate(np.transpose(x)):
            delta_t = temperature
            delta_e = temperature - obj_temperature
            if 30 <= t <= 50:
                # door opened
                temperature = temperature - delta_t * 0.03
            else:
                # door closed
                temperature = temperature - delta_t * 0.01
            if t >= 40:
                # object in oven
                temperature = temperature - delta_e * 0.025
                obj_temperature = obj_temperature + delta_e * 0.0125
            temperature = temperature + 50 * v
            temperatures.append(temperature)

        temperatures = np.array(temperatures).T
        cost = np.sum(x, axis=-1)
        diffs = np.sum(((temperatures[..., 40:] - 1500) / 500) ** 2, axis=-1)
        return -cost - diffs
