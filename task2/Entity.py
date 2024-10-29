class Entity:
    def __init__(self, genome, value):
        self.genome = genome
        self.value = value

    def __gt__(self, other_entity):
        return self.value > other_entity.value

    def __le__(self, other_entity):
        return self.value < other_entity.value
