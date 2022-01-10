'''this is a neuron
holds links to other neurons
modifies link strengths according to conditions
'''
import random
import time

from tqdm import tqdm
import numpy as np

from _old.activations import sigmoid

class Brain:
    def __init__(self, size, density, input_size, output_size):
        self.size = size
        self.input_size = input_size
        self.output_size = output_size
        self.density = density
        self.input_neurons = []
        self.output_neurons = []
        self.neurons = []
        self.synapses = []
        self.generate(size, density, input_size, output_size)

    def generate(self, size, density, input_size, output_size):
        self.neurons = [Neuron() for _ in range(size)]
        for i, n in enumerate(tqdm(self.neurons)):
            available_neurons = self.neurons[:i] + self.neurons[i+1:]
            num = int(density * len(available_neurons))
            target_neurons = random.sample(available_neurons, num)
            synapses = [Synapse(n, t) for t in target_neurons]
            self.synapses.extend(synapses)
            n.synapses.extend(synapses)
        
        self.input_neurons = self.neurons[:input_size]
        self.output_neurons = self.neurons[-output_size:]

    def step(self):
        for n in self.neurons:
            n.step()
        for s in self.synapses:
            s.step()

    def store(self, data):
        if len(data) != self.input_size:
            raise ValueError("data size does not match input size")
        for d, n in zip(data, self.input_neurons):
            n.excitation = d

    def load(self):
        return [n.excitation for n in self.output_neurons]

    def __repr__(self) -> str:
        return f"Brain: size: {self.size}, density: {self.density}"

class Neuron:
    def __init__(self, num_params, num_synapses) -> None:
        self.id = id(self)
        self.params = np.array(num_params)
        
        # synapse
        self.hebbians = np.array(num_params)
        self.synapse_strength = np.array(num_synapses)
        self.synapses_ids = np.array(num_synapses) # 1, 4, 5, 2, 7

    def set_synapses(self, ids):
        self.synapses = np.array(ids)

    def step(self):
        total = sum(
            self.excitation * synapse.strength * synapse.target.excitation
            for synapse in self.synapses
        )
        self.excitation = sigmoid(total)

class Synapse:
    def __init__(self, host, target) -> None:
        self.host = host
        self.target = target

        self.hebbian = 1.0
        self.strength = 0

    def step(self):
        self.strength = self.hebbian * self.target.excitation

if __name__ == "__main__":
    brain = Brain(size=1_000, density=0.01, input_size=5, output_size=5)
    print(brain)

    data = [random.random() for _ in range(brain.input_size)]
    print(data)
    brain.store(data)
    for _ in range(10):
        brain.step()
        print(brain.load())
