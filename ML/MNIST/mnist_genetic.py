import os
import torch
import torchvision.datasets as dset
from torch.distributions import Bernoulli
import torchvision.transforms as transforms
import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.stats import halfnorm


BATCH_SIZE = 100
NUM_GENERATIONS = 50
POP_SIZE = 100
MUATION_RATE = 0.001


class Individual:
    def __init__(self,param, fitness=0):
        self.param = param
        self.fitness = fitness


def spawn_population(param_size=(784,10),pop_size=1000):
    return [Individual(torch.randn(*param_size)) for i in range(pop_size)]


def evaluate_population(pop, train_loader, loss_fn):
    avg_fit = 0 #avg population fitness
    for individual in pop:
        x,y = next(iter(train_loader))
        pred = model(x.reshape(BATCH_SIZE,784),individual.param)
        loss = loss_fn(pred,y)
        fit = loss
        individual.fitness = 1.0 / fit
        avg_fit += fit
    avg_fit = avg_fit / len(pop)
    return pop, avg_fit


def recombine(x1,x2): #x1,x2 : Individual
    w1 = x1.param.view(-1) #flatten
    w2 = x2.param.view(-1)
    cross_pt = random.randint(0,w1.shape[0])
    child1 = torch.zeros(w1.shape)
    child2 = torch.zeros(w1.shape)
    child1[0:cross_pt] = w1[0:cross_pt]
    child1[cross_pt:] = w2[cross_pt:]
    child2[0:cross_pt] = w2[0:cross_pt]
    child2[cross_pt:] = w1[cross_pt:]
    child1 = child1.reshape(784,10)
    child2 = child2.reshape(784,10)
    c1 = Individual(child1)
    c2 = Individual(child2)
    return [c1,c2]


def mutate(pop, mut_rate=0.01):
    param_shape = pop[0].param.shape
    l = torch.zeros(*param_shape)
    l[:] = mut_rate
    m = Bernoulli(l)
    for individual in pop:
        mut_vector = m.sample() * torch.randn(*param_shape)
        individual.param = mut_vector + individual.param
    return pop


def seed_next_population(pop,pop_size=1000, mut_rate=0.01):
    new_pop = []
    while len(new_pop) < pop_size: #until new pop is full
        parents = random.choices(pop,k=2, weights=[x.fitness for x in pop])
        offspring = recombine(parents[0],parents[1])
        new_pop.extend(offspring)
    new_pop = mutate(new_pop,mut_rate)
    return new_pop


def model(x,W):
    return torch.nn.Softmax()(x @ W)


def main():
    root = os.path.join(os.path.dirname(__file__),'Data')
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

    x = next(iter(train_loader))[0]
    x = x.reshape(100,784)

    model(x,torch.rand(784,10))
    loss_fn = torch.nn.CrossEntropyLoss()
    pop = spawn_population()

    pop_fit = []
    pop = spawn_population(pop_size=POP_SIZE) #initial population
    for gen in range(NUM_GENERATIONS):
        # trainning
        pop, avg_fit = evaluate_population(pop, train_loader, loss_fn)
        print(avg_fit)
        pop_fit.append(avg_fit) #record population average fitness
        new_pop = seed_next_population(pop, pop_size=POP_SIZE, mut_rate=MUATION_RATE)
        pop = new_pop

    plt.plot(pop_fit)
    plt.show()


if __name__ == '__main__':
    main()
