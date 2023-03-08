import random
from matplotlib import pyplot as plt


ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.! "
NUM_GENERATIONS = 200
POP_SIZE = 3000
MUTATION_RATE = 0.001


class Individual:
    def __init__(self, string, fitness=0):
        self.string = string
        self.fitness = fitness
from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def spawn_population(length=26,size=100):
    pop = []
    for i in range(size):
        string = ''.join(random.choices(ALPHABET,k=length))
        individual = Individual(string)
        pop.append(individual)
    return pop


def mutate(x, mut_rate=0.01):
    new_x_ = []
    for char in x.string:
        if random.random() < mut_rate:
            new_x_.extend(random.choices(ALPHABET,k=1))
        else:
            new_x_.append(char)
    new_x = Individual(''.join(new_x_))
    return new_x


def recombine(p1_, p2_): #produces two children from two parents
    p1 = p1_.string
    p2 = p2_.string
    child1 = []
    child2 = []
    cross_pt = random.randint(0,len(p1))
    child1.extend(p1[0:cross_pt])
    child1.extend(p2[cross_pt:])
    child2.extend(p2[0:cross_pt])
    child2.extend(p1[cross_pt:])
    c1 = Individual(''.join(child1))
    c2 = Individual(''.join(child2))
    return c1, c2


def evaluate_population(pop, target):
    avg_fit = 0
    for i in range(len(pop)):
        fit = similar(pop[i].string, target)
        pop[i].fitness = fit
        avg_fit += fit
    avg_fit /= len(pop)
    return pop, avg_fit


def next_generation(pop, size=100, length=26, mut_rate=0.01):
    new_pop = []
    while len(new_pop) < size:
        parents = random.choices(pop,k=2, weights=[x.fitness for x in pop])
        offspring_ = recombine(parents[0],parents[1])
        offspring = [mutate(offspring_[0], mut_rate=mut_rate), mutate(offspring_[1], mut_rate=mut_rate)]
        new_pop.extend(offspring) #add offspring to next generation
    return new_pop


def main():
    target = "Hello World!"
    target_len = len(target)
    pop = spawn_population(length=target_len)
    
    pop_fit = []
    pop = spawn_population(size=POP_SIZE, length=target_len) #initial population
    for gen in range(NUM_GENERATIONS):
        # trainning
        pop, avg_fit = evaluate_population(pop, target)
        print(f'Generation {gen:>3}: Fitness: {avg_fit:.6f}')
        pop_fit.append(avg_fit) #record population average fitness
        new_pop = next_generation(pop, size=POP_SIZE, length=target_len, mut_rate=MUTATION_RATE)
        pop = new_pop

    plt.figure(figsize=(10,5))
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.plot(pop_fit)
    plt.show()

    pop.sort(key=lambda x: x.fitness, reverse=True)
    print(pop[0].string)


if __name__ == '__main__':
    main()
