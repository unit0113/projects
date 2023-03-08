import torch
import numpy as np
import gym
from matplotlib import pyplot as plt


def model(x, unpacked_params):
    l1,b1,l2,b2,l3,b3 = unpacked_params # Unpack the parameter vector into individual layer matrices
    y = torch.nn.functional.linear(x,l1,b1) # Simple linear layer with bias
    y = torch.relu(y) # Rectified linear unit activation function
    y = torch.nn.functional.linear(y,l2,b2)
    y = torch.relu(y)
    y = torch.nn.functional.linear(y,l3,b3)
    y = torch.log_softmax(y,dim=0) # The last layer will output log-probabilities over actions
    return y


def unpack_params(params, layers=[(25,4),(10,25),(2,10)]): # The `layers` parameter specifies the shape of each layer matrix
    unpacked_params = [] # Store each individual layer tensor
    e = 0
    for layer in layers: # Iterate through each layer
        s, e = e, e + np.prod(layer)
        weights = params[s:e].view(layer) # Unpack the indivudal layer into matrix form
        s,e = e, e + layer[0]
        bias = params[s:e]
        unpacked_params.extend([weights,bias]) # Add the unpacked tensor to the list
    return unpacked_params


def spawn_population(N=50, size=407): # `N` is the number of individuals in the population, `size` is the length of the parameter vectors
    pop = []
    for _ in range(N):
        vec = torch.randn(size) / 2.0 # Create a randomly initialized parameter vector
        fit = 0
        p = {'params':vec, 'fitness':fit} # Create a dictionary to store the parameter vector and its associated fitness score
        pop.append(p)
    return pop


def recombine(x1, x2): # x1 and x2 are agents which are dictionaries
    x1 = x1['params'] # Extract just the parameter vector
    x2 = x2['params']
    l = x1.shape[0]

    split_pt = np.random.randint(l) # Randomly produce a split or crossover point
    child1 = torch.zeros(l)
    child2 = torch.zeros(l)

    child1[0:split_pt] = x1[0:split_pt] #The first child is produced by taking the first segment of parent 1 and the second segment of parent 2
    child1[split_pt:] = x2[split_pt:]
    child2[0:split_pt] = x2[0:split_pt]
    child2[split_pt:] = x1[split_pt:]

    c1 = {'params':child1, 'fitness': 0.0} #Create new children agents by packaging the new parameter vectors into dictionaries
    c2 = {'params':child2, 'fitness': 0.0}
    return c1, c2


def mutate(x, rate=0.01): # `rate` is the mutation rate where 0.01 is a 1% mutation rate
    x_ = x['params']
    num_to_change = int(rate * x_.shape[0]) # Use the mutation rate to decide how many elements in the parameter vector to mutate
    idx = np.random.randint(low=0,high=x_.shape[0],size=(num_to_change,))
    x_[idx] = torch.randn(num_to_change) / 10.0 # Randomly reset the selected elements in the parameter vector
    x['params'] = x_
    return x


def test_model(agent, env):
    done = False
    state = torch.from_numpy(env.reset()).float()
    score = 0
    while not done: # While game is not lost
        params = unpack_params(agent['params'])
        probs = model(state,params) # Get the action probabilities from the model using the agent's parameter vector
        action = torch.distributions.Categorical(probs=probs).sample() # Probabilistically select an action by sampling from a categorical distribution
        state_, _, done, _ = env.step(action.item())
        state = torch.from_numpy(state_).float()
        score += 1 # Keep track of the number of time steps the game is not lost as the score
    return score



def evaluate_population(pop, env):
    tot_fit = 0 # Total fitness for this population; used to later calculate the average fitness of the population
    lp = len(pop)
    for agent in pop: # Iterate through each agent in the population
        score = test_model(agent, env) # Run the agent in the environment to assess its fitness
        agent['fitness'] = score # Store the fitness value
        tot_fit += score
    avg_fit = tot_fit / lp
    return pop, avg_fit


def next_generation(pop, mut_rate=0.001, tournament_size=0.2):
    new_pop = []
    lp = len(pop)
    while len(new_pop) < len(pop): # While the new population is not full
        rids = np.random.randint(low=0,high=lp,size=(int(tournament_size*lp))) # Select a percentage of the full population as a subset
        batch = np.array([[i,x['fitness']] for (i,x) in enumerate(pop) if i in rids]) # Subset the population to get a batch of agents and match each one with their index value in the original population
        scores = batch[batch[:, 1].argsort()] # Sort this batch in increasing order of score
        i0, i1 = int(scores[-1][0]),int(scores[-2][0]) # The last agents in the sorted batch are the agents with the highest scores; select the top 2 as parents
        parent0,parent1 = pop[i0],pop[i1]

        offspring_ = recombine(parent0,parent1) # Recombine the parents to get offspring
        child1 = mutate(offspring_[0], rate=mut_rate) # Mutate the children before putting them into the next generation
        child2 = mutate(offspring_[1], rate=mut_rate)
        offspring = [child1, child2]
        new_pop.extend(offspring)

    return new_pop


def running_mean(x,n=5):
    conv = np.ones(n)
    y = np.zeros(x.shape[0]-n)
    for i in range(x.shape[0]-n):
        y[i] = (conv @ x[i:i+n]) / n
    return y


def main():
    env = gym.make('CartPole-v0')
    num_generations = 20 # The number of generations to evolve
    population_size = 500 # The number of individuals in each generation
    mutation_rate = 0.01
    pop_fit = []
    pop = spawn_population(N=population_size, size=407) # Initialize a population
    for gen in range(num_generations):
        pop, avg_fit = evaluate_population(pop, env) # Evaluate the fitness of each agent in the population
        print(f'Generation {gen:>3}\tFitness: {avg_fit:.6f}')
        pop_fit.append(avg_fit)
        pop = next_generation(pop, mut_rate=mutation_rate,tournament_size=0.2) # Populate the next generation
    
    plt.figure(figsize=(12,7))
    plt.xlabel("Generations",fontsize=22)
    plt.ylabel("Score",fontsize=22)
    plt.plot(running_mean(np.array(pop_fit),3))
    plt.show()


if __name__ == '__main__':
    main()