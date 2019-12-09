import time
import numpy as np

class Individual:
    def __init__(self, genes:list = []):
        self.chromosome = genes

    def randomize(self, geneSize:int):
        self.chromosome = np.random.randint(1, 30, geneSize)
        return self

    def __str__(self):
        return "{}".format(self.chromosome)

class GeneticAlgorithm:
    def __init__(self, constraint:int, populationSize:int, pc:float, pm:float, alpha:float):
        self.constraint = constraint
        self.populationSize = populationSize
        self.population = []
        self.pc = pc
        self.pm = pm
        self.alpha = alpha
        self.file = open('calculation.txt', 'w+')

        # initialize random population
        for i in range(0, populationSize):
            self.population.append(Individual().randomize(4))

        print(end="\n" + "=" * 20 + "\n", file=self.file)
        print("Initialization", end="\n" + "=" * 20 + "\n", file=self.file)
        
        print("Constraint : {} | Population Size : {} | Pc : {} | Pm : {}".format(self.constraint, self.populationSize, self.pc, self.pm), file=self.file)

        for population in self.population:
            print(population, file=self.file)

    def fitness(self, individual: Individual, lanes:list):
        fitness = 0
        
        for gene, lane in zip(individual.chromosome, lanes):
            if (gene - lane < 0):
                fitness += 500
            else:
                fitness += gene - lane

        return 1 / (fitness + 1)
        
    def selectionRWS(self, lanes):
        fitness = []
        probabilities = []
        cummProbabilities = []

        print("-> Fitness Evaluation", end="\n" + "-" * 25 + "\n", file=self.file)

        for individual in self.population:
            individualFitness = self.fitness(individual, lanes)
            fitness.append(individualFitness)
            print("{} | Fitness : {}".format(individual, individualFitness), file=self.file)

        for individualFitness in fitness:
            probabilities.append(individualFitness / sum(fitness))
        
        zipped = zip(self.population, probabilities)
        self.population, probabilities = zip(*sorted(zipped, key=lambda key: key[1])) # individuals are sorted
        self.population = list(self.population)

        for i in range(0, len(probabilities)):
            totalProb = 0
            for j in range(0, i + 1):
                totalProb += probabilities[j]
            cummProbabilities.append(totalProb)

        zipped = zip(self.population, cummProbabilities)
        _, cummProbabilities = zip(*sorted(zipped, key=lambda key: key[1]))
        
        print("-> RWS", end="\n" + "-" * 10 + "\n", file=self.file)
        rwsRandom = sorted(np.random.rand(int(self.pc * self.populationSize)))
        print("Random Picks :", rwsRandom, file=self.file)

        for cummProbability, individual in zip(cummProbabilities, self.population):
            print("{} | Cummulative Probability : {}".format(individual, cummProbability), file=self.file)

        selectIndex = []
        for rand in rwsRandom:
            counter = 0
            for cummProbability in cummProbabilities:
                if (rand < cummProbability):
                    selectIndex.append(counter)
                    break
                counter += 1

        print("Selected Index :", selectIndex, file=self.file)

        return selectIndex

    def crossover(self, selectIndex):
        selectIndex = selectIndex
        offsprings = []
        random = np.random.randint(0, 4)
        
        for i in range(0, int((self.pc * self.populationSize) / 2)):
            parent1 = np.asarray(self.population[selectIndex[2 * i]].chromosome)
            parent2 = np.asarray(self.population[selectIndex[2 * i + 1]].chromosome)

            # arithmetic recombination
            offspring = Individual(np.concatenate((parent1[:random] * self.alpha + parent2[:random] * (1 - self.alpha), parent1[random:] * self.alpha + parent2[random:] * (1 - self.alpha))))
            offspring = Individual([int(gene) for gene in offspring.chromosome]) # flooring
            offsprings.append(offspring)

            offspring = Individual(np.concatenate((parent2[:random] * self.alpha + parent1[:random] * (1 - self.alpha), parent1[random:] * (1 - self.alpha) + parent2[random:] * self.alpha)))
            offspring = Individual([int(gene) for gene in offspring.chromosome]) # flooring
            offsprings.append(offspring)

        print("-> Crossover Offsprings", end="\n" + "-" * 25 + "\n", file=self.file)
        print("Crossover point :", random, file=self.file)

        for offspring in offsprings:
            print(offspring, file=self.file)

        return offsprings

    def mutation(self, offsprings):
        offsprings = offsprings
        totalAlleles = int(self.pc * self.populationSize) * 4
        random = np.random.randint(0, totalAlleles, int(totalAlleles * self.pm))
        
        for number in random:
            offsprings[number // 4].chromosome[number % 4] = np.random.randint(1, 30)

        print("-> Mutation", end="\n" + "-" * 15 + "\n", file=self.file)
        print("Random mutation numbers", random, file=self.file)
        
        for offspring in offsprings:
            print(offspring, file=self.file)

        return offsprings

    def elitism(self, mutatedOffsprings, lanes):
        self.population = [i for i in self.population]
        offsprings = mutatedOffsprings
        self.population.extend(offsprings)
        
        fitness = []

        for individualFitness in [self.fitness(individual, lanes) for individual in self.population]:
            fitness.append(individualFitness)
        
        zipped = zip(self.population, fitness)
        self.population, fitness = zip(*sorted(zipped, key=lambda key: key[1], reverse = True)) # individuals are sorted
        self.population = list(self.population)

        print("-> Population before elitism", end="\n" + "-" * 15 + "\n", file=self.file)
        for individual in self.population:
            individualFitness = self.fitness(individual, lanes)
            print("{} | Fitness : {}".format(individual, individualFitness), file=self.file)

        self.population = self.population[: self.populationSize]
        fitness = fitness[: self.populationSize]

    def evolve(self, lanes: list):
        for i in range(0, 10):
            rwsResult = self.selectionRWS(lanes)
            offsprings = self.crossover(rwsResult)
            mutatedOffsprings = self.mutation(offsprings)
            self.elitism(mutatedOffsprings, lanes)
        return self.population[0]

lanes, allocations = [0, 0, 0, 0], Individual([0, 0, 0, 0])
turn = 1
constraint = 100
populationSize = 10
pc = 0.8
pm = 0.1
alpha = 0.3

t0 = time.time()
t = 0

while (True):
    if (round(time.time() - t0, 2) - t < 1.2):
        t = round(time.time() - t0, 2)
    else:
        print('Program too late')
        break

    if(allocations.chromosome[turn] == 0 or lanes[turn] < 1): # time runs out or lane empty, switch
        model = GeneticAlgorithm(constraint, populationSize, pc, pm, alpha)
        allocations = model.evolve(lanes)
        if turn + 1 > 3:
            turn = 0
        else:
            turn += 1
    else:
        allocations.chromosome[turn] -= 1
        if lanes[turn] >0:
            lanes[turn] -= 1
        
    if(np.random.rand()>0.5): # threshold probability of incoming cars, indicates traffic
        incoming = np.random.randint(0, 3, 4)
    else:
        incoming = [0, 0, 0, 0]

    for i in range(0, 4):
        lanes[i] += incoming[i]

    print(lanes, t, 'sec | turn :', turn, '| time allocation :', allocations, end='\r')

    time.sleep(1)
model.file.close()