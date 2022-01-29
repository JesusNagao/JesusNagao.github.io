import pandas as pd
from deap import base, creator
from deap import algorithms
from deap import tools
import numpy as np
import random

p = pd.read_csv("p08_p.txt")
w = pd.read_csv("p08_w.txt")

w_max = 6404180


def func_eval(u):
    fu = np.sum(np.asarray(u) * np.asarray(p))
    return fu,

def feasible(u):
    wi = np.sum(np.asarray(u) * np.asarray(w))
    if (wi<w_max and wi>0):
        return True 
    else:
        return False

def distance(u):
    wi = np.sum(np.asarray(u) * np.asarray(w))
    return (wi-w_max)**2

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("select", tools.selRoulette)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("evaluate", func_eval)
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 65731.0, distance))
toolbox.register("attribute", random.randint, a=0, b=1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=10)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("std", np.std)

hof = tools.HallOfFame(3)
print('¿Cual algoritmo quieres correr?')
print('1. eaSimple')
print('2. eaMuPlusLambda')
print('3. eaMuCommaLambda')

num = int(input())

if(num == 1):
    log = algorithms.eaSimple(population=pop, toolbox=toolbox, halloffame=hof, cxpb=1.0, mutpb=1.0, ngen=1000, stats=stats, verbose=True)
elif(num == 2):
    log = algorithms.eaMuPlusLambda(population=pop, toolbox=toolbox, mu=10, lambda_=10 , cxpb=0.5, mutpb=0.5, ngen = 1000, stats=stats, halloffame=hof, verbose=True)
elif(num == 3):
    log = algorithms.eaMuCommaLambda(population=pop, toolbox=toolbox, mu=10, lambda_=10, cxpb=0.5, mutpb=0.5, ngen=1000, stats=stats, halloffame=hof, verbose=True)

print(hof)