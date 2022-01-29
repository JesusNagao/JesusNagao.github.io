from deap import base, creator, algorithms, tools, gp
from deap.tools import History
import operator
import random
import numpy as np
from sklearn.metrics import mean_absolute_error
import math


entradas = range(1,2,0.01)
print(entradas)
salidas = [90, 82, 74, 66, 58, 50, 42, 34, 26, 18]

inp1, inp2 = zip(*entradas) 


toolbox = base.Toolbox()


def eval_func(ind, inputs1,inputs2, outputs):
    func_eval = toolbox.compile(expr=ind)
    predictions = list(map(func_eval, inputs1, inputs2))
    return mean_absolute_error(outputs, predictions),

def div(a,b):
    try:
        return a/b
    except:
        return 1

pset = gp.PrimitiveSet('Main', 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)

pset.renameArguments(ARG0='x0')
pset.renameArguments(ARG1='x1')
pset.addEphemeralConstant('R', lambda: random.randint(0,10))
pset.addTerminal(math.pi, name='pi')

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=3, max_=5)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('mutate', gp.mutNodeReplacement, pset=pset)
toolbox.register('evaluate', eval_func, inputs1=inp1, inputs2=inp2, outputs=salidas)
toolbox.register('compile', gp.compile, pset=pset)

toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("std", np.std)

hof = tools.HallOfFame(1)
pop = toolbox.population(n=10)

results, log  = algorithms.eaSimple(pop, toolbox, cxpb=1.0, mutpb=0.1, ngen=10, stats=stats, halloffame=hof)


for ind in hof:
    print(ind)