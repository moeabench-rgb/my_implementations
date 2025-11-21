from MoeaBench.base_moea import BaseMoea
from MoeaBench.integration_moea import integration_moea
import random
from deap import base, creator, tools, algorithms
import array
import numpy as np

class my_NSGA2deap(integration_moea):
        
        def __init__(self,population = 160, generations = 300):
          self.population=population
          self.generations=generations
      
      
        def instance(self,problem):
          return NSGA2deap(problem,self.population,self.generations)
      

class NSGA2deap(BaseMoea):

  def __init__(self,problem=None,population = 160, generations = 300):
    super().__init__(problem,population,generations)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * self.get_M())
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    self.toolbox = base.Toolbox()
    self.toolbox.register("attr_float", self.uniform, 0, 1, self.get_N())
    self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_float)
    self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
    self.toolbox.register("evaluate",self.evaluate)
    self.evalue = self.toolbox.evaluate
    random.seed(None)
    self.toolbox.decorate("evaluate", tools.DeltaPenality(self.feasible,1000))
    self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20)
    self.toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20, indpb=1/self.get_N())
    self.toolbox.register("select", tools.selNSGA2)


  def uniform(self,low, up, size=None):
    try:
      return [random.uniform(a,b) for a,b in zip(low,up)]
    except TypeError as e:
      return [random.uniform(a,b) for a,b in zip([low]*size,[up]*size)]


  def evaluate(self,X):
    self.result = self.evaluation_benchmark(X)
    return self.result['F'][0]


  def feasible(self,X):
    self.evaluate(X)
    if 'G' in self.result:
      if self.result["feasible"]:
       return True
    return False
  

  def evaluation(self):
    pop = self.toolbox.population(n=self.get_population())
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
    F_gen_all=[]
    X_gen_all=[]
    for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit
    F_gen_all.append(np.column_stack([np.array([ind.fitness.values for ind in pop ])]))
    X_gen_all.append(np.column_stack([np.array([np.array(ind) for ind in pop ])]))
    pop = self.toolbox.select(pop, len(pop))
    for gen in range(1, self.get_generations()):
      offspring = tools.selTournamentDCD(pop, len(pop))
      offspring = [self.toolbox.clone(ind) for ind in offspring]
      for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() <= 0.9:
          self.toolbox.mate(ind1, ind2)
        self.toolbox.mutate(ind1)
        self.toolbox.mutate(ind2)
        del ind1.fitness.values, ind2.fitness.values
      invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
      fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
      for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
      pop = self.toolbox.select(pop + offspring, len(pop))
      F_gen_all.append(np.column_stack([np.array([ind.fitness.values for ind in pop ])]))
      X_gen_all.append(np.column_stack([np.array([np.array(ind) for ind in pop ])]))
    F = np.column_stack([np.array([ind.fitness.values for ind in pop ])])
    return F_gen_all,X_gen_all,F,self.get_generations(),self.get_population()





