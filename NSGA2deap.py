from MoeaBench.base_moea import BaseMoea


class NSGA2deap(BaseMoea):

  import random
  from deap import base, creator, tools, algorithms
  import array
  import numpy as np

  def __init__(self,problem=None,population=0,generations=0):
    self.problem=problem
    self.generations=generations
    self.population = population
    self.n_ieq= self.problem.get_CACHE().get_BENCH_CI().get_n_ieq_constr()

    self.creator.create("FitnessMin", self.base.Fitness, weights=(-1.0,) * self.problem.get_CACHE().get_BENCH_CI().get_M())
    self.creator.create("Individual", self.array.array, typecode='d', fitness=self.creator.FitnessMin)
    self.toolbox = self.base.Toolbox()
    self.toolbox.register("attr_float", self.uniform, 0, 1,self.problem.get_CACHE().get_BENCH_CI().get_Nvar())
    self.toolbox.register("individual", self.tools.initIterate, self.creator.Individual, self.toolbox.attr_float)
    self.toolbox.register("population", self.tools.initRepeat, list, self.toolbox.individual)
    self.toolbox.register("evaluate",self.evaluate)
    self.evalue = self.toolbox.evaluate
    self.random.seed(None)
    self.toolbox.decorate("evaluate", self.tools.DeltaPenality(self.feasible,1000))
    self.toolbox.register("mate", self.tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20)
    self.toolbox.register("mutate", self.tools.mutPolynomialBounded, low=0, up=1, eta=20, indpb=1/self.problem.get_CACHE().get_BENCH_CI().get_Nvar())
    self.toolbox.register("select", self.tools.selNSGA2)


  def uniform(self,low, up, size=None):
    try:
      return [self.random.uniform(a,b) for a,b in zip(low,up)]
    except TypeError as e:
      return [self.random.uniform(a,b) for a,b in zip([low]*size,[up]*size)]


  def evaluate(self,X):
    self.resul = self.problem.evaluation(self.np.array([X]),self.n_ieq)
    return self.resul['F'][0]


  def feasible(self,X):
    self.evaluate(X)
    if 'G' in self.resul:
      if self.resul["feasible"]:
       return True
    return False


  def evaluation(self):
    pop = self.toolbox.population(n=self.population)
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
    F_gen_all=[]
    X_gen_all=[]
    for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit
    F_gen_all.append(self.np.column_stack([self.np.array([ind.fitness.values for ind in pop ])]))
    X_gen_all.append(self.np.column_stack([self.np.array([self.np.array(ind) for ind in pop ])]))
    pop = self.toolbox.select(pop, len(pop))
    for gen in range(1, self.generations):
      offspring = self.tools.selTournamentDCD(pop, len(pop))
      offspring = [self.toolbox.clone(ind) for ind in offspring]
      for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if self.random.random() <= 0.9:
          self.toolbox.mate(ind1, ind2)
        self.toolbox.mutate(ind1)
        self.toolbox.mutate(ind2)
        del ind1.fitness.values, ind2.fitness.values
      invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
      fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
      for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
      pop = self.toolbox.select(pop + offspring, len(pop))
      F_gen_all.append(self.np.column_stack([self.np.array([ind.fitness.values for ind in pop ])]))
      X_gen_all.append(self.np.column_stack([self.np.array([self.np.array(ind) for ind in pop ])]))
    F = self.np.column_stack([self.np.array([ind.fitness.values for ind in pop ])])
    return F_gen_all,X_gen_all,F,self.generations,self.population