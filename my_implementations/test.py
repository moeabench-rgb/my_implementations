from MoeaBench import mb
import os
import numpy as np



os.system("cls")  


exp = mb.experiment()
exp.benchmark = mb.benchmarks.DTLZ1()
exp.moea = mb.moeas.NSGA3(generations = 400, population = 190)
#exp.run()

opt_front = exp.optimal_front()
print(opt_front.ndim)


opt_set = exp.optimal_set()
print(opt_set.ndim)