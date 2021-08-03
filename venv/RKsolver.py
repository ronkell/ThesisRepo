import random
import itertools
import numpy as np
from itertools import product
from timeit import default_timer as timer
from adapter import adapter
from gridBoxlinear import gridBoxlinear

class RKsolver():
    def __init__(self,problemname):
        self.problem_name=problemname
        self.problem_instance=None
        self.problem_adapter=None
        self.problem_dict_names={'boxMove':self.boxMoveSetup}
        setup=self.problem_dict_names[problemname]
        setup()

    def boxMoveSetup(self):
        self.problem_instance = gridBoxlinear(2, 2, 2, 1, 1, [(0, 1), (1, 0)], [(0, 0), (0, 0)])
        self.problem_adapter=adapter()
        self.problem_adapter.initstates(self.problem_instance.initialStateDisterbution)
        self.problem_adapter.initacitonsobs(self.problem_instance.actionSpace,self.problem_instance.observationSpace)



