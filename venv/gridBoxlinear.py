import random
import itertools
import numpy as np
from itertools import product
from timeit import default_timer as timer
from adapter import adapter
#actions per agent (0 idle),(1 left),(2 right),(3 up) (4 down) (5-x where x=5+number of boxes sense) (x+1-y where y=4*number of small boxes) (y+1 z where z=4*number of big boxes)
#obs per agent (0 idle) (1 no box) (2 yes box)
class gridBoxlinear():
    def __init__(self,numberOfAgents,grid_width,grid_height,numberOfSmallBoxes,numberOfBigBoxes,agents_init_locations,boxes_init_locations):
        self.grid_width=grid_width
        self.grid_height = grid_height
        self.numberOfAgents=numberOfAgents
        self.numberOfSmallBoxes=numberOfSmallBoxes
        self.numberOfBigBoxes=numberOfBigBoxes
        self.initialState=agents_init_locations+boxes_init_locations
        #self.realState=self.initialState.copy()
        self.objects_indexes=list(range(0,numberOfAgents+numberOfSmallBoxes+numberOfBigBoxes))
        move_idle_actions=list(range(0,5))
        sense_actions_perbox=list(range(5,5+numberOfSmallBoxes+numberOfBigBoxes))
        pushstartindex=sense_actions_perbox[-1]+1
        push_actions_perbox=list(range(pushstartindex,pushstartindex+4*numberOfSmallBoxes))
        if len(push_actions_perbox)>0:
            collabpushstartindex=push_actions_perbox[-1]+1
        else:
            collabpushstartindex=pushstartindex
        collab_push_perbox=list(range(collabpushstartindex,collabpushstartindex+4*numberOfBigBoxes))
        self.actions_per_agent=move_idle_actions+sense_actions_perbox+push_actions_perbox+collab_push_perbox
        actions_linear=move_idle_actions+sense_actions_perbox+push_actions_perbox
        actions_parelal=[(x,x) for x in collab_push_perbox]
        product_actions1 =list(product(actions_linear,[0]))
        product_actions2 = list(product([0], actions_linear[1:]))
        #push_collab_test_eliran=list(product(collab_push_perbox,collab_push_perbox))
        self.actionSpace=product_actions1+product_actions2+actions_parelal
        self.observationSpace=[ele for ele in product(range(0,3), repeat=numberOfAgents)]
        self.counter=numberOfSmallBoxes+numberOfBigBoxes
        self.epsilon=0.1
        self.countgood=0
        self.countbad=0
        self.initialStateDisterbution=[]


