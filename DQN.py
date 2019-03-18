# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:32:44 2019

@author: janwa
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:47:05 2019

@author: janwa
"""

import numpy as np

class DQN(object):
    
    def __init__(self, maxMemory, discount = 0.9):
        self.maxMemory = maxMemory
        self.discount = discount
        self.memory = list()
        
    def remember(self, transition, gameOver):
        self.memory.append([transition, gameOver])
        if len(self.memory) > self.maxMemory:
            del self.memory[0]
            
    def getBatch(self, model, batchSize):
        lenMemory = len(self.memory)
        numOutputs = model.output_shape[-1]
        inputs = np.zeros((min(lenMemory, batchSize), self.memory[0][0][0].shape[1],self.memory[0][0][0].shape[2],self.memory[0][0][0].shape[3]))
        targets = np.zeros((min(lenMemory, batchSize), numOutputs))
        for i, inx in enumerate(np.random.randint(0, lenMemory, size = min(lenMemory, batchSize))):
            currentState, action, reward, nextState = self.memory[inx][0]
            gameOver = self.memory[inx][1]
            inputs[i] = currentState
            targets[i] = model.predict(currentState)[0]
            Qsa = np.max(model.predict(nextState)[0])
            if gameOver:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Qsa
            
        return inputs, targets
    
    def backPropReward(self, nSteps, rewardDiscount):
        currentReward = 0.0
        for i in reversed(range(len(self.memory) - min(len(self.memory), nSteps),len(self.memory))):
            currentReward = self.memory[i][0][2] + rewardDiscount * currentReward
            self.memory[i][0][2] = currentReward
            