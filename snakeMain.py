# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:26:14 2019

@author: janwa
"""

import pygame 
import numpy as np
import keras 
import DQN
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt

WIDTH = 800
HEIGHT = 800
BLACK = (0,0,0)
SNAKE = (255,255,255)
BALL = (255,0,0)
HEAD = (255,255,0)
rowSize = 80
columnSize = 80
numChannels = 4

snakeLength = 2

epsilon = 1
epsilonDecayRate = 0.0002
minEpsilon = 0.001

nbEpoch = 300000
learningRate = 0.0001
memSize = 60000
gamma = 0.9
batchSize = 32

defaultReward = -0.1
negativeReward = -1
positiveReward =2

train = True


nRows = int(HEIGHT/rowSize)
nColumns = int(WIDTH/columnSize)
scrn = np.zeros((nRows, nColumns))

gameOver = False
currentSnakePos = list()
col = False
bth = 1
snakeLength -= 1
class Snake(object):
    
    def __init__(self, iS = (100,100,3), nO = 3, lr = 0.0005):
        self.learningRate = lr
        self.inputShape = iS
        self.numOutputs = nO
        self.model = Sequential() 
        
        self.model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = self.inputShape))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Conv2D(64, (2,2), activation = 'relu'))
        
        self.model.add(Flatten())
        
        self.model.add(Dense(units = 256, activation = 'relu'))
        
        self.model.add(Dense(units = self.numOutputs))
        self.model.compile(loss = 'mean_squared_error', optimizer = Adam(lr = self.learningRate))
        
    def loadModel(self, fp):
        self.model = load_model(fp)
        return self.model 
    
def drawSnake(direction):
    global snakeLength
    global currentSnakePos
    global gameOver
    global col
    
    if direction == -1:
        middleX = int(nColumns  / 2)
        middleY = int(nRows  / 2)
        for i in range(snakeLength):
            scrn[middleY + i][middleX] = 1
            currentSnakePos.append([middleY + i, middleX])
    elif direction == 1:
        x = currentSnakePos[0][1]
        y = currentSnakePos[0][0]
        if y - 1 >= 0:
            for i in range(len(currentSnakePos)):
                if currentSnakePos[i][0] == y - 1 and currentSnakePos[i][1] == x:
                    gameOver = True
            currentSnakePos.insert(0,([y-1,x]))
            for i in range(len(currentSnakePos)):
                if i > 0:
                    scrn[currentSnakePos[i][0]][currentSnakePos[i][1]] = 1
                else:
                    if bally == y-1 and ballx == x:
                        col = True
                    scrn[currentSnakePos[i][0]][currentSnakePos[i][1]] = 1
            
            if not col:
                del currentSnakePos[len(currentSnakePos) - 1]
            
        else:
            gameOver = True
    elif direction == 2:
        x = currentSnakePos[0][1]
        y = currentSnakePos[0][0]
        if x + 1 < nColumns:
            for i in range(len(currentSnakePos)):
                if currentSnakePos[i][1] == x + 1 and currentSnakePos[i][0] == y:
                    gameOver = True
            currentSnakePos.insert(0,([y,x+1]))
            for i in range(len(currentSnakePos)):
                if i > 0:
                    scrn[currentSnakePos[i][0]][currentSnakePos[i][1]] = 1
                else:
                    if bally == y and ballx == x+1:
                        col = True
                    scrn[currentSnakePos[i][0]][currentSnakePos[i][1]] = 1
            
            if not col:
                del currentSnakePos[len(currentSnakePos) - 1]
            
        else:
            gameOver = True
    elif direction == 3:
        x = currentSnakePos[0][1]
        y = currentSnakePos[0][0]
        if x - 1 >= 0:
            for i in range(len(currentSnakePos)):
                if currentSnakePos[i][1] == x - 1 and currentSnakePos[i][0] == y:
                    gameOver = True
            currentSnakePos.insert(0,([y,x-1]))
            for i in range(len(currentSnakePos)):
                if i > 0:
                    scrn[currentSnakePos[i][0]][currentSnakePos[i][1]] = 1
                else:
                    if bally == y and ballx == x-1:
                        col = True
                    scrn[currentSnakePos[i][0]][currentSnakePos[i][1]] = 1
            
            if not col:
                del currentSnakePos[len(currentSnakePos) - 1]
            
        else:
            gameOver = True
    elif direction == 4:
        x = currentSnakePos[0][1]
        y = currentSnakePos[0][0]
        if y + 1 < nRows:
            for i in range(len(currentSnakePos)):
                if currentSnakePos[i][0] == y + 1 and currentSnakePos[i][1] == x:
                    gameOver = True
            currentSnakePos.insert(0,([y+1,x]))
            for i in range(len(currentSnakePos)):
                if i > 0:
                    scrn[currentSnakePos[i][0]][currentSnakePos[i][1]] = 1
                else:
                    if bally == y+1 and ballx == x:
                        col = True
                    scrn[currentSnakePos[i][0]][currentSnakePos[i][1]] = 1
            
            if not col:
                del currentSnakePos[len(currentSnakePos) - 1]
            
        else:
            gameOver = True 
        
def drawBall():
    rnd = (np.random.randint(0,nRows), np.random.randint(0,nColumns))
    return rnd[0],rnd[1]
    
def mapArray():
    for row in range(nRows):
        for column in range(nColumns):
            x = column * columnSize
            y = row * rowSize
            if scrn[row][column] == 2:
                pygame.draw.rect(screen, BALL, (x + bth ,y + bth ,columnSize - (bth*2),rowSize - (bth*2)))
            elif scrn[row][column] == 1:
                pygame.draw.rect(screen, SNAKE, (x + bth,y + bth,columnSize - (bth*2),rowSize - (bth*2)))
            elif scrn[row][column] == 3:
                pygame.draw.rect(screen, HEAD, (x + bth,y + bth,columnSize - (bth*2),rowSize - (bth*2)))
                
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption('Snake')
screen.fill(BLACK)
direction = 1
drawSnake(-1)
bally, ballx = drawBall() 
start = True

snake = Snake((nRows, nColumns, numChannels), 4, learningRate)
model = snake.model
if not train:
    model = snake.loadModel('snakeBrain.h5')
reward = 0.0
nActions = 0
dqn = DQN.DQN(memSize, gamma)
highScore = 0
score = 0

results = []
nEpochsTotScore = 0.
maxQValue = []
lastMove = direction
for epoch in range(nbEpoch):
    currentSnakePos = list()
    nActions = 0
    gameOver = False
    currentState = np.zeros((1,nRows,nColumns,1))
    nextState = np.zeros((1,nRows,nColumns,1))
    for i in range(numChannels):
        currentState = np.concatenate((currentState, np.array(scrn.reshape((1,nRows,nColumns,1)) / 2)), axis = 3)
    currentState = np.delete(currentState, 0, axis = 3)
    for i in range(numChannels):
        nextState = np.concatenate((nextState, np.array(scrn.reshape((1,nRows,nColumns,1)) / 2)), axis = 3)
    nextState = np.delete(nextState, 0, axis = 3)
    loss = 0.0
    reward = 0.0
    drawSnake(-1)
    direction = 1
    scrn = np.zeros((nRows,nColumns))
    score = 0
    qValue = 0.
    rqv = 0.
    while not gameOver and nActions < 40000:
        nActions += 1
        if start:
            reward = defaultReward
            Qv = model.predict(currentState)[0]
            rqv += np.max(Qv)
            if (np.random.rand() <= epsilon and train):
                action = np.random.randint(0, 4) + 1
            else:
                
                action = np.argmax(Qv) + 1
            direction = action
            if action == 1 and lastMove == 4:
                direction = 4
            if action == 4 and lastMove == 1:
                direction = 1
            if action == 2 and lastMove == 3:
                direction = 3
            if action == 3 and lastMove == 2:
                direction = 2
            if nActions == 1:
                direction = 1
                #bally, ballx = drawBall()
            
#        for event in pygame.event.get():
#                if event.type == pygame.QUIT:
#                    gameOver = True
#                if event.type == pygame.KEYDOWN:
#                    if event.key == pygame.K_SPACE and not start:
#                        start = True
#                    if event.key == pygame.K_UP and direction != 4:
#                        direction = 1
#                    elif event.key == pygame.K_RIGHT and direction != 3:
#                        direction = 2
#                    elif event.key == pygame.K_LEFT and direction != 2:
#                        direction = 3
#                    elif event.key == pygame.K_DOWN and direction != 1:
#                        direction = 4
        
        if start:
            scrn[bally][ballx] = 2
            drawSnake(direction)  
            mapArray()
            
        if col:
            while scrn[bally][ballx] != 0:
                bally, ballx = drawBall() 
            scrn[bally][ballx] = 2
            reward = positiveReward
            score += 1
            nEpochsTotScore += 1
            col = False

            
        if gameOver:
            reward = negativeReward
        
        nextState = np.concatenate((nextState, scrn.reshape((1,nRows,nColumns,1)) / 2), axis = 3)
        nextState = np.delete(nextState, 0, axis = 3)
        dqn.remember([currentState, action - 1, reward, nextState], gameOver)
        if score > highScore and train: 
            highScore = score
            model.save('snakeBrain2.h5')
            
        lastMove = direction
        currentState = np.copy(nextState)
    
        pygame.display.flip()     
        if train:
            pygame.time.wait(0)
        else:
            pygame.time.wait(50)
        screen.fill(BLACK)
        scrn = np.zeros((nRows,nColumns))
        if train: 
            inputs,targets = dqn.getBatch(model, batchSize)
            loss += model.train_on_batch(inputs, targets)
            
    if epsilon > minEpsilon:
        epsilon -= epsilonDecayRate
    qValue += rqv / nActions
    if epoch % 100 == 0 and epoch != 0:
        results.append(nEpochsTotScore / 100)
        nEpochsTotScore = 0
        plt.plot(results)
        plt.ylabel('Average Score')
        plt.xlabel('Epoch / 100')
        plt.show()
        maxQValue.append(qValue/100)
        plt.plot(maxQValue)
        plt.ylabel('Avg Max Q Value')
        plt.xlabel('Epoch / 100')
        plt.show()
        qValue = 0.
    print('Avg Loss: ' + str(round(loss / nActions,3)) + ' Epoch: ' + str(epoch) + ' Best Score: ' + str(highScore) + ' Epsilon: ' + str(round(epsilon,5)))
    
    