import numpy as np
import pygame
from time import time,sleep
from random import randint as r
import random
import pickle

n = 7
scrx = n*100
scry = n*100
background = (51,51,51)

screen = pygame.display.set_mode((scrx,scry))
# colors = [(51,51,51),(51,51,51),(255,0,0),(51,51,51),(255,0,0),(255,255,0),(51,51,51),(255,0,0),(51,51,51),(255,0,0),(255,255,0),(51,51,51)
# ,(51,51,51),(0,255,0),(51,51,51),(255,255,0)]
# reward = np.array([[-1,-5,-1,-1],[-1,1,-5,10],[-5,-1,1,-1],[-1,-5,-1,1]])
# terminals = [1,6,7,8,13]
colors = [(51,51,51),(51,51,51),(51,51,51),(51,51,51),(51,51,51),(255,0,0),(255,0,0),(51,51,51),(51,51,51),(51,51,51),(0,255,0),(51,51,51)
,(51,51,51),(51,51,51),(51,51,51),(51,51,51)]
reward = np.array([[0,0,0,0],[0,-1,0,0],[0,-1,1,0],[0,0,0,0]])
terminals = [5,9,10]
colors = [(51,51,51) for i in range(n**2)]
reward = np.zeros((n,n))
# goals = 1
terminals = []
penalities = 10

c = 0
for k in range(10000):
    for i in range(0,scrx,100):
        for j in range(0,scry,100):
            pygame.draw.rect(screen,(255,255,255),(j,i,j+100,i+100),0)
            pygame.draw.rect(screen,colors[c],(j+3,i+3,j+95,i+95),0)
            c+=1