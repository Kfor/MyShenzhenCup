# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:28:37 2019

@author: K
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


POP_SIZE = 200           # population size
CROSS_RATE = 0.9        # mating probability (DNA crossover)
MUTATION_RATE = 0.05    # mutation probability
N_GENERATIONS = 500


raw_data = []
    
N = 14 # 老师数

# 老师和班级包是在dna之外的label
SIZE_teacher_id = 4 # 一个老师的id由5位二进制数组成,0-32
SIZE_package_id = 8 # 由于一个老师最多带两个班，于是最多有老师数*2个班级包

# 以下是真正进行遗传算法的dna
SIZE_time = 3 # 由于有两个班级包，且只针对一天来进行排课（每周中各天等价），故用两个3位的数字来表示两个课的时段，0-8
DNA_SIZE_time = SIZE_time*2 
SIZE_classroom_id = 4
DNA_SIZE_classroom_id = SIZE_classroom_id*2 # 最多有16个教室，同理1个dna有两个教室
BIG_ONCE = (DNA_SIZE_classroom_id + DNA_SIZE_time)*2
DNA_TOTAL = BIG_ONCE*N # 完整大小，将所有老师拼成一条

def Conflict(result,value,index): # 对冲突的进行罚分
    
    for i in range(len(result)-1):
        for j in range(i+1,len(result)):
            
            this_crId1 = result[i][0]
            this_time1 = result[i][1]
            this_crId2 = result[i][2]
            this_time2 = result[i][3]
            
            crId1 = result[j][0]
            time1 = result[j][1]
            crId2 = result[j][2]
            time2 = result[j][3]
            
            if this_crId1 == crId1 and this_time1 == time1 or \
            this_crId2 == crId2 and this_time2 == time2 or \
            this_crId1 == crId2 and this_time1 == time2 or \
            this_crId2 == crId1 and this_time2 == time1:
                
                value[index] -= 5
                    
                if value[index] < 0:
                    value[index] = 0
                
                #return
            
                    
def F(pop):
    value = np.array([100]*POP_SIZE).reshape((POP_SIZE,1))
    for i in range(POP_SIZE):
        onepop = pop[i]
        result = translateDNA(onepop)
        
        for oneline in result:
            this_crId1 = oneline[0]
            this_time1 = oneline[1]
            this_crId2 = oneline[2]
            this_time2 = oneline[3]
            
            if this_crId1 == this_crId2 or this_time1 == this_time2:
                
                value[i] -= 20
                if value[i] < 0:
                    value[i] = 0
                    
            if abs(this_crId1-this_crId2) <= 1 :
                value[i] -= 10
            
            if abs(this_time1-this_time2) >= 5 :
                value[i] -= 10
            
            Conflict(result,value,i)
            
            if value[i] < 0:
                value[i] = 0
        
    return value     # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred): 
    return 1/(100 - pred + 1)


# convert binary DNA to decimal
def translateDNA(onepop):
    
    result = []
    for i in range(N):
                
        crId1 = onepop[0+i*BIG_ONCE:SIZE_classroom_id+i*BIG_ONCE]
        time1 = onepop[SIZE_classroom_id+i*BIG_ONCE:SIZE_classroom_id+SIZE_time+i*BIG_ONCE]
        crId_result1 = crId1.dot(2**np.arange(SIZE_classroom_id)[::-1])#16是教室max数
        time_result1 = time1.dot(2**np.arange(SIZE_time)[::-1])#4是最大时间段数
        
        crId2 = onepop[BIG_ONCE//2+i*BIG_ONCE:BIG_ONCE//2+SIZE_classroom_id+i*BIG_ONCE]
        time2 = onepop[BIG_ONCE//2+SIZE_classroom_id+i*BIG_ONCE:BIG_ONCE//2+SIZE_classroom_id+SIZE_time+i*BIG_ONCE]
        crId_result2 = crId2.dot(2**np.arange(SIZE_classroom_id)[::-1])#16是教室max数
        time_result2 = time2.dot(2**np.arange(SIZE_time)[::-1])#4是最大时间段数
        
        result.append([crId_result1,time_result1,crId_result2,time_result2])
        
    return result


def select(pop, fitness):    # 使用轮盘赌方法来随机选取dna
    
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum()
                           #p = 1/fitness.flatten()
                           )
    return pop[idx]


def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)  # select another individual from pop
        cross_points = np.random.randint(0, 2, 
                                         size=DNA_TOTAL
                                         ).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]    # mating and produce one child
    return parent


def mutate(child): # 变异函数，孩子的每一位若变异，就取反
    for point in range(DNA_TOTAL):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child

pop = np.random.randint(2, 
                        size=(POP_SIZE, 
                              DNA_TOTAL))   # initialize the pop DNA



l = []

for ri in range(N_GENERATIONS):
    
    F_values = F(pop)    # compute function value by extracting DNA
    
    # GA part (evolution)
    fitness = get_fitness(F_values).T[0]
    mmax = fitness.max()
    #print(ri,"Most fitted DNA: ", pop[np.argmax(fitness), :])
 
    print(ri,"%.3f"%(mmax))
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    l.append(mmax)
    
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child
       
        
plt.plot(l)


    


