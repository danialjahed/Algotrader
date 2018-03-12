import numpy as np
import pandas as pd
import math
import copy
import scipy.spatial


# Loading Data
trainData = pd.read_csv("Data/geneticInput/DJI_Train.csv")
# print(trainData.head())
# exit()

# Genetic Parameters
crossover_probability = 0.7
mutation_probability = 0.05
chromosome_number = 50
number_of_crossover = 2 * round((crossover_probability * chromosome_number) / 2)
number_of_mutation = round(mutation_probability * chromosome_number)
Iteration_Number = 50
# perssure
beta = 300


class chromosome:
    """
        buyValue_Down
        buyInterval_Down
        sellValue_Down
        sellInterval_Down
        buyValue_Up
        buyInterval_Up
        sellValue_Up
        sellInterval_Up
        fitnessValue
    """


    def __init__(self):

        self.fitnessValue = 0
        self.buyValue_Down = np.random.randint(5,40)
        self.buyInterval_Down = np.random.randint(5,20)
        self.sellValue_Down = np.random.randint(60,95)
        self.sellInterval_Down = np.random.randint(5,20)
        self.buyValue_Up = np.random.randint(5,40)
        self.buyInterval_Up = np.random.randint(5,20)
        self.sellValue_Up = np.random.randint(60,95)
        self.sellInterval_Up = np.random.randint(5,20)

    def fittness(self):
        buyPoint = 0
        sellPoint = 0
        gain = 0
        totalGain = 0
        shareNumber = 0
        moneyTemp = 0
        maximumMoney = 0
        minimumMoney = 10000
        maximumGain = 0
        maximumLost = 100
        totalPercentProfit = 0
        money=10000
        stopLoss = 0.05
        transactionCount = 0

        for i in range(len(trainData)):
            trend = trainData.trend[i]
            if trend == -1:
                # print(trend)
                pass
            elif trend :#Up trend
                # print(trend)
                # print(trainData['interval'+str(self.buyInterval_Up)][i],"******",self.buyValue_Up)
                # print(trainData['interval'+str(self.sellInterval_Up)][i],"******",self.sellValue_Up)
                if trainData['interval'+str(self.buyInterval_Up)][i] <= self.buyValue_Up:
                    # print(trainData['interval'+str(self.buyInterval_Up)][i],"******",self.buyValue_Up)
                    buyPoint = trainData.close[i]
                    buyPoint = buyPoint*100
                    shareNumber = (money-1)/buyPoint
                    forceSell = False
                    for j in range(i,len(trainData)):
                        sellPoint = trainData.close[j]
                        sellPoint = sellPoint*100
                        moneyTemp = (shareNumber*sellPoint)-1
                        if money*(1-stopLoss)>moneyTemp:
                            money=moneyTemp
                            forceSell=True

                        if trainData['interval'+str(self.sellInterval_Up)][i] <= self.sellValue_Up :
                            sellPoint = trainData.close[j]
                            sellPoint = sellPoint*100
                            gain = sellPoint - buyPoint
                            moneyTemp = (shareNumber*sellPoint) - 1
                            money = moneyTemp
                            if money > maximumMoney :
                                maximumMoney = money
                            if money < minimumMoney :
                                minimumMoney = money
                            transactionCount = transactionCount + 1

                            totalPercentProfit = totalPercentProfit + (gain/buyPoint)
                            i = j+1
                            totalGain = totalGain + gain
                            break


            else:#Down trend
                # print(trend)
                if trainData['interval'+str(self.buyInterval_Down)][i] <= self.buyValue_Down:
                    buyPoint = trainData.close[i]
                    buyPoint = buyPoint*100
                    shareNumber = (money-1)/buyPoint
                    forceSell = False
                    for j in range(i,len(trainData)):
                        sellPoint = trainData.close[j]
                        sellPoint = sellPoint*100
                        moneyTemp = (shareNumber*sellPoint)-1
                        if money*(1-stopLoss)>moneyTemp:
                            money=moneyTemp
                            forceSell=True

                        if trainData['interval'+str(self.sellInterval_Down)][i] <= self.sellValue_Down :
                            sellPoint = trainData.close[j]
                            sellPoint = sellPoint*100
                            gain = sellPoint - buyPoint
                            moneyTemp = (shareNumber*sellPoint) - 1
                            money = moneyTemp
                            if money > maximumMoney :
                                maximumMoney = money
                            if money < minimumMoney :
                                minimumMoney = money
                            transactionCount = transactionCount + 1
                            totalPercentProfit = totalPercentProfit + (gain/buyPoint)
                            i = j+1
                            totalGain = totalGain + gain
                            break

        self.fitnessValue = totalGain
        return totalGain

    #$print override function here$


c1 = chromosome()
c2 = chromosome()
c1.fittness()
c2.fittness()
# print(c2.fitnessValue,
#         c2.buyValue_Down,
#         c2.buyInterval_Down,
#         c2.sellValue_Down,
#         c2.sellInterval_Down,
#         c2.buyValue_Up,
#         c2.buyInterval_Up,
#         c2.sellValue_Up,
#         c2.sellInterval_Up,)
# exit()

# Cross over definition
def crossover_function(parent1, parent2):
    new_chromosome = chromosome()

    if (np.random.random_sample() <= crossover_probability):
        new_chromosome.buyValue_Down = parent1.buyValue_Down
    else:
        new_chromosome.buyValue_Down = parent2.buyValue_Down

    if (np.random.random_sample() <= crossover_probability):
        new_chromosome.buyInterval_Down = parent1.buyInterval_Down
    else:
        new_chromosome.buyInterval_Down = parent2.buyInterval_Down

    if (np.random.random_sample() <= crossover_probability):
        new_chromosome.sellValue_Down = parent1.sellValue_Down
    else:
        new_chromosome.sellValue_Down = parent2.sellValue_Down

    if (np.random.random_sample() <= crossover_probability):
        new_chromosome.sellInterval_Down = parent1.sellInterval_Down
    else:
        new_chromosome.sellInterval_Down = parent2.sellInterval_Down

    if (np.random.random_sample() <= crossover_probability):
        new_chromosome.buyValue_Up = parent1.buyValue_Up
    else:
        new_chromosome.buyValue_Up = parent2.buyValue_Up

    if (np.random.random_sample() <= crossover_probability):
        new_chromosome.buyInterval_Up = parent1.buyInterval_Up
    else:
        new_chromosome.buyInterval_Up = parent2.buyInterval_Up

    if (np.random.random_sample() <= crossover_probability):
        new_chromosome.sellValue_Up = parent1.sellValue_Up
    else:
        new_chromosome.sellValue_Up = parent2.sellValue_Up

    if (np.random.random_sample() <= crossover_probability):
        new_chromosome.sellInterval_Up = parent1.sellInterval_Up
    else:
        new_chromosome.sellInterval_Up = parent2.sellInterval_Up

    return new_chromosome



def mutation(mutate):
        new_chromosome = copy.deepcopy(mutate)

        if (np.random.random_sample() <= mutation_probability):
            new_chromosome.buyValue_Down = np.random.randint(5,95)

        if (np.random.random_sample() <= mutation_probability):
            new_chromosome.buyInterval_Down = np.random.randint(2,22)

        if (np.random.random_sample() <= mutation_probability):
            new_chromosome.sellValue_Down = np.random.randint(5,95)

        if (np.random.random_sample() <= mutation_probability):
            new_chromosome.sellInterval_Down = np.random.randint(2,22)

        if (np.random.random_sample() <= mutation_probability):
            new_chromosome.buyValue_Up = np.random.randint(5,95)

        if (np.random.random_sample() <= mutation_probability):
            new_chromosome.buyInterval_Up = np.random.randint(2,22)

        if (np.random.random_sample() <= mutation_probability):
            new_chromosome.sellValue_Up = np.random.randint(5,95)

        if (np.random.random_sample() <= mutation_probability):
            new_chromosome.sellInterval_Up = np.random.randint(2,22)

        return new_chromosome


newC = crossover_function(c1, c2)
mutatedC = mutation(newC)
# print(c1.fitnessValue,
#         c1.buyValue_Down,
#         c1.buyInterval_Down,
#         c1.sellValue_Down,
#         c1.sellInterval_Down,
#         c1.buyValue_Up,
#         c1.buyInterval_Up,
#         c1.sellValue_Up,
#         c1.sellInterval_Up,)
# print("***************c1******************")
# print(c2.fitnessValue,
#         c2.buyValue_Down,
#         c2.buyInterval_Down,
#         c2.sellValue_Down,
#         c2.sellInterval_Down,
#         c2.buyValue_Up,
#         c2.buyInterval_Up,
#         c2.sellValue_Up,
#         c2.sellInterval_Up,)
# print("***************c2******************")
# print(newC.fitnessValue,
#         newC.buyValue_Down,
#         newC.buyInterval_Down,
#         newC.sellValue_Down,
#         newC.sellInterval_Down,
#         newC.buyValue_Up,
#         newC.buyInterval_Up,
#         newC.sellValue_Up,
#         newC.sellInterval_Up,)
# print("***************newC******************")
# print(mutatedC.fitnessValue,
#         mutatedC.buyValue_Down,
#         mutatedC.buyInterval_Down,
#         mutatedC.sellValue_Down,
#         mutatedC.sellInterval_Down,
#         mutatedC.buyValue_Up,
#         mutatedC.buyInterval_Up,
#         mutatedC.sellValue_Up,
#         mutatedC.sellInterval_Up,)
# print("***************mutatedC******************")
# exit()


def Boltzman(Costs):
    # print(Costs)
    Costs.sort()
    worst_cost = Costs[Costs.size - 1]
    P = np.exp(-beta * Costs / worst_cost)
    P = P / np.sum(P)
    return P


def roulette_wheel_selection(P):
    rand = np.random.uniform()
    sum_all = np.cumsum(P)
    return np.where(rand <= sum_all)[0][0]


# Main Algorithm Loop

All_samples = np.array([chromosome() for _ in range(chromosome_number)])
best_answers = np.array([])
for i in range(0, Iteration_Number):
# while 1:
    Costs = np.array([])
    for j in range(chromosome_number):
        Costs = np.append(Costs, All_samples[j].fittness())
    # Cross Over
    # P = Boltzman(Costs)
    Cross_over_sample = np.array([])
    for j in range(int(number_of_crossover / 2)):
        # random_first_parent = roulette_wheel_selection(P)
        # random_second_parent = roulette_wheel_selection(P)
        random_first_parent = np.random.randint(0,50)
        random_second_parent = np.random.randint(0,50)
        children = crossover_function(All_samples[random_first_parent], All_samples[random_second_parent])
        Cross_over_sample = np.append(Cross_over_sample, children)
    # print("Cross Over is Done")
    # Mutation
    Mutation_Sample = np.array([])
    for j in range(int(number_of_mutation)):
        # random_mutation = roulette_wheel_selection(P)
        random_mutation = np.random.randint(0,50)
        mutated = mutation(All_samples[random_mutation])
        Mutation_Sample = np.append(Mutation_Sample, mutated)
    # print("Mutation is done")

    NewAllSamples = np.concatenate((All_samples, Cross_over_sample, Mutation_Sample), axis=0)
    NewAllSamples = sorted(NewAllSamples, key=lambda x: x.fitnessValue, reverse=True)
    All_samples = NewAllSamples[0:chromosome_number]
    best_answers = np.append(best_answers, All_samples[0].fitnessValue)
    if NewAllSamples[0].fitnessValue > 100000000000000000000000000000000000:#$40?$
        break
    print("This is the iteration number:", i)
    print("--------------------------")

print(All_samples[0].fitnessValue)
#$unroll chromosome here$
output_X_TrainData = []
output_Y_TrainData = []
for i in range(len(All_samples)):
    output_X_TrainData.append( [All_samples[i].buyValue_Down,All_samples[i].buyInterval_Down,0] )
    output_Y_TrainData.append([0,1,0])
    output_X_TrainData.append( [All_samples[i].sellValue_Down,All_samples[i].sellInterval_Down,0] )
    output_Y_TrainData.append([0,0,1])

    output_X_TrainData.append( [All_samples[i].buyValue_Up,All_samples[i].buyInterval_Up,1] )
    output_Y_TrainData.append([0,1,0])
    output_X_TrainData.append( [All_samples[i].sellValue_Up,All_samples[i].sellInterval_Up,1] )
    output_Y_TrainData.append([0,0,1])

    avgValue = np.ceil((All_samples[i].buyValue_Down + All_samples[i].sellValue_Down + All_samples[i].buyValue_Up + All_samples[i].sellValue_Up)/4)
    avgInterval = np.ceil((All_samples[i].buyInterval_Down + All_samples[i].sellInterval_Down + All_samples[i].buyInterval_Up + All_samples[i].sellInterval_Up)/4)
    output_X_TrainData.append( [avgValue+5 , avgInterval+2 , 0] )
    output_Y_TrainData.append([1,0,0])
    output_X_TrainData.append( [avgValue , avgInterval ,0] )
    output_Y_TrainData.append([1,0,0])

#$save CSV here$
output_X_TrainData = pd.DataFrame(output_X_TrainData)
output_X_TrainData.columns = ['RSIValue', 'RSIInterval', 'Trend']
output_X_TrainData.to_csv('Data/geneticOutput/output_X_TrainData.csv',index=False)
output_Y_TrainData = pd.DataFrame(output_Y_TrainData)
output_Y_TrainData.columns = ['hold', 'buy', 'sell']
output_Y_TrainData.to_csv('Data/geneticOutput/output_Y_TrainData.csv',index=False)
# plt.plot(range(Iteration_Number), best_answers)
# plt.savefig("fitnessValue.png")
