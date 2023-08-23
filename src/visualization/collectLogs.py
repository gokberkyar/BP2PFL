import pickle
from random import random
import sys
import docker
import json
from collections import defaultdict, deque
import os
from p2pml.commons import NUM_ROUNDS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy
from p2pml.commons import OUTPUT_DIR, PROJECT_DIR, DATA_PATH,NUM_ROUNDS
import p2pml.utils as utils
import networkx as nx
from matplotlib import pyplot as plt, test

def collectLogsDocker(peerCount):
    client = docker.from_env()
    history = defaultdict(list)
    for peer in range(peerCount):
        container = client.containers.get(f'peer{peer}')
        log = container.logs().decode().splitlines()

        for line in log:
            search_key ='evaluation: ' 
            pos = line.find(search_key)
            if pos != -1:
                evalution_dict = json.loads(line[pos+ len(search_key):].replace("'", '"'))
                history[peer].append(evalution_dict)
    return history


def collectEvaluationFiles(peerCount, numRound, experimentNumber):

    def getFilePath(peer, round_ind):
        filePath  = os.path.join(PROJECT_DIR,f'experiment_{experimentNumber}', f'Peers/peer-{peer}/Evaluation/evaluation-{round_ind}.json')
        return os.path.join(filePath)

    # history[round_ind][peer] = {key:value}
    history = defaultdict(dict)
    for round_ind in range(1, numRound+1):
        # print(round_ind)
        for peer in range(peerCount):
            filePath = getFilePath(peer,round_ind)
            if not os.path.isfile(filePath): continue
            with open(filePath, 'r') as f:
                data = json.load(f)
                history[round_ind][peer]=data
        
    return history


def hopDistance(experimentNumber):
    config = utils.readConfig(experimentNumber)
    graph = utils.getGraph(config, False)
    numberNodes = len(graph.nodes)

    adversarialNodes = utils.getAdversarialNodes(config,graph,numberNodes, True)
    
    hopDistance = [0]*numberNodes

    for node in range(numberNodes):

        visited = [False ] * numberNodes
        queue = deque()
        queue.append((node,0))
        while queue:
            top, depth = queue.popleft()

            if top in adversarialNodes:
                hopDistance[node] = depth
                # print(depth)
                break
            edges = graph.edges(top)
            for _, y in edges:
                if not visited[y]:
                        queue.append((y, depth+1))
                        visited[y] = True
    
    result = defaultdict(list)
    for node, hop in enumerate(hopDistance):
        result[hop].append(node)
    return result

def hopDistanceAcc(experimentNumber, acc_type, plot=True):
    config = utils.readConfig(experimentNumber)
    graph = utils.getGraph(config, False)
    numberNodes = len(graph.nodes)
    history_path = os.path.join(OUTPUT_DIR, f'output_{experimentNumber}.pickle')
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
        
    # print(history)
    hopDistances = hopDistance(experimentNumber)
    #print(hopDistances)
    
    history_hop_distance = [[0 for j in range(NUM_ROUNDS+1)] for i in  range(len(hopDistances)) ]

    def avgSingleRound(hopValue, curr_history):
        peerCount = len(curr_history)
        acc = [curr_history[i][acc_type]  for i in range(peerCount)]
        matrix = [0] * peerCount
        # print(peerCount)
        for i in hopValue:
            matrix[i] = 1
        totalClient = sum(matrix)
        
        return round(numpy.dot(matrix, acc)/totalClient,2)
        
    
    

    for i in range(len(hopDistances)):
        for j in range(1,NUM_ROUNDS+1):
            history_hop_distance[i][j] = avgSingleRound(hopDistances[i], history[j])
    if plot:
        plotHopAcc(history_hop_distance, acc_type, experimentNumber)
    return numpy.array(history_hop_distance)
        

def plotHopAcc(history_hop_distance,acc_type,experimentNumber, single =False):
        plt.clf()
        for i in range(len(history_hop_distance)):
            legend_name = f'adversary-honest hop = {i}'
            color = None
            if i == 0:
                if not single:
                    continue
                legend_name = 'adversary'
                color = 'red'
            if i == 1:
                color = 'purple'
            if i == 2:
                color = 'blue'
            if single:
                legend_name = 'honest'
                color = 'blue'

            plt.plot((history_hop_distance[i][1:]), label=legend_name, color=color)
            plt.legend(loc='lower right' )
            plt.xticks(list(range(0,NUM_ROUNDS+1,5)))
            plt.yticks([i/100 for i in list(range(0,101,10))])
            plt.xlabel('Num Rounds')
            if 'backdoor' in acc_type:
                title = 'Attack Success'
            else:
                title = 'Test Acc'
            plt.ylabel(title)
            # plt.title(f'{title} vs Round')
        file_path = os.path.join(f'Results/Experiment-{acc_type}-{experimentNumber}.png')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path,bbox_inches='tight')
    


def singleRun(experimentNumber):


    backdoorAccs= [hopDistanceAcc(experimentNumber,'test_backdoor_acc', False),  hopDistanceAcc(experimentNumber+1,'test_backdoor_acc',False),
    hopDistanceAcc(experimentNumber+2,'test_backdoor_acc',False)]
    testAccs= [hopDistanceAcc(experimentNumber,'test_acc', False),  hopDistanceAcc(experimentNumber+1,'test_acc',False),
    hopDistanceAcc(experimentNumber+2,'test_acc',False)]

    minDimention = min(map(lambda x: x.shape[0], backdoorAccs))
    testAccs = [i[0:minDimention, :] for i in testAccs ]
    backdoorAccs = [i[0:minDimention, :] for i in backdoorAccs ]

  
    backdoorAccsMean = numpy.mean(backdoorAccs, axis =0)
    testAccsMean = numpy.mean(testAccs, axis =0)


    backdoorLast  =  [backdoorAccsMean[0, 70],backdoorAccsMean[1, 70] if len(backdoorAccsMean) >=2 else 0 ,backdoorAccsMean[2, 70] if len(backdoorAccsMean) >=3 else 0 ]
    testLast = [testAccsMean[0, 70],testAccsMean[1, 70] if len(backdoorAccsMean) >=2 else 0,testAccsMean[2, 70] if len(backdoorAccsMean) >=3 else 0]


    return backdoorLast, testLast

def varyAdversarySamplingStrategy(mode='attacksuccess',ratio=False):

    all_index= [15, 24, 27, 30, 33]
    all_data =[hop_general(i) for i in all_index]
    all_data_test =[hop_general(i, 'test_acc') for i in all_index]

    experiment_adversary_count_map = { 15:'random', 24:'maxdegree', 27:'maxens', 30:'maxpagerank', 33:'maxclustering' } 
    strategies = []
    testAcc = []
    backdoorAcc = []

    for exp, strategy in experiment_adversary_count_map.items():
        backdoor,testacc = singleRun(exp)
        
        strategies.append(strategy)
        testAcc.append(testacc)
        backdoorAcc.append(backdoor)
    print(strategies)
    print(testAcc)
    print(backdoorAcc)

    if ratio:

        for i in range(1,len(testAcc)):
            for j in range(len(testAcc[i])):
                base_acc  = testAcc[0][j]
                base_back = backdoorAcc[0][j]
                testAcc[i][j] /= base_acc
                backdoorAcc[i][j] /= base_back
        for j in range(len(testAcc[i])):
            testAcc[0][j] = 1
            backdoorAcc[0][j] = 1
            
        


    N = len(strategies)
    ind = numpy.arange(N)
    width = 0.27 
    plt.clf()

    if mode == 'attacksuccess':
        if ratio:
            file_path = os.path.join(f'Results/AdversarySamplingStrategyvsAttackSuccess_Normalized.png')
        else:
            file_path = os.path.join(f'Results/AdversarySamplingStrategyvsAttackSuccess.png')

    else:
        if ratio:
            file_path = os.path.join(f'Results/AdversarySamplingStrategyvsTestAcc_Normalized.png')
        else:
            file_path = os.path.join(f'Results/AdversarySamplingStrategyvsTestAcc.png')


    for i in range(3):
        legend_name = f'adversary-honest hop = {i}'
        color = None
        if i == 0:
            legend_name = 'all'
            color = 'green'
            X_curr = ind-width
            if mode == 'attacksuccess':
                back =all_data
            else:
                back =all_data_test

        if i == 1:
            color = 'purple'
            X_curr = ind
            if mode == 'attacksuccess':
                back = [item[i] for item in backdoorAcc ]
            else:
                back    = [item[i] for item in testAcc ]
        if i == 2:
            color = 'blue'
            X_curr = ind+width

            if mode == 'attacksuccess':
                back = [item[i] for item in backdoorAcc ]
            else:
                back = [item[i] for item in testAcc ]
        # plt.plot(adversarialNodeCount, back, label=legend_name, color=color)
        bars = plt.bar(X_curr, back,width, label=legend_name, color=color, align='center')

# access the bar attributes to place the text in the appropriate location
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .005, format(yval,'.2f'))
        if ratio or 'test' in mode:
            plt.legend(loc='lower right' )
        else:
            plt.legend(loc='upper left')
        plt.xticks(ind ,strategies )
        if not ratio:
            plt.ylim(0, 1)
        
        # plt.yticks()
        # plt.xticks(list(range(0,NUM_ROUNDS+1,5)))
        # plt.yticks([i/100 for i in list(range(0,101,10))])
        # plt.xlabel('AdversaryCount')
        if mode == 'attacksuccess':
            if not ratio:
                plt.ylabel('Attack Success')
            else:
                plt.ylabel('Attack Success Normalized wrt Random')
        else:
            if not ratio:
                plt.ylabel('Test Acc')
            else:
                plt.ylabel('Test Acc Normalized wrt Random')
        # plt.title(f'{title} vs Round')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path,bbox_inches='tight')
def varyAdversarySamplingStrategyScatter(mode='attacksuccess',ratio=False):
    #
    
    experiment_adversary_count_map = { 15:'random', 24:'maxdegree', 27:'maxens', 30:'maxpagerank', 33:'maxclustering' } 
    strategies = []
    testAcc = []
    backdoorAcc = []

    for exp, strategy in experiment_adversary_count_map.items():
        backdoor,testacc = singleRun(exp)
        
        strategies.append(strategy)
        testAcc.append(testacc)
        backdoorAcc.append(backdoor)
    print(strategies)
    print(testAcc)
    print(backdoorAcc)

    if ratio:

        for i in range(1,len(testAcc)):
            for j in range(len(testAcc[i])):
                base_acc  = testAcc[0][j]
                base_back = backdoorAcc[0][j]
                testAcc[i][j] /= base_acc
                backdoorAcc[i][j] /= base_back
        for j in range(len(testAcc[i])):
            testAcc[0][j] = 1
            backdoorAcc[0][j] = 1
            
        


    N = len(strategies)
    ind = numpy.arange(N)
    width = 0.27 
    plt.clf()
    varyAdvesaryCountPlot(mode)
    
    if mode == 'attacksuccess':
        if ratio:
            file_path = os.path.join(f'Results/AdversarySamplingStrategyvsAttackSuccess_Scatter_Normalized.png')
        else:
            file_path = os.path.join(f'Results/AdversarySamplingStrategyvsAttackSuccess_Scatter.png')

    else:
        if ratio:
            file_path = os.path.join(f'Results/AdversarySamplingStrategyvsTestAcc_Scatter_Normalized.png')
        else:
            file_path = os.path.join(f'Results/AdversarySamplingStrategyvsTestAcc_Scatter.png')

    colors = ['blue', 'purple', 'red', 'black', 'green']
    for i in range(3):
        legend_name = f'adversary-honest hop = {i}'
        if i == 0:
            legend_name = 'adversary'
            continue
        if i == 1:
            marker = 'o'

        if i == 2:
            marker = 'v'


        if mode == 'attacksuccess':
            back = [item[i] for item in backdoorAcc ]
        else:
            back = [item[i] for item in testAcc ]
        # plt.plot(adversarialNodeCount, back, label=legend_name, color=color)
        for index, k in enumerate(back):
            print('-'*50)
            print(colors[index], marker)
            print('-'*50)

            plt.scatter([3], k, label = strategies[index], marker=marker,color=colors[index])
        plt.legend()
        plt.ylim(0.2, 0.5)
# # access the bar attributes to place the text in the appropriate location
#         for bar in bars:
#             yval = bar.get_height()
#             plt.text(bar.get_x(), yval + .005, format(yval,'.2f'))
#         if ratio or 'test' in mode:
#             plt.legend(loc='lower right' )
#         else:
#             plt.legend(loc='upper left')
#         plt.xticks(ind+width/2 ,strategies )
#         if not ratio:
#             plt.ylim(0, 1)
        
        # plt.yticks()
        # plt.xticks(list(range(0,NUM_ROUNDS+1,5)))
        # plt.yticks([i/100 for i in list(range(0,101,10))])
        # plt.xlabel('AdversaryCount')
        if mode == 'attacksuccess':
            if not ratio:
                plt.ylabel('Attack Success')
            else:
                plt.ylabel('Attack Success Normalized wrt Random')
        else:
            if not ratio:
                plt.ylabel('Test Acc')
            else:
                plt.ylabel('Test Acc Normalized wrt Random')
        # plt.title(f'{title} vs Round')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path,bbox_inches='tight')

def varyAdvesaryCountPlot(mode='attacksuccess'):
    experiment_adversary_count_map = { 9:1,12:2, 15:3, 18:4, 21:5, 3:6,}
    adversarialNodeCount = []
    testAcc = []
    backdoorAcc = []

    for exp, count in experiment_adversary_count_map.items():
        backdoor,testacc = singleRun(exp)
        
        adversarialNodeCount.append(count)
        testAcc.append(testacc)
        backdoorAcc.append(backdoor)

    plt.clf()

    for i in range(3):
        legend_name = f'random adversary-honest hop = {i}'
        color = None
        if i == 0:
            legend_name = 'adversary'
            color = 'red'
            continue
        if i == 1:
            color = 'purple'
        if i == 2:
            color = 'blue'

        if mode == 'attacksuccess':
            back = [item[i] for item in backdoorAcc ]
        else:
            back = [item[i] for item in testAcc ]
        plt.plot(adversarialNodeCount, back, label=legend_name, color=color)
        plt.scatter(adversarialNodeCount, back, label='_nolegend_', color=color)
        plt.legend(loc='lower right' )
        plt.ylim(0, 1)


        # plt.xticks(list(range(0,NUM_ROUNDS+1,5)))
        # plt.yticks([i/100 for i in list(range(0,101,10))])
        plt.xlabel('AdversaryCount')
        if mode == 'attacksuccess':
            plt.ylabel('Attack Success')
        else:
            plt.ylabel('Test Acc')
        # plt.title(f'{title} vs Round')
    if mode == 'attacksuccess':
        file_path = os.path.join(f'Results/AdversaryCountvsAttackSuccess.png')
    else:
        file_path = os.path.join(f'Results/AdversaryCountvsTestAcc.png')

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path,bbox_inches='tight')

def varyAdvesaryCountPageRankPlot(mode='attacksuccess'):
    experiment_adversary_count_map = { 42:1,45:2, 30:3, 36:4, 39:5, 48:6,}
    adversarialNodeCount = []
    testAcc = []
    backdoorAcc = []

    for exp, count in experiment_adversary_count_map.items():
        backdoor,testacc = singleRun(exp)
        
        adversarialNodeCount.append(count)
        testAcc.append(testacc)
        backdoorAcc.append(backdoor)

    plt.clf()
    varyAdvesaryCountPlot(mode)
    for i in range(3):
        legend_name = f'page rank adversary-honest hop = {i}'
        color = None
        if i == 0:
            legend_name = 'adversary'
            color = 'red'
            continue
        if i == 1:
            color = 'green'
        if i == 2:
            color = 'black'

        if mode == 'attacksuccess':
            back = [item[i] for item in backdoorAcc ]
        else:
            back = [item[i] for item in testAcc ]
        plt.plot(adversarialNodeCount, back, label=legend_name, color=color)
        plt.scatter(adversarialNodeCount, back, label='_nolegend_', color=color)
        plt.legend(loc='lower right' )
        plt.ylim(0, 1)


        # plt.xticks(list(range(0,NUM_ROUNDS+1,5)))
        # plt.yticks([i/100 for i in list(range(0,101,10))])
        plt.xlabel('AdversaryCount')
        if mode == 'attacksuccess':
            plt.ylabel('Attack Success')
        else:
            plt.ylabel('Test Acc')
        # plt.title(f'{title} vs Round')
    if mode == 'attacksuccess':
        file_path = os.path.join(f'Results/PageRankAdversaryCountvsAttackSuccess.png')
    else:
        file_path = os.path.join(f'Results/PageRankAdversaryCountvsTestAcc.png')

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path,bbox_inches='tight')

def varyTotalNodeCountPlot(mode='attacksuccess'):
    experiment_adversary_count_map = {15:60, 60:80, 54:100}
    adversarialNodeCount = []
    testAcc = []
    backdoorAcc = []

    for exp, count in experiment_adversary_count_map.items():
        backdoor,testacc = singleRun(exp)
        
        adversarialNodeCount.append(count)
        testAcc.append(testacc)
        backdoorAcc.append(backdoor)

    plt.clf()

    for i in range(3):
        legend_name = f'random adversary-honest hop = {i}'
        color = None
        if i == 0:
            legend_name = 'adversary'
            color = 'red'
            continue
        if i == 1:
            color = 'purple'
        if i == 2:
            color = 'blue'

        if mode == 'attacksuccess':
            back = [item[i] for item in backdoorAcc ]
        else:
            back = [item[i] for item in testAcc ]
        plt.plot(adversarialNodeCount, back, label=legend_name, color=color)
        plt.scatter(adversarialNodeCount, back, label='_nolegend_', color=color)
        plt.legend(loc='lower right' )
        plt.ylim(0, 1)


        # plt.xticks(list(range(0,NUM_ROUNDS+1,5)))
        # plt.yticks([i/100 for i in list(range(0,101,10))])
        plt.xlabel('NodeCount')
        if mode == 'attacksuccess':
            plt.ylabel('Attack Success')
        else:
            plt.ylabel('Test Acc')
        # plt.title(f'{title} vs Round')
    if mode == 'attacksuccess':
        file_path = os.path.join(f'Results/NodeCountvsAttackSuccess.png')
    else:
        file_path = os.path.join(f'Results/NodeCountvsTestAcc.png')

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path,bbox_inches='tight')

def varyTotalNodeCountPageRankPlot(mode='attacksuccess'):
    experiment_adversary_count_map = { 30:60, 57:80, 51:100}
    adversarialNodeCount = []
    testAcc = []
    backdoorAcc = []

    for exp, count in experiment_adversary_count_map.items():
        backdoor,testacc = singleRun(exp)
        
        adversarialNodeCount.append(count)
        testAcc.append(testacc)
        backdoorAcc.append(backdoor)

    plt.clf()
    varyTotalNodeCountPlot(mode)
    for i in range(3):
        legend_name = f'page rank adversary-honest hop = {i}'
        color = None
        if i == 0:
            legend_name = 'adversary'
            color = 'red'
            continue
        if i == 1:
            color = 'green'
        if i == 2:
            color = 'black'

        if mode == 'attacksuccess':
            back = [item[i] for item in backdoorAcc ]
        else:
            back = [item[i] for item in testAcc ]
        plt.plot(adversarialNodeCount, back, label=legend_name, color=color)
        plt.scatter(adversarialNodeCount, back, label='_nolegend_', color=color)
        plt.legend(loc='lower right' )
        plt.ylim(0, 1)


        # plt.xticks(list(range(0,NUM_ROUNDS+1,5)))
        # plt.yticks([i/100 for i in list(range(0,101,10))])
        plt.xlabel('NodeCount')
        if mode == 'attacksuccess':
            plt.ylabel('Attack Success')
        else:
            plt.ylabel('Test Acc')
        # plt.title(f'{title} vs Round')
    if mode == 'attacksuccess':
        file_path = os.path.join(f'Results/PageRankNodeCountvsAttackSuccess.png')
    else:
        file_path = os.path.join(f'Results/PageRankNodeCountvsTestAcc.png')

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path,bbox_inches='tight')

def Average3HopGraph(experimentNumber, name, single=False):

    # experimentNumber = 9

    backdoorAccs= [hopDistanceAcc(experimentNumber,'test_backdoor_acc', False),  hopDistanceAcc(experimentNumber+1,'test_backdoor_acc',False),
    hopDistanceAcc(experimentNumber+2,'test_backdoor_acc',False)]
    testAccs= [hopDistanceAcc(experimentNumber,'test_acc', False),  hopDistanceAcc(experimentNumber+1,'test_acc',False),
    hopDistanceAcc(experimentNumber+2,'test_acc',False)]

    
    
    minDimention = min(map(lambda x: x.shape[0], backdoorAccs))
    testAccs = [i[0:minDimention, :] for i in testAccs ]
    backdoorAccs = [i[0:minDimention, :] for i in backdoorAccs ]

  
    backdoorAccsMean = numpy.mean(backdoorAccs, axis =0)
    testAccsMean = numpy.mean(testAccs, axis =0)

    print(backdoorAccsMean[0, 70],testAccsMean[0, 70])
    print(backdoorAccsMean[1, 70],testAccsMean[1, 70])
    print(backdoorAccsMean[2, 70],testAccsMean[2, 70])
    print(backdoorAccsMean[2, 70],testAccsMean[2, 70])
    print(backdoorAccsMean.shape)
    

    plotHopAcc(backdoorAccsMean,'test_backdoor_acc', name, single )
    plotHopAcc(testAccsMean,'test_acc', name, single )

def cleanExperiment():

    experimentNumber = 0
    backdoorAccs= [hopDistanceAcc(experimentNumber,'test_backdoor_acc', False),  hopDistanceAcc(experimentNumber+1,'test_backdoor_acc',False),
    hopDistanceAcc(experimentNumber+2,'test_backdoor_acc',False)]
    testAccs= [hopDistanceAcc(experimentNumber,'test_acc', False),  hopDistanceAcc(experimentNumber+1,'test_acc',False),
    hopDistanceAcc(experimentNumber+2,'test_acc',False)]

    backdoorAccs = numpy.array(backdoorAccs)
    backdoorAccsMean = numpy.mean(backdoorAccs, axis =0)
    testAccsMean = numpy.mean(testAccs, axis =0)
    print(backdoorAccsMean[0, 70],testAccsMean[0, 70] )
    plotHopAcc(backdoorAccsMean,'test_backdoor_acc', 'clean',True )
    plotHopAcc(testAccsMean,'test_acc', 'clean',True )
    print('-'*50)

def varyGraph(mode='attacksuccess', random_select=False):
    all_index= [63, 30, 66]
    all_data =[hop_general(i) for i in all_index]
    all_data_test =[hop_general(i, 'test_acc') for i in all_index]
    all_index_random= [72, 15, 75]
    all_data_random = [hop_general(i) for i in all_index_random]
    all_data_test_random =[hop_general(i, 'test_acc') for i in all_index_random]


    experiment_adversary_count_map        = {  63:'Erdos', 30:'Watts', 66:'Barabasi', }# 78:'Complete' }  # pagerank for complete does not make sense all equal
    experiment_adversary_count_map_random = {  72:'Erdos', 15:'Watts',  75:'Barabasi',}# 78:'Complete' }
    if random_select:
        experiment_adversary_count_map = experiment_adversary_count_map_random
        all_data = all_data_random
        all_data_test = all_data_test_random

    strategies = []
    testAcc = []
    backdoorAcc = []

    for exp, strategy in experiment_adversary_count_map.items():
        backdoor,testacc = singleRun(exp)
        
        strategies.append(strategy)
        testAcc.append(testacc)
        backdoorAcc.append(backdoor)
    print(strategies)
    print(testAcc)
    print(backdoorAcc)


    N = len(strategies)
    ind = numpy.arange(N)
    width = 0.27 
    plt.clf()

    if mode == 'attacksuccess':

        if random_select:
                file_path = os.path.join(f'Results/GraphvsAttackSuccess_random.png')
        else:
            file_path = os.path.join(f'Results/GraphvsAttackSuccess_pagerank.png')


    else:
        if random_select:
            file_path = os.path.join(f'Results/GraphvsTestAcc_random.png')
        else:
            file_path = os.path.join(f'Results/GraphvsTestAcc_pagerank.png')


    for i in range(3):
        legend_name = f'adversary-honest hop = {i}'
        color = None
        if i == 0:
            legend_name = 'all'
            color = 'green'
            back = all_data
            X_curr = ind - width
            if mode == 'attacksuccess':
                back = all_data
            else:
                back = all_data_test
        if i == 1:
            color = 'purple'
            X_curr = ind
            if mode == 'attacksuccess':
                back = [item[i] for item in backdoorAcc ]
            else:
                back = [item[i] for item in testAcc ]
        if i == 2:
            color = 'blue'
            X_curr = ind+width
            if mode == 'attacksuccess':
                back = [item[i] for item in backdoorAcc ]
            else:
                back = [item[i] for item in testAcc ]

   
        # plt.plot(adversarialNodeCount, back, label=legend_name, color=color)
        bars = plt.bar(X_curr, back,width, label=legend_name, color=color, align='center')

# access the bar attributes to place the text in the appropriate location
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .005, format(yval,'.2f'))
        if  'test' in mode:
            plt.legend(loc='lower right' )
        else:
            plt.legend(loc='upper right')
        plt.xticks(ind ,strategies )
       
        plt.ylim(0, 1)
        
        # plt.yticks()
        # plt.xticks(list(range(0,NUM_ROUNDS+1,5)))
        # plt.yticks([i/100 for i in list(range(0,101,10))])
        # plt.xlabel('AdversaryCount')
        if mode == 'attacksuccess':
            plt.ylabel('Attack Success')
        else:
            plt.ylabel('Test Acc')
        # plt.title(f'{title} vs Round')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path,bbox_inches='tight')

def varyDataset(mode='attacksuccess', random_select=False):
    all_index= [30, 81]
    all_data =[hop_general(i) for i in all_index]
    all_data_test =[hop_general(i, 'test_acc') for i in all_index]
    all_index_random= [15, 84]
    all_data_random = [hop_general(i) for i in all_index_random]
    all_data_test_random =[hop_general(i, 'test_acc') for i in all_index_random]


    experiment_adversary_count_map        = {  30:'Emnist', 81:'FashionMnist' }  # pagerank for complete does not make sense all equal
    experiment_adversary_count_map_random = {  15:'Emnist', 84:'FashionMnist'}
    if random_select:
        experiment_adversary_count_map = experiment_adversary_count_map_random
        all_data = all_data_random
        all_data_test = all_data_test_random
    strategies = []
    testAcc = []
    backdoorAcc = []

    for exp, strategy in experiment_adversary_count_map.items():
        backdoor,testacc = singleRun(exp)
        
        strategies.append(strategy)
        testAcc.append(testacc)
        backdoorAcc.append(backdoor)
    print(strategies)
    print(testAcc)
    print(backdoorAcc)


    N = len(strategies)
    ind = numpy.arange(N)
    width = 0.27 
    plt.clf()

    if mode == 'attacksuccess':

            if random_select:
                    file_path = os.path.join(f'Results/DatasetvsAttackSuccess_random.png')
            else:
                file_path = os.path.join(f'Results/DatasetvsAttackSuccess_pagerank.png')


    else:
        if random_select:
            file_path = os.path.join(f'Results/DatasetvsTestAcc_random.png')
        else:
            file_path = os.path.join(f'Results/DatasetvsTestAcc_pagerank.png')


    for i in range(3):
        legend_name = f'adversary-honest hop = {i}'
        color = None
        if i == 0:
            legend_name = 'all'
            color = 'green'
            X_curr = ind - width
            if mode == 'attacksuccess':
                back = all_data
            else:
                back = all_data_test
        if i == 1:
            color = 'purple'
            X_curr = ind
            if mode == 'attacksuccess':
                back = [item[i] for item in backdoorAcc ]
            else:
                back = [item[i] for item in testAcc ]
        if i == 2:
            color = 'blue'
            X_curr = ind+width

            if mode == 'attacksuccess':
                back = [item[i] for item in backdoorAcc ]
            else:
                back = [item[i] for item in testAcc ]
        # plt.plot(adversarialNodeCount, back, label=legend_name, color=color)
        bars = plt.bar(X_curr, back,width, label=legend_name, color=color, align='center')

# access the bar attributes to place the text in the appropriate location
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .005, format(yval,'.2f'))
        if  'test' in mode:
            plt.legend(loc='lower right' )
        else:
            plt.legend(loc='upper left')
        plt.xticks(ind-width ,strategies )
       
        plt.ylim(0, 1)
        
        # plt.yticks()
        # plt.xticks(list(range(0,NUM_ROUNDS+1,5)))
        # plt.yticks([i/100 for i in list(range(0,101,10))])
        # plt.xlabel('AdversaryCount')
        if mode == 'attacksuccess':
            plt.ylabel('Attack Success')
        else:
            plt.ylabel('Test Acc')
        # plt.title(f'{title} vs Round')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path,bbox_inches='tight')


def baseline():



    def get_round_avg(peers, field):
        peer_count = 0
        total = 0
        
        for peer, score in peers.items():
            total += score[field]
            peer_count += 1
        return total / peer_count



    def single_seed(experimentNumber):
        # config = utils.readConfig(experimentNumber)
        # graph = utils.getGraph(config, False)
        # peerCount = len(graph.nodes)
        path = f'/net/data/p2pml/p2pml/Output/output_{experimentNumber}.pickle'
        with open(path, 'rb') as file:
            data = pickle.load(file)

        # data = collectEvaluationFiles(peerCount, NUM_ROUNDS, experimentNumber)
        result = [0] * NUM_ROUNDS
        for round, peers in data.items():
            result[round-1] = get_round_avg(peers, 'test_acc')
        return numpy.array(result)

    def average_experiments(experiments):
        result = numpy.zeros(NUM_ROUNDS)
        for experiment in experiments:
            curr = single_seed(experiment)
            # print(curr)
            result = numpy.add(result, curr )
        return result/len(experiments)

    def take_average_last_5(experiments):
        result = average_experiments(experiments)
        average = numpy.mean(result[-5:])
        return average

    # result = take_average_last_5([87, 88, 89])

    def node_count_acc_watts():

        #a = get_matrix_average([87, 88, 89])
        #print(a.shape)
        # b=get_matrix_average([102,103, 104])
        # print(b.shape)
        # raise 'a'
        data = {
            '60' :  np.mean(get_matrix_average([87, 88, 89]), axis=0),
            '80' :  np.mean(get_matrix_average([102,103, 104]), axis=0),
            '100' : np.mean(get_matrix_average([105, 106, 107]), axis=0),
        }
        return data

    def clipping_node_count_watts():

        #a = get_matrix_average([87, 88, 89])
        #print(a.shape)
        # b=get_matrix_average([102,103, 104])
        # print(b.shape)
        # raise 'a'
        data = {
            '60' :  np.mean(get_matrix_average([87, 88, 89]), axis=0),
            '60-clipped':np.mean(get_matrix_average([126, 127, 128]), axis=0),
          #  '80' :  np.mean(get_matrix_average([102,103, 104]), axis=0),
          #  '100' : np.mean(get_matrix_average([105, 106, 107]), axis=0),
        }
        return data

    def fault_tolerant_baseline():
        data = {
            '0' :  np.mean(get_matrix_average([87, 88, 89]), axis=0),
            '2' :  np.mean(get_matrix_average([117,118, 119]), axis=0),
            '4' : np.mean(get_matrix_average([120, 121, 122]), axis=0),
            '6' : np.mean(get_matrix_average([123, 124, 125]), axis=0),
        }
        return data

    def get_matrix(experimentNumber):
        path = f'/net/data/p2pml/p2pml/Output/output_{experimentNumber}.pickle'
        with open(path, 'rb') as file:
            data = pickle.load(file)
            round_count = len(data.keys())
            num_peers = len(data[1].keys())
            print(num_peers)



        matrix = numpy.zeros(( num_peers,round_count))
        for round, peers in data.items():
            for peer, val in peers.items():
                matrix[peer][round-1] = val['test_acc']
        print(matrix)
        return matrix

    def get_matrix_average(experiments):
        matrix = numpy.array([get_matrix(experiment) for experiment in experiments])
        matrix= matrix.mean(axis=0)
        return matrix

    def learning_speed():

        data = {
        'erdos' : np.mean(get_matrix_average([90,91, 92]), axis=0), 
        'watts' :    np.mean(get_matrix_average([87, 88, 89]), axis=0),
        'barabasi' : np.mean(get_matrix_average([93,94,95]), axis=0),
        'complete':  np.mean(get_matrix_average([96,97,98]), axis=0),
         }
         

        df = pd.DataFrame(data)
        print(df)
        sns.lineplot(data=df)
        plt.xlabel('Round')
        plt.ylabel('Test Accuracy')

        file_path = 'Results/speed_baseline.png'
        plt.savefig(file_path,bbox_inches='tight')


    def cdf():
        #matrix = get_matrix_average([96, 97, 98])
        #matrix = get_matrix_average([96, 97, 98])


        data = {
        'erdos' : get_matrix_average([90,91, 92])[:, -1],
        'watts' : get_matrix_average([87, 88, 89])[:, -1],
        'barabasi' : get_matrix_average([93,94,95])[:, -1],
        'complete':  get_matrix_average([96,97,98])[:, -1],
         }

                # make dataframe
        df = pd.DataFrame(data)
        print(df)

        # plot melted dataframe in a single command
        sns.histplot(df.melt(), x='value', hue='variable',
             multiple='dodge', shrink=1, bins=10)




        plt.xlabel('Test Acc')
        plt.ylabel('Count')
        file_path = 'Results/cdf_baseline.png'
        plt.savefig(file_path,bbox_inches='tight')
    






        







    def graph_type():
        data = {
        'erdos' :    np.mean(get_matrix_average([90,91, 92]), axis=0),
        'watts' :    np.mean(get_matrix_average([87, 88, 89]), axis=0),
        'barabasi' : np.mean(get_matrix_average([93,94,95]), axis=0),
        # 'complete':  np.mean(get_matrix_average([96,97,98]), axis=0),
         }
        return data

    def plot_figure(data, pathname, x_label):
        N = len(data.keys())
        ind = numpy.arange(N)
        width_3chars = 0.15 
        file_path = os.path.join(pathname ) 
        plt.clf()
        

        df = pd.DataFrame(data)
        print(df)
        sns.lineplot(data=df)

        plt.savefig(file_path,bbox_inches='tight')
        # plt.xticks(ind ,data.keys() )
        plt.ylim(0, 1)
        plt.ylabel('Test Acc')
        plt.xlabel('Round')


        # plt.legend(loc='lower right' )

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path,bbox_inches='tight', dpi=1200)



    def sub_main():
        plot_figure(graph_type(), 'Results_Combined/Baseline_graphtype.png', 'Graph Type')
        plot_figure(node_count_acc_watts(), 'Results_Combined/Baseline_node_count.png', 'Node Count')
        plot_figure(fault_tolerant_baseline(), 'Results_Combined/Baseline_tolerant.png', 'Tolerant')
        # plot_figure(clipping_node_count_watts(), 'Results_Combined/Baseline_clipping_no_attack.png', 'Node Count')
        # 'complete':  np.mean(get_matrix_average([96,97,98]), axis=0),
         
        # print('node-count:', node_count_acc_watts())
        # print('fault_tolerant:', fault_tolerant_baseline())
        # print('graph_type:', graph_type())
        

    sub_main()
    # cdf()
    #learning_speed()

def hop_general(experimentNumber, acc_type='test_backdoor_acc'):
    def single(experimentNumber):
        hop_acc=hopDistanceAcc(experimentNumber,acc_type, False)
        hop_acc=hop_acc[1:, NUM_ROUNDS]
        hop_distances = hopDistance(experimentNumber)
        element_counts =np.array([len(hop_distances[i]) for i in range(1, len(hop_distances))])
        #print(hop_acc)
        #print(element_counts)
        total = np.dot(hop_acc, element_counts)
        mean = total/np.sum(element_counts)
        #  print(mean)
        return mean
    
    average =  sum([single(experimentNumber=experimentNumber),
    single(experimentNumber=experimentNumber+1),
    single(experimentNumber=experimentNumber+2)]) / 3
    
    print(average)
    return average

def fault_tolerant_attack():
    def get_matrix(experimentNumber):
        path = f'/net/data/p2pml/p2pml/Output/output_{experimentNumber}.pickle'
        with open(path, 'rb') as file:
            data = pickle.load(file)
            round_count = len(data.keys())
            num_peers = len(data[1].keys())
            print(num_peers)



        matrix = numpy.zeros(( num_peers,round_count))
        for round, peers in data.items():
            for peer, val in peers.items():
                matrix[peer][round-1] = val['test_backdoor_acc']
        print(matrix)
        return matrix

    def get_matrix_average(experiments):
        matrix = numpy.array([get_matrix(experiment) for experiment in experiments])
        matrix= matrix.mean(axis=0)
        return matrix



    def get_round_avg(peers, field):
        peer_count = 0
        total = 0
        
        for peer, score in peers.items():
            # print(score.keys())
            total += score[field]
            peer_count += 1
        return total / peer_count



    def single_seed(experimentNumber):
        # config = utils.readConfig(experimentNumber)
        # graph = utils.getGraph(config, False)
        # peerCount = len(graph.nodes)
        path = f'/net/data/p2pml/p2pml/Output/output_{experimentNumber}.pickle'
        with open(path, 'rb') as file:
            data = pickle.load(file)

        # data = collectEvaluationFiles(peerCount, NUM_ROUNDS, experimentNumber)
        result = [0] * NUM_ROUNDS
        for round, peers in data.items():
            result[round-1] = get_round_avg(peers, 'test_backdoor_acc')
        return numpy.array(result)

    def average_experiments(experiments):
        result = numpy.zeros(NUM_ROUNDS)
        for experiment in experiments:
            curr = single_seed(experiment)
            # print(curr)
            result = numpy.add(result, curr )
        return result/len(experiments)

    def take_average_last_5(experiments):
        result = average_experiments(experiments)
        average = numpy.mean(result[-5:])
        return average



    def plot_figure(data, pathname, x_label, y_label):
        N = len(data.keys())
        ind = numpy.arange(N)
        width_3chars = 0.15 
        file_path = os.path.join(pathname ) 
        plt.clf()
        
        df = pd.DataFrame(data)
        print(df)
        sns.lineplot(data=df)

        plt.savefig(file_path,bbox_inches='tight')
        # plt.xticks(ind ,data.keys() )
        plt.ylim(0, 1)
        plt.ylabel(y_label)
        plt.xlabel('Round')


        # plt.legend(loc='lower right' )

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path,bbox_inches='tight', dpi=1200)




        # plt.legend(loc='lower right' )

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path,bbox_inches='tight')





    def hop_plot(mode='attacksuccess'):
        #hop = hopDistance(54)
        #print(get_matrix_average([54]))
        #print(hop)
        all_index= [54, 108, 111, 114]
        all_data =[hop_general(i) for i in all_index]
        all_data_test =[hop_general(i, 'test_acc') for i in all_index]
        if mode!= 'attacksuccess':
            all_data= all_data_test
        data = {
        0 :singleRun(54),
        2 : singleRun(108),
        4 : singleRun(111),
        6 : singleRun(114),
        }
        
        if mode == 'attacksuccess':
            file_path = os.path.join(f'Results/Fault_tolerant_attack.png')
            for key in data:
                data[key] = numpy.array(data[key][0])
        else:
            file_path = os.path.join(f'Results/Fault_tolerant_test.png')
            for key in data:
                data[key] = numpy.array(data[key][1])

        df = pd.DataFrame(data)
        df = df.T
        
        df = df.drop(columns=[0])

        plt.clf()
        N = len(data)
        ind = numpy.arange(N)
        width = 0.27 

        for i in range(3):
            legend_name = f'adversary-honest hop = {i}'
            color = None

            if i == 1:
                color = 'purple'
                X_curr = ind
                back = df[i].to_list()

            if i == 2:
                color = 'blue'
                X_curr = ind+width
                back = df[i].to_list()


            
            if i == 0:
                X_curr = ind-width

                legend_name = 'all'
                color = 'green'
                back = all_data

            # plt.plot(adversarialNodeCount, back, label=legend_name, color=color)
            bars = plt.bar(X_curr, back,width, label=legend_name, color=color, align='center')
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x(), yval + .005, format(yval,'.2f'))
        plt.xticks(ind,data.keys() )
        plt.xlabel('r robust')
        plt.ylim(0, 1)
        
        if mode == 'attacksuccess':
            plt.ylabel('Attack Success')
            plt.legend(loc='upper left' )

        else:
            plt.ylabel('Test Acc')
            plt.legend(loc='lower right' )


        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path,bbox_inches='tight')




    # def submain():
    #     data = {
    #     '0' : np.mean(get_matrix_average([54,55, 56]),axis=0),
    #     '2' : np.mean(get_matrix_average([108,109, 110]),axis=0),
    #     '4' : np.mean(get_matrix_average([111, 112, 113]),axis=0),
    #     '6' : np.mean(get_matrix_average([114,115,116]),axis=0),
    #     }
    #     plot_figure(data,'Results/Fault_tolerant_attack.png', 'Tolerant', 'Attack Success' )
    
    # submain()
    #hop_general(54)
    hop_plot()
    hop_plot('')

def main():

    # cleanExperiment()
    # Average3HopGraph(3, 'strong_attack')
    # Average3HopGraph(6, 'strongest_attack')
    # Average3HopGraph(9, 'single_adversary')
    # Average3HopGraph(12, 'two_adversary')
    # Average3HopGraph(15, 'three_adversary')
    # Average3HopGraph(18, 'four_adversary')
    # varyAdvesaryCountPlot('testaccs')
    #  Average3HopGraph(24, 'highestdegree')
    #  Average3HopGraph(30, 'highestpagerank')
    #  Average3HopGraph(33, 'maxclustering')

    # varyAdversarySamplingStrategy()
    # varyAdversarySamplingStrategy('testaccs')
    # varyAdversarySamplingStrategy(ratio=True)
    # varyAdversarySamplingStrategy('testaccs',ratio=True,)

    # varyAdversarySamplingStrategyScatter()
    # varyAdvesaryCountPlot()
    # varyAdvesaryCountPageRankPlot()
    # varyAdvesaryCountPageRankPlot('testaccs')
    # Average3HopGraph(51, '100_nodes_pagerank')
    # Average3HopGraph(54, '100_nodes_random')
    # varyTotalNodeCountPageRankPlot()
    # varyTotalNodeCountPageRankPlot('testaccs')

    # varyGraph()
    # varyGraph('testaccs')
    # varyGraph(random_select=True)
    # varyGraph('testaccs', random_select=True)

    # varyDataset()
    # varyDataset('testaccs')
    # varyDataset(random_select=True)
    # varyDataset('testaccs', random_select=True)

    baseline()
    # fault_tolerant_attack()
if __name__ == '__main__':
    main()
