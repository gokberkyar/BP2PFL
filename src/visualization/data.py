import pickle
# from p2pml.commons import OUTPUT_DIR, PROJECT_DIR, DATA_PATH,NUM_ROUNDS
import p2pml.utils as utils
import networkx as nx
from collections import defaultdict, deque
import os
from p2pml.commons import NUM_ROUNDS, OUTPUT_DIR
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# sns.set(font_scale=1)

#OUTPUT_DIR = '/net/data/p2pml/p2pml/Output'
OUTPUT_DIR = '/net/data/p2pml/p2pml/results_jun13_n60k12/Output'
#OUTPUT_DIR = '/net/data/p2pml/p2pml/results_jun15_n30k12/Output'
isAttack = True

class Data:

    def _get_hopdistace(self, experiment_num):
        if experiment_num == 1561:
            experiment_num = 156
        config = utils.readConfig(experiment_num)
        graph = utils.getGraph(config, False)
        numberNodes = len(graph.nodes)

        adversarialNodes = utils.getAdversarialNodes(config,graph,numberNodes, True)
        if len(adversarialNodes) == 0: isAttack = False

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
        
        # result = defaultdict(list)
        # for node, hop in enumerate(hopDistance):
        #     result[hop].append(node)
        return hopDistance

    def _getmatrix(self, experiment_num):
        hop_distance = np.array(self._get_hopdistace(experiment_num), dtype=float)
        # print(hop_distance)
        
        path = os.path.join(OUTPUT_DIR, f'output_{experiment_num}.pickle')
        print("Getting the eval data from ", path)

        with open(path, 'rb') as file:
            data = pickle.load(file)
            round_count = len(data.keys())
            num_peers = len(data[1].keys())

        matrix = np.zeros(( num_peers,round_count, 3))
        for _round, peers in data.items():
            for peer, val in peers.items():
                matrix[peer][_round-1][0] = val['test_backdoor_acc']
                matrix[peer][_round-1][1] = val['test_acc']
                matrix[peer][_round-1][2] = hop_distance[peer]
        print('-'*50)
        print(f'matrix shape, {matrix.shape}')
        print('-'*50)
        return matrix, round_count

    def plot_rounds(self,y,ylabel,path, style="Distance", changedLabels=None, moveLegend=False):
        # y= test_acc
        # y= attack_succ

        #print(self.df_all)
        #linestyles = ['dashed', 'dotted', 'dashdot', 'dashdotted', 'dashdotdotted', 'solid']
        #linestyles = ['--', ':', '-.', (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)), '-']
        #linestyles = [(0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5))]
        
        plt.clf()
        # ax =sns.lineplot(x='round',y=y,data=self.df_all,hue='identifier',errorbar=None, style=style , markers= False)
        
        # ax =sns.lineplot(x='round', y=y, data=self.df_all, hue='identifier', style='identifier', errorbar=None,  markers=False, dashes=True, lw=3, dashes=linestyles, palette=['C0', 'C1', 'C2', 'C3', 'C4', 'k'])
        ax =sns.lineplot(x='round', y=y, data=self.df_all, hue='identifier', style='identifier', errorbar=None,  markers=False, lw=3, dashes=True, palette=['C1', 'C0', 'C2', 'C3', 'C4', 'k'])
        #ax.lines[0].set_linestyle("-.")
        #ax.lines[5].set_linestyle("-")

        handles, labels = ax.get_legend_handles_labels()
        [ha.set_linewidth(3) for ha in handles ]
        if changedLabels:
            labels = changedLabels
        plt.legend(handles, labels, fontsize=16)

        #plt.legend(title = None, fontsize=18)
        #if hue_str in ['partial_view', 'defense']:

        if  moveLegend:
            #place legend outside center right border of plot
            sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)

        # for legend text
        plt.setp(ax.get_legend().get_texts(), fontsize='16')

        plt.xlabel('Rounds', fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path,bbox_inches='tight')


    def plot_max_round(self, path, hue_str):
        if hue_str == "defense":
            strategies = ['No Defense', 'Clipping 1', 'Clipping 0.5', 'Clipping 0.25', 'Trimmed Mean', 'Our Defense']

        max_round = max(self.df_all['round'])
        df_tmp = self.df_all[(self.df_all['round'] == max_round) & (self.df_all['Distance'] == 'All')]
        print(df_tmp)

        ax =sns.barplot(x='round', y='attack_succ', data = df_tmp, hue='identifier', errorbar=None)
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, fontsize=16)

        if hue_str in ['partial_view', 'defense']:
            #place legend outside center right border of plot
            sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
            # for legend text
            plt.setp(ax.get_legend().get_texts(), fontsize='16')

        plt.xlabel('Training Round 200', fontsize=18)
        plt.ylabel('Attack Success', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path,bbox_inches='tight')


    def plot_by_accuracy(self, path, hue_str, acc_list):
        plt.clf()

        if hue_str == 'strategies':
            strategies = ['Random', 'Degree', 'ENS', 'PageRank', 'Clustering']
        elif hue_str == 'graph_type':
            strategies = ['Erdos', 'Watts', 'Barabasi']
        elif hue_str == 'partial_view':
            strategies = ['Global IID', 'Partial IID', 'Global non-IID', 'Partial non-IID']
        elif hue_str == 'iid_vs_noniid':
            strategies = ['IID', 'non-IID']
        elif hue_str == 'iid_vs_noniid_alpha':
            strategies = ['IID', 'non-IID alpha=10', 'non-IID alpha=1', 'non-IID alpha=0.1']
        elif hue_str ==  'dataset':
            strategies = ['EMNIST', 'Fashion MNIST']
        elif hue_str == "adversary_count":
            strategies = ["1", "2", "3", "4", "5", "6"]
        elif hue_str == "node_count":
            strategies = ["60", "80", "100"]
        elif hue_str == "fault_tolerant":
            strategies = ['0', '2', '4', '6']
        elif hue_str == "defense":
            strategies = ['No Defense', 'Trimmed Mean', 'Clipping 1', 'Clipping 0.5', 'Clipping 0.25', 'Our Defense']
        elif hue_str == "defense_noniid":
            strategies = ["IID", "non-IID", "IID Defense", "non-IID Defense"]
        elif hue_str == "datasets_v2":
            strategies = ["MNIST", "EMNIST", "Fashion"]
        elif hue_str == "boosting":
            strategies = ["MNIST boosting=10", "MNIST boosting=5", "MNIST boosting=1"]

        rows = []

        for iden in strategies:
            df_tmp = self.df_all[(self.df_all['identifier'] == iden) & (self.df_all['Distance'] == 'All')]
            print("Max acc: ", iden, df_tmp['test_acc'].max())
            print("Max attack: ", iden, df_tmp['attack_succ'].max())


        for acc in acc_list:
            for iden in strategies:
                #df_tmp = self.df_all[(self.df_all['test_acc'] >= acc) & (self.df_all['identifier'] == iden) & (self.df_all['Distance'] == 'All') & (self.df_all['round'] >= 100)]
                df_tmp = self.df_all[(self.df_all['test_acc'] >= acc) & (self.df_all['identifier'] == iden) & (self.df_all['Distance'] == 'All')]
                if acc == max(acc_list): 
                    df_tmp = df_tmp[(round(df_tmp['test_acc'],2) >= acc)]
                    #df_tmp = df_tmp[(round(df_tmp['test_acc'],2) == acc)]
                else:
                    df_tmp = df_tmp[(round(df_tmp['test_acc'],2) == acc)]
                print(df_tmp)
                for _, row in df_tmp.iterrows():
                    if hue_str in ["adversary_count", "node_count", "fault_tolerant"]:
                        rows.append({'identifier':iden, 'test_acc':str(acc), 'attack_succ':row['attack_succ']})
                    else:
                        rows.append({'identifier':iden, 'test_acc':acc, 'attack_succ':row['attack_succ']})

        df_tmp = pd.DataFrame(rows)
        print(df_tmp)
       
        if hue_str in ["adversary_count", "node_count", "fault_tolerant"]:
            ax =sns.lineplot(x='identifier', y='attack_succ', data = df_tmp, hue='test_acc', style='test_acc', lw=3, markersize=8, errorbar=None, markers=True) 
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles, labels, fontsize=16, title="Accuracy")
            plt.setp(ax.get_legend().get_texts(), fontsize='16') 
            plt.setp(ax.get_legend().get_title(), fontsize='16')
            if hue_str == "adversary_count":
                plt.xlabel('Compromised Nodes', fontsize=18)
            elif hue_str == "node_count":
                plt.xlabel('Total Nodes', fontsize=18)
            elif hue_str == "fault_tolerant":
                plt.xlabel('Failed Neighboring Links', fontsize=18)
        else:
            ax =sns.barplot(x='test_acc', y='attack_succ', data = df_tmp, hue='identifier', errorbar=None) 
            handles, labels = ax.get_legend_handles_labels()

            if hue_str == 'iid_vs_noniid_alpha':
                labels = ["IID", r"non-IID $\alpha=10$", r"non-IID $\alpha=1$", r"non-IID $\alpha=0.1$"]
            plt.legend(handles, labels, fontsize=16)
            plt.xlabel('Test Accuracy', fontsize=18)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f')

        if hue_str in ['partial_view', 'defense', 'iid_vs_noniid_alpha']:
            #place legend outside center right border of plot
            sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
            # for legend text
            plt.setp(ax.get_legend().get_texts(), fontsize='16')  
        
        plt.ylabel('Attack Success', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path,bbox_inches='tight')


    def __init__(self, identifiers_map, xlabel, path, mode, val_vis = True):
        self.xlabel = xlabel
        self.path   = path
        self.mode =mode
        print(identifiers_map)
        
        rows = []
        total_rounds = dict()
        for identifier, seeds in identifiers_map.items():
            for seed in seeds:
                pathseed = os.path.join(OUTPUT_DIR, f'output_{seed}.pickle')
                if not os.path.isfile(pathseed):
                    print("File does not exist: ", pathseed)
                    continue

                print(seed)
                matrix, total_rounds[seed] = self._getmatrix(seed)
                for node, rounds in enumerate(matrix):
                    # print(f'node:{node}, rounds.shape{rounds.shape}')
                    for _round, vals in enumerate(rounds):
                        rows.append({'identifier':identifier, 'seed':seed, 'node':node, 'round':_round,  'test_acc': vals[1], 'attack_succ':vals[0], 'hop_distance':vals[2]})

        df_all =  pd.DataFrame(rows)
        print("df_all: ", df_all)
        print("total_rounds: ", total_rounds)
        

        if isAttack:
            df_all =  df_all[ df_all['hop_distance']!=0 ]


        df1 = df_all[(df_all['hop_distance']==1)]
        df2 = df_all[(df_all['hop_distance']==2)]
        df_all['Distance']= 'All'
        df1['Distance']= 'Hop-1'
        df2['Distance']= 'Hop-2'

        df_all=pd.concat([df_all, df1,df2])

        # just taking first 200 rounds
        df_all = df_all[(df_all['round'] <= 200)]

        self.df_all = df_all

        if mode == None:
            return
        
        df = pd.DataFrame(rows)
        max_round_num = max(df['round'])
        print("max_round_num: ", max_round_num)
 
        if isAttack:
            df = df[(df['round'] == max_round_num) & (df['hop_distance']!=0) ]
        else:
            df = df[(df['round'] == max_round_num)]
  
        
        df1 = df[(df['hop_distance']==1)]
        df2 = df[(df['hop_distance']==2)]
        df['Distance']= 'All'
        df1['Distance']= 'Hop-1'
        df2['Distance']= 'Hop-2'

        df=pd.concat([df, df1,df2])
        df=pd.concat([df, df1])
        

        self.df = df
        #print("df:", self.df)
        #print("Round 100: ")

        #for iden in ['Random', 'Degree', 'ENS', 'PageRank', 'Clustering']:
        #    dddd = self.df_all[(self.df_all['round'] == 100) & (self.df_all['identifier'] == iden)]
        #    print('\n', iden, '\n', dddd[["test_acc","attack_succ"]].mean())

        #print("Round 70: ")
        #print(self.df_all[(self.df_all['round'] == 70)].mean(axis=0))

        #self.plot_final('attack_succ', 'Attack Success', val_vis)
        #self.plot_final('test_acc', 'Test Acc', val_vis)
        

    def plot_final(self, y, ylabel, val_vis = True):
        plt.clf()
        print(self.df)
        if self.mode == 'newbar':
            ax =sns.barplot(x=None, y=y, data=self.df, hue='identifier', errorbar=None)

        elif self.mode == 'bar':
            #ax =sns.barplot(x='identifier',y=y,data=self.df,hue='Distance',errorbar=None)
            ax =sns.barplot(x='identifier',y=y,data=self.df,hue='Distance')
        elif self.mode =='line':
            ax =sns.lineplot(x='identifier',y=y,data=self.df,hue='Distance',errorbar=None, style="Distance", markers= True)
        elif self.mode == 'normalized':
            ax =sns.lineplot(x='identifier',y=y,data=self.df,hue='Distance',errorbar=None, style="Distance", markers= True)


            for container in ax.containers:
                print('data', container.data_values())
                # ax.bar_label(container,fmt= '%.2f')


                #points = path.get_offsets()
                #print(f"hue: {path.get_label()}; y-values: {points[:, 1]}")
            #all_x_values = [path.get_offsets()[:, 0] for path in ax.collections]
            #all_y_values = [path.get_offsets()[:, 1] for path in ax.collections]
            # print(all_x_values)
            # print(all_y_values)



        file_path = self.path

        if val_vis:
            for container in ax.containers:
                ax.bar_label(container,fmt= '%.2f')

        plt.xlabel(self.xlabel)
        plt.ylabel(ylabel)
        file_path = f'{self.path}_{y}.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path,bbox_inches='tight')

        # print(df)

    def combine_plot(self, other,  xlabel, y, ylabel, path):
        df1 = self.df
        df2 = other.df

        df1['Selection']= 'Random'
        df2['Selection']= 'Pagerank'

        df=pd.concat([df1,df2])
        print('-'*50)
        print(df.columns)
        print(df)
        print(pd.pivot_table(df, values=[y], index=['identifier','Distance'],
                    columns=['Selection',], aggfunc=np.mean))
        print('-'*50)

        plt.clf()
        ax =sns.lineplot(x='identifier',y=y,data=df,hue='Distance',errorbar=None, style="Selection", markers= True)

        file_path = path

        for container in ax.containers:
           ax.bar_label(container,fmt= '%.2f')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(file_path,bbox_inches='tight')





fault_tolerant = {
    '0': [15, 16, 17],
    '2': [108, 109, 110],
    '4': [111, 112, 113],
    '6': [114, 115, 116]
}

vary_dataset_pagerank = {
    'EMNIST': [30, 31, 32],
    'Fashion MNIST': [81, 82, 83],
}

vary_dataset_pagerank_v2 = {
    'MNIST': [5001],
    'EMNIST': [301],
    'Fashion': [811],
}

vary_boosting = {
    'MNIST boosting=10': [5001],
    'MNIST boosting=5': [5003],
    'MNIST boosting=1': [5002],
}

vary_dataset_random = {
    'Emnist': [15, 16, 17],
    'Fashion MNIST': [84, 85, 86],
}

vary_graph_pagerank = {
    'Erdos': [63, 64, 65],
    'Watts': [30, 31, 32],
    'Barabasi': [66, 67, 68],
}

vary_graph_random= {
    'Erdos': [72, 73, 74],
    'Watts': [15, 16, 17],
    'Barabasi': [75, 76, 77],
}

vary_selection= {
    'Random': [15, 16, 17],
    'Degree': [24, 25, 26],
    'ENS': [27, 28, 29],
    'PageRank': [30, 31, 32],
    'Clustering': [33, 34, 35],
}

vary_adversary_count_random= {
    1: [9, 10, 11],
    2: [12, 13, 14],
    3: [15, 16, 17],
    4: [18, 19, 20],
    5: [21, 22, 23],
    6: [3, 4, 5],
}

vary_adversary_count_pagerank= {
    "1": [42, 43, 44],
    "2": [45, 46, 47],
    "3": [30, 31, 32],
    "4": [36, 37, 38],
    "5": [39, 40, 41],
    "6": [48, 49, 50],
}


vary_total_node_random= {
    60: [15, 16, 17],
    80: [60, 61, 62],
    100: [54, 55, 56],
}

vary_total_node_pagerank= {
    "60": [30, 31, 32],
    "80": [57, 58, 59],
    "100": [51, 52, 53],
}

'''
non-iid gets less dta in these exp, redo
#non_iid_defense = {
#    'IID': [30, 31, 32],
#    'non-IID':[156, 157, 158],
#    'IID Defense':[150, 151, 152],
#    'non-IID Defense': [162, 163, 164],
#}
'''

non_iid_defense = {
    'IID': [30, 31, 32],
    'non-IID':[1561],
    'IID Defense':[189, 190, 191],  # results for 0.05 malicious
    'non-IID Defense': [1622],
}

defense_v2 = {
    'No Defense': [48, 49, 50],
    'Trimmed Mean':[132, 133, 134],
    'Clipping 1': [129, 130, 131],
    'Clipping 0.5': [144, 145, 146],
    'Clipping 0.25': [147, 148, 149],
    'Our Defense':[150, 151, 152], # results here are for 0.1 malicious
}

defense = {
   'No defense': [3, 4, 5],
   'Clipping 1': [129, 130, 131],
   'Clipping 0.5': [144, 145, 146],
   'Clipping 0.25': [147, 148, 149],
#    'Trim Mean'
   'T.Mean':[132, 133, 134],
#    'T.Median':[135, 136, 137],
#    'T.Mean, Clip':[138, 139, 140],
   # 'T.Median, Clip':[141, 142, 143]


}

Clipping = {
   'No defense': [3, 4, 5],
   'Clipping 1': [129, 130, 131],
   'Clipping 0.5': [144, 145, 146],
   'Clipping 0.25': [147, 148, 149],
#    'Trim Mean'
#    'T.Median':[135, 136, 137],
#    'T.Mean, Clip':[138, 139, 140],
   # 'T.Median, Clip':[141, 142, 143]
}

TrimmedMean = {
   'No defense': [3, 4, 5],
#    'Clipping 1': [129, 130, 131],
#    'Clipping 0.5': [144, 145, 146],
#    'Clipping 0.25': [147, 148, 149],
#    'Trim Mean'
   'T.Mean':[132, 133, 134],
#    'T.Mean, Clip':[138, 139, 140],
   # 'T.Median, Clip':[141, 142, 143]


}

OurDefense = {
   'No defense': [3, 4, 5],
#    'Clipping 1': [129, 130, 131],
#    'Clipping 0.5': [144, 145, 146],
#    'Clipping 0.25': [147, 148, 149],
#    'Trim Mean'
   'Our defense':[150, 151, 152],
#    'T.Mean, Clip':[138, 139, 140],
   # 'T.Median, Clip':[141, 142, 143]
}

# TODO: use later noniid and local experiments
# I: iid
# P: partial view
# G: global view
# D: defense

non_iid_partial_pagerank_2 = {
    'N P seed 20 nodes=120 k=5': [160],
}

non_iid_partial_pagerank_3 = {
    'N G seed 2':[156],
    'N G seed 20':[157],
    'N G seed 99':[158],
    'N P seed 2': [159],
    'N P seed 20': [160],
    'N P seed 99': [161],
}

non_iid_partial_pagerank_4 = {
    'I G': [30, 31, 32],
    'I P':[153, 154, 155],
    'N G':[156, 157, 158],
    'N P': [159, 160, 161],
}

#non_iid_vs_iid_global_pagerank = {
#    'iid': [30, 31, 32],
#    'non-iid':[156, 157, 158],
non_iid_vs_iid_global_pagerank = {
    'IID': [30, 31, 32],
    'non-IID':[156, 157, 158],
}

#non-iid gets less data in these exp, redo
#partial_view = {
#    'Global IID': [30, 31, 32],
#    'Partial IID':[153, 154, 155],
#    'Global non-IID':[156, 157, 158],
#    'Partial non-IID': [159, 160, 161],
#}

partial_view_alpha1 = {
    'Global IID': [30, 31, 32],
    'Partial IID':[153, 154, 155],
    'Global non-IID':[1561],
    'Partial non-IID': [1591, 1601, 1611], # alpha = 1
}


iid_vs_noniid_alpha = {
    'IID': [30, 31, 32],
    "non-IID alpha=10":[156],
    "non-IID alpha=1":[1561],
    "non-IID alpha=0.1":[1562],
}

non_iid_partial_pagerank = {
    'I G': [30, 31, 32],
    'I P':[153, 154, 155],
    'N G':[156, 157, 158],
    'N P': [159, 160, 161],
    'I G D':[150, 151, 152],
    'I P D':[168, 169, 170],
    'N G D': [162, 163, 164],
    'N P D': [165, 166, 167],
}


non_iid_partial_view_random = {
    'I G' : [15, 16, 17], # for random
    'I P': [177, 178, 179], # for random
    'N G':[183, 184,185 ], #185
    'N P': [171, 172, 173] # for random
    # 'I G D':[150, 151, 152],
    # 'I P D':[168, 169, 170],
    # 'N G D': [162, 163, 164],
    # 'N P D': [165, 166, 167],
}

non_iid_partial_view_ens = {
    'I G': [27, 28, 29], #for ens
    'I P' : [180, 181, 182],# for ens
    'N G':[186, 187, 188],
    'N P': [174, 175, 176] # for ens
  
    # 'I G D':[150, 151, 152],
    # 'I P D':[168, 169, 170],
    # 'N G D': [162, 163, 164],
    # 'N P D': [165, 166, 167],
}


result_folder = 'Results10' 



# dataset
'''
acc_list = [0.76, 0.91, 0.98]
res2 = Data(vary_dataset_pagerank_v2, None, f'{result_folder}/dataset/Dataset_pagerank_hops', 'bar')
res2.plot_rounds('attack_succ','Attack Success',f'{result_folder}/dataset/dataset_pagerank_attack_evolution.png', style=None)
res2.plot_rounds('test_acc','Test Accuracy',f'{result_folder}/dataset/dataset_pagerank_accuracy_evolution.png', style=None)
res2.plot_by_accuracy(f'{result_folder}/dataset/dataset_pagerank.png', 'datasets_v2', acc_list)
'''
'''
acc_list = [0.76, 0.91, 0.98]
res2 = Data(vary_boosting, None, f'{result_folder}/boosting/boosting_pagerank_hops', 'bar')
res2.plot_rounds('attack_succ','Attack Success',f'{result_folder}/boosting/boosting_pagerank_attack_evolution.png', style=None)
res2.plot_rounds('test_acc','Test Accuracy',f'{result_folder}/boosting/boosting_pagerank_accuracy_evolution.png', style=None)
res2.plot_by_accuracy(f'{result_folder}/boosting/boosting_pagerank.png', 'boosting', acc_list)
'''

# Data(vary_dataset_random, None, f'{result_folder}/Dataset_random', 'bar')

# topology
'''
acc_list = [0.9, 0.92, 0.95, 0.97]
res5  = Data(vary_graph_pagerank, None, f'{result_folder}/graph_type_pagerank/graph_type_pagerank', 'bar')
res5.plot_rounds('attack_succ','Attack Success',f'{result_folder}/graph_type_pagerank/graph_type_pagerank_attack_evolution.png', style=None)
res5.plot_rounds('test_acc','Test Accuracy',f'{result_folder}/graph_type_pagerank/graph_type_pagerank_accuracy_evolution.png', style=None)
res5.plot_by_accuracy(f'{result_folder}/graph_type_pagerank/graph_types.png', 'graph_type', acc_list)
'''

'''
res3 = Data(vary_graph_random, None, f'{result_folder}/graph_type_random/graph_type_random', 'bar')
res3.plot_rounds('attack_succ','Attack Success',f'{result_folder}/graph_type_random/graph_type_random_attack_evolution.png', style=None)
res3.plot_rounds('test_acc','Test Accuracy',f'{result_folder}/graph_type_random/graph_type_random_accuracy_evolution.png', style=None)
'''


# selection strategy
'''
res4 = Data(vary_selection, None, f'{result_folder}/selection/selection_last_round', 'bar')
res4.plot_rounds('attack_succ','Attack Success',f'{result_folder}/selection/selection_attack_evolution.png', style=None)
res4.plot_rounds('test_acc','Test Accuracy',f'{result_folder}/selection/selection_accuracy_evolution.png', style=None)
res4.plot_by_accuracy(f'{result_folder}/selection/selection_strategies.png', 'strategies')
'''


# iid vs non-iid
'''
#acc_list = [0.8, 0.85, 0.9, 0.95, 0.97]
acc_list = [0.93, 0.95, 0.97]
changedlabels = ["IID", r"non-IID $\alpha=10$", r"non-IID $\alpha=1$", r"non-IID $\alpha=0.1$"]
#res6 = Data(non_iid_vs_iid_global_pagerank, None, f'{result_folder}/non_iid_vs_iid_global_pagerank/non_iid_vs_iid_global_pagerank_bars', 'bar', val_vis=False)
res6 = Data(iid_vs_noniid_alpha, None, f'{result_folder}/non_iid_vs_iid_global_pagerank/non_iid_vs_iid_global_pagerank_bars', 'bar', val_vis=False)
res6.plot_rounds('attack_succ','Attack Success',f'{result_folder}/non_iid_vs_iid_global_pagerank/non_iid_vs_iid_global_pagerank_attack_evolution.png', style=None, changedLabels=changedLabels)
res6.plot_rounds('test_acc','Test Accuracy',f'{result_folder}/non_iid_vs_iid_global_pagerank/non_iid_vs_iid_global_pagerank_accuracy_evolution.png', style=None, changedLabels=changedLabels)
res6.plot_by_accuracy(f'{result_folder}/non_iid_vs_iid_global_pagerank/non_iid_vs_iid_global_pagerank.png', 'iid_vs_noniid_alpha', acc_list)
'''

# partial view

#acc_list = [0.8, 0.85, 0.9, 0.95, 0.97]

acc_list = [0.9, 0.95, 0.97]
res8 = Data(partial_view_alpha1, None, f'{result_folder}/partial_view_alpha1/partial_view_pagerank', 'bar', val_vis=False)
res8.plot_rounds('attack_succ','Attack Success',f'{result_folder}/partial_view_alpha1/partial_view_pagerank_attack_evolution.png', style=None)
res8.plot_rounds('test_acc','Test Accuracy',f'{result_folder}/partial_view_alpha1/partial_view_pagerank_accuracy_evolution.png', style=None)
res8.plot_by_accuracy(f'{result_folder}/partial_view_alpha1/partial_view.png', 'partial_view', acc_list)


# iid vs non-iid with defense
'''
acc_list = [0.9, 0.95, 0.97]
res12 = Data(non_iid_defense, None, f'{result_folder}/defense_noniid/defense_noniid_bar', 'bar', val_vis=False)
res12.plot_rounds('attack_succ','Attack Success',f'{result_folder}/defense_noniid/defense_noniid_attack_evolution.png', style=None)
res12.plot_rounds('test_acc','Test Accuracy',f'{result_folder}/defense_noniid/defense_noniid_accuracy_evolution.png', style=None)
res12.plot_by_accuracy(f'{result_folder}/defense_noniid/defense_noniid.png', 'defense_noniid', acc_list)
'''


# adversary count
'''
acc_list = [0.8, 0.9, 0.93, 0.96]
res9 = Data(vary_adversary_count_pagerank, None, f'{result_folder}/adversary_count/advcount_pagerank_hops', 'bar')
res9.plot_rounds('attack_succ','Attack Success',f'{result_folder}/adversary_count/advcount_pagerank_attack_evolution.png', style=None)
res9.plot_rounds('test_acc','Test Accuracy',f'{result_folder}/adversary_count/advcount_pagerank_accuracy_evolution.png', style=None)
res9.plot_by_accuracy(f'{result_folder}/adversary_count/advcount_pagerank.png', 'adversary_count', acc_list)
'''

# total node count with attack
'''
acc_list = [0.8, 0.9, 0.93, 0.96]
res10 = Data(vary_total_node_pagerank, None, f'{result_folder}/node_count_pagerank/node_count_pagerank_hops', 'bar')
res10.plot_rounds('attack_succ','Attack Success',f'{result_folder}/node_count_pagerank/node_count_pagerank_attack_evolution.png', style=None)
res10.plot_rounds('test_acc','Test Accuracy',f'{result_folder}/node_count_pagerank/node_count_pagerank_accuracy_evolution.png', style=None)
res10.plot_by_accuracy(f'{result_folder}/node_count_pagerank/node_count_pagerank.png', 'node_count', acc_list)
'''

# fault tolerance
'''
acc_list = [0.8, 0.9, 0.95, 0.97]
res9 = Data(fault_tolerant, 'r robust', f'{result_folder}/fault_tolerant/fault_tolerant_pagerank_hops', 'bar')
res9.plot_rounds('attack_succ','Attack Success',f'{result_folder}/fault_tolerant/fault_tolerant_pagerank_attack_evolution.png', style=None)
res9.plot_rounds('test_acc','Test Accuracy',f'{result_folder}/fault_tolerant/fault_tolerant_pagerank_accuracy_evolution.png', style=None)
res9.plot_by_accuracy(f'{result_folder}/fault_tolerant/fault_tolerant_pagerank.png', 'fault_tolerant', acc_list)
'''

# defense
'''
#acc_list = [0.8, 0.9, 0.95, 0.97]
acc_list = [0.75, 0.84, 0.92, 0.96]

res11 = Data(defense_v2, None, f'{result_folder}/defense/defense_hops', 'newbar', val_vis=False)
#res11.plot_max_round(f'{result_folder}/defense/defense_hops_2', 'defense')

res11.plot_rounds('attack_succ','Attack Success',f'{result_folder}/defense/defense_attack_evolution.png', style=None, moveLegend=True)
res11.plot_rounds('test_acc','Test Accuracy',f'{result_folder}/defense/defense_accuracy_evolution.png', style=None,  moveLegend=True)
res11.plot_by_accuracy(f'{result_folder}/defense/defense.png', 'defense', acc_list)
'''

'''
Max acc:  No Defense 0.9637539386749268
Max acc:  Clipping 0.25 0.7577880024909973
Max acc:  Clipping 0.5 0.8456602692604065
Max acc:  Clipping 1 0.9273608922958374
Max acc:  Trimmed Mean 0.8461011052131653
Max acc:  Our Defense 0.9603252410888672
'''

# Data(fault_tolerant, 'r robust', f'{result_folder}/Fault_tolerant', 'bar')


# Data(non_iid_partial_view_random, None, f'{result_folder}/non_iid_partial_view_random', 'bar', val_vis=False)
# Data(non_iid_partial_view_ens, None, f'{result_folder}/non_iid_partial_view_ens', 'bar', val_vis=False)

# clip = Data(defense, None, f'{result_folder}/Clipping', None)
# clip.plot_rounds('attack_succ','Attack Success',f'{result_folder}/Clipping_attack.png', style=None)
# clip.plot_rounds('test_acc','Test Accuracy',f'{result_folder}/Clipping_test.png', style=None)

# Data(Clipping, None, f'{result_folder}/Clipping_bar', 'bar')
# Data(TrimmedMean, None, f'{result_folder}/TrimmedMean_bar', 'bar')
# Data(OurDefense, None, f'{result_folder}/OurDefense', 'bar')
# Data(vary_selection, None, f'{result_folder}/Selection_normalized', 'normalized')


# adversary_random = Data(vary_adversary_count_random, None, f'{result_folder}/adversary_count', 'line')
# adversary_pagerank =  Data(vary_adversary_count_pagerank, None, f'{result_folder}/adversary_count_pagerank', 'line')
# adversary_random.combine_plot(adversary_pagerank,'Adversary Count', 'attack_succ', 'Attack Success', f'{result_folder}/adversary_count.png')
# node_random = Data(vary_total_node_random, None, f'{result_folder}/node_count_random', 'line')
# node_pagerank = Data(vary_total_node_pagerank, None, f'{result_folder}/node_count_pagerank', 'line')
# node_random.combine_plot(node_pagerank, 'Node Count', 'attack_succ', 'Attack Succcess',f'{result_folder}/node_count.png')






