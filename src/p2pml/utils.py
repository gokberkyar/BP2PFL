import json
import os
import time
from p2pml.commons import START_PORT,KEY_LIST_PATH,PROJECT_DIR,MODELS_DIR,WEIGHTS_DIR, NUM_ROUNDS, OUTPUT_DIR
import shutil
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from numpy.random import default_rng
import docker
import threading
from collections import  defaultdict
from visualization.collectLogs import collectEvaluationFiles
import pickle
import heapq
from collections import deque


def writeModelPy(graph,config):
    experimentNumber = config.get('id')
    name = config['data']
    if name not in ['dbpedia','mnist','emnist','fashionMnist']:
        raise NotImplemented

    for node in graph.nodes:
        dst = os.path.join(PROJECT_DIR, f'experiment_{experimentNumber}' ,f'Peers/peer-{node}/models.py')
        src = os.path.join(MODELS_DIR,f'{name}.py')
        #copy model.py
        shutil.copyfile(src, dst)
        #copy intialWeight
        dst = os.path.join(PROJECT_DIR, f'experiment_{experimentNumber}', f'Peers/peer-{node}/mymodel-0.h5')
        src = os.path.join(PROJECT_DIR, f'experiment_{experimentNumber}', f'Weights/{name}.h5')
        shutil.copyfile(src, dst)


def getAdversarialNodes(config,graph,numberNodes, verbose = True):
    experimentNumber = config.get('id')
    plt.figure()
    adversary_kwarg = config.get('kwargs').get('adversary',None)
    if adversary_kwarg is None:
        return []

    print(adversary_kwarg)
    structure = adversary_kwarg.get('structure')
    ratio     = adversary_kwarg.get('ratio')
    count = int(ratio * numberNodes)

    #random
    #clustered
    #Nodes of highest degree
    #Nodes of highest effective network size (ENS)
    #Nodes of highest PageRank
    #Nodes of highest clustering coefficient

    if  structure == 'random' :
        rng = default_rng(seed=config.get('seed'))
        adversarialNodes= rng.choice(list(range(numberNodes)),size=(count))
 

    elif structure == 'clustered':
        rng = default_rng(seed=config.get('seed'))

        startNode =  rng.choice(nx.center(graph))
        startNode = int(startNode)
        neighbours = list(graph.neighbors(startNode))
        others =  list(rng.choice(neighbours,replace=False,size =(count-1)))
        adversarialNodes = [startNode] + others
    
        # print(f'neighbours:{neighbours},others:{others}, adversarialNodes:{adversarialNodes}')
    elif structure == 'highestdegree':
        degrees =nx.degree(graph)
        print(degrees)
        largest_count = heapq.nlargest(count, degrees, key= lambda x: x[1])
        print(largest_count)
        adversarialNodes = list(map(lambda x:x[0], largest_count))
    elif structure == 'highestens':
        degrees =nx.effective_size(graph)
        print(degrees)
        largest_count = heapq.nlargest(count, degrees, key= lambda x:degrees[x])
        print(largest_count)
        adversarialNodes = largest_count
        
    elif structure == 'highestpagerank':
        print(f'edges: {graph.edges()}')
        degrees =nx.pagerank(graph)
        print(degrees)

        print(dict(sorted(degrees.items(), key=lambda item: item[1], reverse=True)))

        largest_count = heapq.nlargest(count, degrees, key= lambda x:degrees[x])

        print("Top pagerank scores:", heapq.nlargest(30, degrees, key= lambda x:degrees[x]))
        print(largest_count)
        adversarialNodes = largest_count


    elif structure == 'clusteringcoefficient':
        degrees =nx.clustering(graph)
        print(degrees)
        largest_count = heapq.nlargest(count, degrees, key= lambda x:degrees[x])
        print(largest_count)
        adversarialNodes = largest_count
    elif structure == 'local':


        '''
        Input:  
                - Undirected Graph, G
                - probability p
                - maximum depth d
                - maximum budget c
        Output: 
                - List of adversarial nodes

        Choose a random node in the graph, G
        Do a limited depth contrainted breadth first search starting in this node V.
        Details for limited depth contrainted bfs:
            - choose probability, p 
            - choose a maximum depth,  d
            - choose a maximum budget of visible nodes (percentage of total nodes),  c

            - rather than de-facto bfs, next node will be added to a probability p to queue
            - dont continue expanding if node is more than d steps from V
            - probability decays exponentially, probability at depth d : p^d

        p, d and c are hardcoded
        '''

        p = 0.5
        d = 3
        c = 0.20

        # create a random number generator
        rng = default_rng(seed=config.get('seed'))

        # budget percentange -> count
        visible_budget =  int(numberNodes * c)
        
        # visible nodes
        visible_nodes = set()
        
        # bfs queue
        queue = deque()

        # push the start node with its depth
        # tuple (node, depth)

        # bfs already visited nodes
        visited_nodes = set()
        while len(visited_nodes) < numberNodes and  len(visible_nodes) < visible_budget:
            # choose the random node V
            random_node= rng.choice(list(range(numberNodes))) 
            print('the random node :', random_node)
            # random node is already visited
            visited_nodes.add(random_node)
            visible_nodes.add(random_node) 
            queue.append((random_node, 0) )

            # do bfs
            while queue:
                
                # pop the curr node with its depth from queue
                node, node_depth = queue.popleft()
                
                # neighbours of node
                adjs = nx.neighbors(graph, node)

                # traverse in node adjs 
                for adj in adjs:

                    # if adj already visited ignore
                    if adj in visited_nodes:
                        continue 

                    # if depth is more than maximum ignore
                    if node_depth + 1 > d:
                        continue
                    
                    # generate a random value between 0 and 1
                    random_val = rng.random()

                    # level visibility threshold, p ^ d
                    threshold = p ** (node_depth+1)

                    if random_val <= threshold and len(visible_nodes) < visible_budget:
                        visible_nodes.add(adj)
                        queue.append((adj, node_depth + 1))

                    visited_nodes.add(adj)

        visible_nodes = list(visible_nodes)
        visible_egdes_both_in_visiblity = [(x,y) for (x,y) in graph.edges() if x in visible_nodes and y in visible_nodes]
        print(f'visible edges: {visible_egdes_both_in_visiblity}')
        visible_subgraph = nx.Graph()
        visible_subgraph.add_edges_from(visible_egdes_both_in_visiblity)
     
        print(f'visible nodes: {visible_nodes}, visible node count: {len(visible_nodes)}')
        print(f'total edge count: {len(nx.edges(graph))}')
        print(f'single subgraph edge count: {len(nx.edges(visible_subgraph))}')

        degrees =nx.pagerank(visible_subgraph)
        
        print("Sorted pagerank scores in visible nodes:", heapq.nlargest(len(visible_nodes), degrees, key= lambda x:degrees[x]))

        largest_count = heapq.nlargest(count, degrees, key= lambda x:degrees[x])
        adversarialNodes = largest_count

    elif structure == 'localrandom':


        '''
        Input:  
                - Undirected Graph, G
                - probability p
                - maximum depth d
                - maximum budget c
        Output: 
                - List of adversarial nodes

        Choose a random node in the graph, G
        Do a limited depth contrainted breadth first search starting in this node V.
        Details for limited depth contrainted bfs:
            - choose probability, p 
            - choose a maximum depth,  d
            - choose a maximum budget of visible nodes (percentage of total nodes),  c

            - rather than de-facto bfs, next node will be added to a probability p to queue
            - dont continue expanding if node is more than d steps from V
            - probability decays exponentially, probability at depth d : p^d

        p, d and c are hardcoded
        '''

        p = 0.5
        d = 3
        c = 0.20

        # create a random number generator
        rng = default_rng(seed=config.get('seed'))

        # budget percentange -> count
        visible_budget =  int(numberNodes * c)
        
        # visible nodes
        visible_nodes = set()
        
        # bfs queue
        queue = deque()

        # push the start node with its depth
        # tuple (node, depth)

        # bfs already visited nodes
        visited_nodes = set()
        while len(visited_nodes) < numberNodes and  len(visible_nodes) < visible_budget:
            # choose the random node V
            random_node= rng.choice(list(range(numberNodes))) 
            print('the random node :', random_node)
            # random node is already visited
            visited_nodes.add(random_node)
            visible_nodes.add(random_node) 
            queue.append((random_node, 0) )

            # do bfs
            while queue:
                
                # pop the curr node with its depth from queue
                node, node_depth = queue.popleft()
                
                # neighbours of node
                adjs = nx.neighbors(graph, node)

                # traverse in node adjs 
                for adj in adjs:

                    # if adj already visited ignore
                    if adj in visited_nodes:
                        continue 

                    # if depth is more than maximum ignore
                    if node_depth + 1 > d:
                        continue
                    
                    # generate a random value between 0 and 1
                    random_val = rng.random()

                    # level visibility threshold, p ^ d
                    threshold = p ** (node_depth+1)

                    if random_val <= threshold and len(visible_nodes) < visible_budget:
                        visible_nodes.add(adj)
                        queue.append((adj, node_depth + 1))

                    visited_nodes.add(adj)

        visible_nodes = list(visible_nodes)
        # visible_egdes_both_in_visiblity = [(x,y) for (x,y) in graph.edges() if x in visible_nodes and y in visible_nodes]
        # print(f'visible edges: {visible_egdes_both_in_visiblity}')
        # visible_subgraph = nx.Graph()
        # visible_subgraph.add_edges_from(visible_egdes_both_in_visiblity)
     
        # print(f'visible nodes: {visible_nodes}, visible node count: {len(visible_nodes)}')
        # print(f'total edge count: {len(nx.edges(graph))}')
        # print(f'single subgraph edge count: {len(nx.edges(visible_subgraph))}')

        # degrees =nx.pagerank(visible_subgraph)
        # largest_count = heapq.nlargest(count, degrees, key= lambda x:degrees[x])
        # adversarialNodes = largest_count
        adversarialNodes = rng.choice(visible_nodes, 3).tolist()
    elif structure == 'localens':


        '''
        Input:  
                - Undirected Graph, G
                - probability p
                - maximum depth d
                - maximum budget c
        Output: 
                - List of adversarial nodes

        Choose a random node in the graph, G
        Do a limited depth contrainted breadth first search starting in this node V.
        Details for limited depth contrainted bfs:
            - choose probability, p 
            - choose a maximum depth,  d
            - choose a maximum budget of visible nodes (percentage of total nodes),  c

            - rather than de-facto bfs, next node will be added to a probability p to queue
            - dont continue expanding if node is more than d steps from V
            - probability decays exponentially, probability at depth d : p^d

        p, d and c are hardcoded
        '''

        p = 0.5
        d = 3
        c = 0.20

        # create a random number generator
        rng = default_rng(seed=config.get('seed'))

        # budget percentange -> count
        visible_budget =  int(numberNodes * c)
        
        # visible nodes
        visible_nodes = set()
        
        # bfs queue
        queue = deque()

        # push the start node with its depth
        # tuple (node, depth)

        # bfs already visited nodes
        visited_nodes = set()
        while len(visited_nodes) < numberNodes and  len(visible_nodes) < visible_budget:
            # choose the random node V
            random_node= rng.choice(list(range(numberNodes))) 
            print('the random node :', random_node)
            # random node is already visited
            visited_nodes.add(random_node)
            visible_nodes.add(random_node)
            queue.append((random_node, 0) )

            # do bfs
            while queue:
                
                # pop the curr node with its depth from queue
                node, node_depth = queue.popleft()
                
                # neighbours of node
                adjs = nx.neighbors(graph, node)

                # traverse in node adjs 
                for adj in adjs:

                    # if adj already visited ignore
                    if adj in visited_nodes:
                        continue 

                    # if depth is more than maximum ignore
                    if node_depth + 1 > d:
                        continue
                    
                    # generate a random value between 0 and 1
                    random_val = rng.random()

                    # level visibility threshold, p ^ d
                    threshold = p ** (node_depth+1)

                    if random_val <= threshold and len(visible_nodes) < visible_budget:
                        visible_nodes.add(adj)
                        queue.append((adj, node_depth + 1))

                    visited_nodes.add(adj)

        visible_nodes = list(visible_nodes)
        visible_egdes_both_in_visiblity = [(x,y) for (x,y) in graph.edges() if x in visible_nodes and y in visible_nodes]
        print(f'visible edges: {visible_egdes_both_in_visiblity}')
        visible_subgraph = nx.Graph()
        visible_subgraph.add_edges_from(visible_egdes_both_in_visiblity)
     
        print(f'visible nodes: {visible_nodes}, visible node count: {len(visible_nodes)}')
        print(f'total edge count: {len(nx.edges(graph))}')
        print(f'single subgraph edge count: {len(nx.edges(visible_subgraph))}')
        degrees =nx.effective_size(visible_subgraph)
        largest_count = heapq.nlargest(count, degrees, key= lambda x:degrees[x])
        adversarialNodes = largest_count
        
    else:
        raise NotImplementedError(f' structure: {structure} is not implemented')
    
    assert count == len(adversarialNodes)
    if verbose:
        color_map = ['red' if i in adversarialNodes else 'lightblue' for i in range(numberNodes) ]
        edgeList = []
        for x,y in graph.edges:
            if x in adversarialNodes or y in adversarialNodes:
                edgeList.append((x,y))
        edge_color = ['red' if u in adversarialNodes or v in adversarialNodes else 'lightblue' for u,v in graph.edges]
        widths = [1 if u in adversarialNodes or v in adversarialNodes else 0.8 for u,v in graph.edges]
        styles = ['solid' if u in adversarialNodes or v in adversarialNodes else 'dashed' for u,v in graph.edges]

        nx.draw(graph, node_color=color_map, edge_color= edge_color, with_labels=True, width=widths,style=styles)
        plt.savefig(f'Results/Graph-{experimentNumber}.png')
    return adversarialNodes


def writePeersJson(graph, adversarialNodes, config):
    experimentNumber = config.get('id')

    with open(KEY_LIST_PATH) as keyListFile:
        keyList= json.load(keyListFile)
    for node in graph.nodes:
        adversary = node in adversarialNodes
        kwargs = config.get('kwargs')
        k_tolerant = kwargs.get('k_tolerant', None)

        clipping_norm   = kwargs.get('clipping_norm', None)
        local_norm   = kwargs.get('local_norm', None)
        defense = kwargs.get('defense', None)
        
        node_count = None
        possible_counts = ['barabasi-nodeCount', 'watts-nodeCount', 'erdos-nodeCount','k']
        for graph_type_count in possible_counts:
            if graph_type_count in kwargs:
                node_count = kwargs[graph_type_count]
        assert(node_count is not None)

        if not adversary:
            model_poisoning = False
            adversary_epochs = -1 
        else:
            adversary_kwarg = kwargs.get('adversary')
            model_poisoning = adversary_kwarg.get('model_poisoning')
            adversary_epochs = adversary_kwarg.get('epochs')

        peerJson = {'id':node, 'k_tolerant':k_tolerant, 'clipping_norm':clipping_norm,'local_norm':local_norm, 'defense':defense,  'adversary': adversary, 'model_poisoning': model_poisoning, 'adversary_epochs':adversary_epochs, 'peers':[ f'peer{i}'  for i  in graph.neighbors(node)] + [f'peer{node}'],'port':node+START_PORT, 'keyList':keyList, 'nodeCount':node_count, 'numRounds':NUM_ROUNDS + 1 }

        #print("peerJson: ", peerJson)

        with open(os.path.join(PROJECT_DIR, f'experiment_{experimentNumber}', f'Peers/peer-{node}/peers.json'), 'w') as f:
            json.dump(peerJson,f)

def getGraph(config,verbose=True):
    graphType = config['graphType']
    kwargs = config['kwargs']
    if graphType ==  'complete':
        graph = nx.complete_graph(kwargs['k'])
    elif graphType ==  'petersen':
        graph = nx.petersen_graph()
    elif graphType ==  'tutte':
        graph = nx.tutte_graph()
    elif graphType ==  'erdos':
        graph = nx.erdos_renyi_graph(kwargs['erdos-nodeCount'],kwargs['erdos-p'],kwargs['erdos-seed'])
    elif graphType ==  'watts':
        graph  = nx.connected_watts_strogatz_graph(kwargs['watts-nodeCount'],kwargs['watts-k'],kwargs['watts-p'],seed=kwargs['watts-seed'])
    elif graphType ==  'barabasi':
        graph = nx.barabasi_albert_graph(kwargs['barabasi-nodeCount'],kwargs['barabasi-k'],seed=kwargs['barabasi-seed'])
    else:
        raise NotImplemented

    if verbose:
        print(f"number of nodes: {graph.number_of_nodes()}")
        print(f"number of edges: {graph.number_of_edges()}")
        print(f"eccentricity  (eccentricity for a node v is the maximum distance from v to all other nodes in G) : {nx.eccentricity(graph)}")
        print(f"radius (min eccentricity): {nx.radius(graph)}")
        print(f"diameter (max eccentricity): {nx.diameter(graph)}")
        print(f"center (nodes that their edge count equal to radius): {nx.center(graph)}")
        print(f"periphery (nodes that their edge count equal to diameter): {nx.periphery(graph)}")
        print(f"density (edge count/(n(n-1))): {nx.density(graph)}")
        print(f"Is Connected : {nx.is_connected(graph)}")
    return graph

def readConfig(experimentNumber):
        with open(f'Experiments/experiment_{experimentNumber}.json') as f:
            config = json.load(f)
        return config

def runNodes(numberNodes, config):
    client = docker.from_env()
    experimentNumber = config.get('id')
    print("numberNodes:", numberNodes)

    def stopContainers():
        print('Stopping Containers')

        for i in range(numberNodes):
            try:
                container = client.containers.get(f'peer{i}')
                container.stop(timeout=0)
            except docker.errors.NotFound:
                continue

    def removeContainers():
        print('Removing Containers')

        for i in range(numberNodes):
            try:
                container = client.containers.get(f'peer{i}')
                container.remove(v=True)
            except docker.errors.NotFound:
                continue

    def buildWorker():
        print('Building worker')
        image = client.images.build(path='Networking/worker',tag='worker')
        print(image)

    def waitContainers():
        print('Waiting Containers')
        for i in range(numberNodes):
            try:
                container = client.containers.get(f'peer{i}')
                container.wait()
            except docker.errors.NotFound:
                    continue

    def removeNetwork():
        print('Removing myNetwork if exists')
        try:
            network = client.networks.get(f'myNetwork')
            network.remove()
        except docker.errors.NotFound:
            pass

    def createNetwork():
        print('Creating myNetwork')

        ipam_pool = docker.types.IPAMPool(
            subnet='124.42.0.0/16',
            iprange='124.42.0.0/24',
            gateway='124.42.0.254',

        )
        ipam_config = docker.types.IPAMConfig(
            pool_configs=[ipam_pool])

        client.networks.create("myNetwork", driver="bridge", ipam=ipam_config)




    def collectStats(e, numberNodes):
        # client = docker.from_env()
        # history = defaultdict(list)
        minute = 0
        while not e.is_set():
            time.sleep(60)
            minute += 1
            print(f'Total Mins passed:{minute}')
            # for peer in range(numberNodes):
            #     container = client.containers.get(f'peer{peer}')
            #     log = container.logs().decode().splitlines()

            #     for line in log:
            #         search_key ='evaluation: ' 
            #         pos = line.find(search_key)
            #         if pos != -1:
            #             evalution_dict = json.loads(line[pos+ len(search_key):].replace("'", '"'))
            #             history[peer].append(evalution_dict)
            # print(history)

    stopContainers()
    removeContainers()
    buildWorker()
    time.sleep(2)
    removeNetwork()
    createNetwork()

    print('Running Images')
    containers = []
    for i in range(numberNodes):
        container = client.containers.run('worker',\
            #  ports={f'{START_PORT+i}/tcp': START_PORT + i},\
            volumes=[f'{PROJECT_DIR}/experiment_{experimentNumber}/Peers/peer-{i}/:/app/data'], \
            detach=True, name=f'peer{i}',  nano_cpus=1000000000, network='myNetwork')
        containers.append(container)

    e = threading.Event()
    wait_thread = threading.Thread(name='wait_for', target=waitContainers)
    collect_thread = threading.Thread(name='collect_stats', target=collectStats, args= (e,numberNodes))
    
    wait_thread.start()
    collect_thread.start()

    wait_thread.join()
    e.set()
    collect_thread.join()



def copyEvaluations(numberNodes,experimentNumber):


    history = collectEvaluationFiles(numberNodes,NUM_ROUNDS,experimentNumber )
    file_path = os.path.join(OUTPUT_DIR,f'output_{experimentNumber}.pickle')
    os.makedirs(os.path.dirname(file_path),exist_ok=True )
    with open(file_path, 'wb') as f:
        pickle.dump(history, f)
    print(f'Experiment is saved to {file_path} ')





            


    


