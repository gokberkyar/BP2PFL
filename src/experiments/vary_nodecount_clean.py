import itertools
import json
import os


def node_count_clean():

    node_counts = [80, 100]
    datasets = ['emnist']
    graphTypes = ['watts']

    seeds=[2,20,99]


    start=102
    counter =0
    for dataset,graph, node_count,seed in itertools.product(datasets, graphTypes,node_counts,seeds):
        kwargs={'complete':{"k":node_count},
        'erdos':{"erdos-nodeCount":node_count, "erdos-p":0.1,"erdos-seed":500},
        'watts':{"watts-nodeCount":node_count, "watts-k":12, "watts-p" :0.5,"watts-seed":500},
        'barabasi':{"barabasi-nodeCount":node_count, "barabasi-k":12, "barabasi-seed":500}
        }


        id = start+counter
        experimentFile = f'Experiments/experiment_{id}.json'
        data= {
        "id":id,
        "data":dataset,
        "graphType":graph,
        "iid":True,
        "targetClass":2,
        "kwargs":kwargs[graph],
        'seed':seed,
        }

        os.makedirs(os.path.dirname(experimentFile),exist_ok=True)
        print(experimentFile)
        with open(experimentFile,'w+') as f:
            json.dump(data,f,indent=4)
            

        counter+=1
    return counter

