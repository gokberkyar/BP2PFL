import itertools
import json
import os


def clean_graph_type():
    datasets = ['emnist','fashionMnist']
    graphTypes = ['watts','erdos', 'barabasi','complete']
    kwargs={'complete':{"k":60},
        'erdos':{"erdos-nodeCount":60, "erdos-p":0.1,"erdos-seed":500},
        'watts':{"watts-nodeCount":60, "watts-k":12, "watts-p" :0.5,"watts-seed":500},
        'barabasi':{"barabasi-nodeCount":60, "barabasi-k":12, "barabasi-seed":500}
    }
    seeds=[2,20,99]


    start=87
    counter =0
    for dataset,graph,seed in itertools.product(datasets, graphTypes,seeds):
        if dataset == 'fashionMnist' and graph != 'watts':
            continue

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

