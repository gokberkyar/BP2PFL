import itertools
import json
import os
def vary_node_count():
    datasets = ['emnist']
    graphTypes = ['watts']

    poisoningTypes = [True]
    ratios = [ 0.05]
    structureTypes = ['highestpagerank','random']
    seeds=[2,20,99]
    node_counts = [100,80]
    start=51
    counter =0
    for node_count, dataset,graph, poisoningType, ratio, structureType,seed  in itertools.product(node_counts, datasets, graphTypes, poisoningTypes, ratios, structureTypes,seeds):
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
        data['kwargs']['adversary'] ={
            "PDR":0.5,
            "ratio":ratio,
            "structure":structureType,
            "model_poisoning":poisoningType,
            "epochs":5

        }

        os.makedirs(os.path.dirname(experimentFile),exist_ok=True)
        print(experimentFile)
        with open(experimentFile,'w+') as f:
            json.dump(data,f,indent=4)
            

        counter+=1
    return counter