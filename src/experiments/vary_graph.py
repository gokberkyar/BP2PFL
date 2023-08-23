import itertools
import json
import os
def vary_graph():
    datasets = ['emnist']
    graphTypes = ['erdos', 'barabasi','complete' ]

    poisoningTypes = [True]
    ratios = [ 0.05]
    structureTypes = ['highestpagerank','random']
    seeds=[2,20,99]
    start=63
    counter =0
    for  structureType, dataset,graph, poisoningType, ratio, seed  in itertools.product(structureTypes, datasets, graphTypes, poisoningTypes, ratios,seeds):
        kwargs={'complete':{"k":60},
        'erdos':{"erdos-nodeCount":60, "erdos-p":0.1,"erdos-seed":500},
        'watts':{"watts-nodeCount":60, "watts-k":12, "watts-p" :0.5,"watts-seed":500},
        'barabasi':{"barabasi-nodeCount":60, "barabasi-k":12, "barabasi-seed":500}
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