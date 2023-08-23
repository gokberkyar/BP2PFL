import itertools
import json
datasets = ['emnist','fashionMnist']
graphTypes = ['complete','erdos','watts','barabasi']
kwargs={'complete':{"k":60},
    'erdos':{"erdos-nodeCount":60, "erdos-p":0.1,"erdos-seed":500},
    'watts':{"watts-nodeCount":60, "watts-k":12, "watts-p" :0.5,"watts-seed":500},
    'barabasi':{"barabasi-nodeCount":60, "barabasi-k":12, "barabasi-seed":500}
}
poisoningTypes = [True,False]
ratios = [0.1, 0.05, 0.0167]
structureTypes = ['random','clustered']




start=0
counter =0
for dataset,graph, poisoningType, ratio, structureType in itertools.product(datasets, graphTypes, poisoningTypes, ratios, structureTypes):

    id = start+counter
    experimentFile = f'experiment_{id}.json'
    data= {
    "id":id,
    "data":dataset,
    "graphType":graph,
    "iid":True,
    "targetClass":2,
    "kwargs":kwargs[graph],
    'seed':1,
    }
    data['kwargs']['adversary'] ={
        "PDR":0.5,
        "ratio":ratio,
        "structure":structureType,
        "model_poisoning":poisoningType,
        "epochs":5

    }


    with open(experimentFile,'w+') as f:
        json.dump(data,f,indent=4)
        

    counter+=1