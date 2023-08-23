import numpy as np
import os
from p2pml.commons import NUM_CLASS, DATA_PATH,PROJECT_DIR,BACKDOOR_PATTERN_TOTAL
from matplotlib import pyplot as plt
import sys
import p2pml.utils as utils
import p2pml.peerFolderCreator as peerFolderCreator
import p2pml.dataSampler as dataSampler


def verify_peer(id,is_adversary,PDR, experimentNumber, dataset_name):
    def is_backdoor_exist(x):
        k = 3
        backdoor_pattern_total = BACKDOOR_PATTERN_TOTAL[dataset_name]
        print('-'*50)
        print('9 square total expected backdoor pattern', backdoor_pattern_total)
        print('-'*50)

        data = x[:,:,:,0]
        region = data[:,0:k,0:k]
        sum_region = region.sum(axis= (1,2))
        backdoored_count = np.isclose(sum_region,backdoor_pattern_total).sum()
        if not is_adversary:
            assert np.isclose(backdoored_count, 0), f'{backdoored_count}-{id}'

    with open(os.path.join(PROJECT_DIR, f'experiment_{experimentNumber}', f'Peers/peer-{id}/x_train.npy'), 'rb') as f:
        train_X = np.load(f)
    
    is_backdoor_exist(train_X)

def verify(config,numberNodes,adversarialNodes):
    experimentNumber = config.get('id')
    dataset_name = config.get('data')

    if len(adversarialNodes) == 0:
        return
    PDR = config.get('kwargs').get('adversary').get('PDR')

    for i in range(numberNodes):
        verify_peer(i, i in adversarialNodes ,PDR, experimentNumber,dataset_name)
    print('sampling and backdoors verified')

def main(experiment_number=None):
    if experiment_number is None:
        experiment_number = sys.argv[1]
    config = utils.readConfig(experiment_number)
    # construct ml graph using networkx library
    graph = utils.getGraph(config)
    numberNodes = len(graph.nodes)
    # get adversarial nodes based on configuration
    adversarialNodes = utils.getAdversarialNodes(config,graph,numberNodes)
    # create peer folders and other files for each peer
    peerFolderCreator.peerFolderMain(numberNodes)
    utils.writePeersJson(graph, adversarialNodes)
    
    # sample with respect to config, numberNodes and adversarial Nodes
    dataSampler.sample(config,numberNodes,adversarialNodes)
    for i in range(numberNodes):
        verify_peer(i, i in adversarialNodes ,0.5)
    print('all tests are passed')

if __name__ == '__main__':
    main()


