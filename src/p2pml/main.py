import p2pml.peerFolderCreator as peerFolderCreator
import p2pml.dataSampler as dataSampler
import sys
import p2pml.utils as utils
import p2pml.dataVerifier as dataVerifier
from models import createInitialWeight

def p2pml(experimentNumber):
    # read experiment number from user and read respected config json
    config = utils.readConfig(experimentNumber)
    # create random weights
    createInitialWeight.initiliazeWeights(experimentNumber)
    # construct ml graph using networkx library
    graph = utils.getGraph(config)
    numberNodes = len(graph.nodes)
     
    # get adversarial nodes based on configuration
    adversarialNodes = utils.getAdversarialNodes(config,graph,numberNodes)
    print(f'adversarialNodes: {adversarialNodes}')
    
    # create peer folders and other files for each peer
    peerFolderCreator.peerFolderMain(numberNodes,experimentNumber)
    print("Finished peerFolderCreator")
    
    utils.writePeersJson(graph, adversarialNodes, config)
    print("Finished writePeersJson")
    
    # sample with respect to config, numberNodes and adversarial Nodes
    dataSampler.sample(config,numberNodes,adversarialNodes)
 
    # verify backdoor clients and sampling
    dataVerifier.verify(config,numberNodes,adversarialNodes)

    # write model.py file to each peer
    utils.writeModelPy(graph,config)
    print("Finished writing models")
    
    # run docker containers
    utils.runNodes(numberNodes, config)
    print("Finished running nodes")
    
    # copy experiment outputs
    utils.copyEvaluations(numberNodes, experimentNumber)
    
    # print experiment is over
    print('Experiment is finished successfully')
    print('Terminating gracefully...')

def main():
    experimentNumber = sys.argv[1]
    p2pml(experimentNumber)

if __name__ == '__main__':
    main()
