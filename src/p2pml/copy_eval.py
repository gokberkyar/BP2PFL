import sys
import p2pml.utils as utils
from models import createInitialWeight

def copy_eval(experimentNumber):
    # read experiment number from user and read respected config json
    config = utils.readConfig(experimentNumber)
    
    # construct ml graph using networkx library
    graph = utils.getGraph(config)
    numberNodes = len(graph.nodes)
    
    # copy experiment outputs
    utils.copyEvaluations(numberNodes, experimentNumber)
    
    # print experiment is over
    print('Experiment is finished successfully')
    print('Terminating gracefully...')

def main():
    experimentNumber = sys.argv[1]
    copy_eval(experimentNumber)

if __name__ == '__main__':
    main()
