from . import emnist
from . import fashionMnist
from . import mnist
import os

from p2pml.commons import PROJECT_DIR
def initiliazeWeights(experimentNumber):
    experimentFolder = os.path.join(PROJECT_DIR, f'experiment_{experimentNumber}')
    filepath  = os.path.join(experimentFolder, f'Weights/emnist.h5')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    emnist.getModel().save(filepath)
    
    filepath  = os.path.join(experimentFolder, f'Weights/fashionMnist.h5')
    fashionMnist.getModel().save(filepath)
    
    filepath  = os.path.join(experimentFolder, f'Weights/mnist.h5')
    mnist.getModel().save(filepath)



