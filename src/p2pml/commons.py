import os 
USER_HOME = os.path.expanduser('~')
PROJECT_DIR_BASE = '/net/data/p2pml/p2pml'
#PROJECT_DIR = f'{PROJECT_DIR_BASE}/results_jun15_n30k12'
PROJECT_DIR = f'{PROJECT_DIR_BASE}/results_jun13_n60k12'
#PROJECT_DIR = f'{USER_HOME}/p2pml_simona_jun6_n60k12'
KEY_LIST_PATH = f'{USER_HOME}/p2pml/Networking/peerId/id.json'
MODELS_DIR    = f'{USER_HOME}/p2pml/src/models'
WEIGHTS_DIR   = f'{USER_HOME}/p2pml/Weights'
OUTPUT_DIR    = f'{PROJECT_DIR}/Output'
START_PORT = 9000

NUM_CLIENT_EPOCHS = {
    "mnist":2,
}

INPUT_SHAPE ={
    "mnist":(28,28,1),
}

NUM_CLASS = {
    'mnist': 10,
    'emnist': 10,
    'fashionMnist':10,
    'dbpedia': 14
}

# fashionEmnist size could be big, exprementation will show it is proper or not.
#CLIENT_SIZE ={
#    "mnist":1000,
#    'emnist':500,
#    'fashionMnist':200,   
#}

BATCH_SIZE ={
    "mnist":32,
}

NUM_ROUNDS = 200

datasetsExperiments = True

DATA_PATH ={
    "mnist"         : f"{PROJECT_DIR_BASE}/datasets/mnist",
    "fashionMnist" : f"{PROJECT_DIR_BASE}/datasets/fashionMnist",
    "dbpedia" : f"{PROJECT_DIR_BASE}/datasets/dbpedia",
    "emnist" : f"{PROJECT_DIR_BASE}/datasets/emnist",
}

BACKDOOR_PATTERN={
    "mnist":1,
    "emnist" : 0,
    'fashionMnist':1
}

BACKDOOR_PATTERN_TOTAL={
    "mnist" : 9, # the backdoor pixel set to 1; summed over 9 pixels
    "emnist" : 0, # backdoor pixel val is set to 0
    'fashionMnist':9  # backdoor pixel val set to 1; 9 pixels form a backdoor so 1x9=9; 
}

# https://arxiv.org/pdf/2205.09986.pdf
#DIRICHLET_ALPHA = 10
DIRICHLET_ALPHA = 1



