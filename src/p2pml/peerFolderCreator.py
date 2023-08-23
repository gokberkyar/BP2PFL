import os
import shutil
from p2pml.commons import PROJECT_DIR
import os

# os.path.join(PROJECT_DIR,f'Peers/peer-{id}

def delPeersFolder(experimentNumber):
    '''
    Deletes Peers folder recursively
    
    '''
    path = os.path.join(PROJECT_DIR,f'experiment_{experimentNumber}','Peers')
    print(f'debug path: {path}')
    shutil.rmtree(path)

def checkPeersFolder(experimentNumber):
    '''
    Returns True if Peers Folder Exists
    '''
    path = os.path.join(PROJECT_DIR,f'experiment_{experimentNumber}','Peers')
    return os.path.exists(path)

def createPeersFolder(k, experimentNumber):
    '''
    Creates a new Peers parent folder and peer-0 to peer-k-1 subfolders 
    '''
    for i in range(k):
        path = os.path.join(PROJECT_DIR,f'experiment_{experimentNumber}',f'Peers/peer-{i}')
        os.makedirs(path)

def peerFolderMain(k,experimentNumber):
    
    if checkPeersFolder(experimentNumber):
        delPeersFolder(experimentNumber)
    createPeersFolder(k, experimentNumber)
