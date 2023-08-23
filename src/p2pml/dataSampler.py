import numpy as np
import os
from tensorflow import keras 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from p2pml.commons import NUM_CLASS, DATA_PATH, PROJECT_DIR, BACKDOOR_PATTERN, DIRICHLET_ALPHA, datasetsExperiments

class Dataset:
    
    def __init__(self,config,numClients,adversarialNodes):
        self.name = config.get('data')
        self.backdoorPattern = BACKDOOR_PATTERN[self.name]
        self.config = config
        self.numClients = numClients
        self.numClass = NUM_CLASS[self.name]
        self.targetClass = config.get('targetClass')
        self.alpha = config.get('alpha', None)
        self.adversarialNodes = adversarialNodes
        self.kwargs = config.get('kwargs')
        self._loadData()
        self._split()
        self._backdoorClients()
        self._backdoorTestSet()
        # print('-'*50)
        # print('backdoor_pattern',self.backdoorPattern)
        # print('-'*50)


    def _loadData(self):
        name = self.config.get('data')
        if name in ["mnist","fashionMnist",'emnist','dbpedia']:
            dirdata = DATA_PATH[name]
            self.x_train = np.load(os.path.join(dirdata,"x_train.npy"))
            self.y_train = np.load(os.path.join(dirdata,"y_train.npy"))
            self.x_test  = np.load(os.path.join(dirdata,"x_test.npy"))
            self.y_test  = np.load(os.path.join(dirdata,"y_test.npy"))

            if name in ['mnist', 'fashionMnist']:
                self.x_train = self.x_train.astype("float32") / 255
                self.x_test = self.x_test.astype("float32") / 255
                # Make sure images have shape (28, 28, 1)
                self.x_train = np.expand_dims(self.x_train, -1)
                self.x_test = np.expand_dims(self.x_test, -1)

            self.y_test = keras.utils.to_categorical(self.y_test, NUM_CLASS[name])
        else:
            raise NotImplementedError 
       

    def _split(self,):
        
        if self.config.get('iid') == True:
            self._split_iid()
        else:
            self._split_non_iid()

        self.plot_data_distrib()


    def plot_data_distrib(self,):

        print(self.matrix)
        experimentNumber = self.config.get('id')

        sns.set()
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 6))

        #cmap="rocket_r",
        sns.heatmap(
            self.matrix,
            cmap="hot_r",
            annot=True,
            fmt=".0f",
            linewidth=1.0,
            square=False,
        )
        plt.xlabel ('Client Id')
        plt.ylabel ('Class (Label)')

        plt.tight_layout()
        plt.savefig('data_distrib_{}_{}.png'.format(self.config.get('iid'), experimentNumber))


    def _split_non_iid(self,):

        self.matrix = np.zeros(shape=(self.numClass, self.numClients), dtype=np.int64)
        
        minimumClassSize = np.min(np.unique(self.y_train, return_counts=True)[1])
        print("minimum class size:", minimumClassSize)
        print("shape train data Y:", self.y_train.shape)

        rng = np.random.default_rng(seed=self.config.get("seed"))
        groupByClass = [ [x for x,y in zip(self.x_train,self.y_train) if y == targetClass] for targetClass in range(self.numClass) ]
        
        for group in  groupByClass:
            rng.shuffle(group)

        self.clientData = []

        if not self.alpha:
            self.alpha = DIRICHLET_ALPHA
        
        all_dirichlets = rng.dirichlet(
            [self.alpha for _ in range(self.numClients)],
            self.numClass
        )

        print("alpha", "all_dirichlets: ", self.alpha, all_dirichlets)

        total_samples = 0
        total_data_distrib = dict()
        for targetClass in range(self.numClass):
            total_data_distrib[targetClass] = 0

        clientX = dict()
        clientY = dict()
        for clientInd in range(self.numClients):
            clientX[clientInd] = []
            clientY[clientInd] = []

        # all_dirichlets[clientInd] -> distribution of each class for  client clientInd
        for targetClass in range(self.numClass):

            curr_counts = minimumClassSize * all_dirichlets[targetClass]
            for clientInd in range(self.numClients):
                clientSize = int(curr_counts[clientInd])
                #print("Client size per class: ", targetClass, clientSize)
                targetClassData = groupByClass[targetClass][:clientSize]
                #print("Len target class data:", len(targetClassData))

                groupByClass[targetClass] = groupByClass[targetClass][clientSize:]

                clientX[clientInd] += targetClassData
                clientY[clientInd] += [targetClass for i in range(clientSize)]

                self.matrix[targetClass, clientInd] = len(targetClassData)
                total_data_distrib[targetClass] += clientSize    

        for clientInd in range(self.numClients):
            clientYcrt = keras.utils.to_categorical(clientY[clientInd], self.numClass)
            clientXcrt = np.array(clientX[clientInd])
            #print("shape X:",clientXcrt.shape)
            #print("shape Y:",clientYcrt.shape)
            total_samples += len(clientYcrt)
            self.clientData.append((clientXcrt, clientYcrt))
        print("total_samples: ", total_samples)
        print("total_data_distrib:", total_data_distrib)


    def _split_iid(self,):
      
        self.matrix = np.zeros(shape=(self.numClass, self.numClients))
        minimumClassSize = np.min(np.unique(self.y_train,return_counts=True)[1])
        print("actual minimum class size:", minimumClassSize)
        
        if datasetsExperiments:
            minimumClassSize = 5400 # our Fashion=6000, mnist=5421 training set is this
        else:
            minimumClassSize = np.min(np.unique(self.y_train,return_counts=True)[1])

        clientSize = minimumClassSize//self.numClients
        print("minimum class size:", minimumClassSize)
        print("clientSize:", clientSize)
        print("shape train data Y:", self.y_train.shape)
        groupByClass = [ [x for x,y in zip(self.x_train,self.y_train) if y == targetClass] for targetClass in range(self.numClass) ]

        rng = np.random.default_rng(seed=self.config.get("seed"))
        for group in  groupByClass:
            rng.shuffle(group)
        
        self.clientData = []

        total_samples = 0
        
        for clientInd in range(self.numClients):

            clientX = []
            clientY = []
            for targetClass in range(self.numClass):

                targetClassData = groupByClass[targetClass][:clientSize]
                groupByClass[targetClass] = groupByClass[targetClass][clientSize:]
                clientX += targetClassData
                clientY += [targetClass for i in range(clientSize)]
            
                self.matrix[targetClass,clientInd] = len(targetClassData)

            clientY = keras.utils.to_categorical(clientY, self.numClass)
            clientX = np.array(clientX)
            print("shape X:",clientX.shape)
            print("shape Y:",clientY.shape)
       
            self.clientData.append((clientX, clientY))
            total_samples += len(clientY)
        print("total_samples: ", total_samples)

    def getClientData(self):
        return self.clientData

    def getTestX(self,):
        return self.x_test
    
    def getTestY(self,):
        return self.y_test

    def _backdoorClients(self):
        print(self.adversarialNodes)
        for client in self.adversarialNodes:
            self._backdoorSingleClient(client)
        
    def _backdoorTestSet(self,):
        
        size = self.x_test.shape[0]
        print("size test set:", size)

        #divide to two non overlapping
        self.x_backdoor =self.x_test[:size//2]
        self.y_backdoor =self.y_test[:size//2]

        self.x_test = self.x_test[size//2:]
        self.y_test = self.y_test[size//2:]

        X = []
        Y = []
        # print(np.unique(np.argmax(self.y_backdoor,axis=1),return_counts=True))
        for x,y in zip(self.x_backdoor,self.y_backdoor):

            if np.argmax(y) != self.targetClass:
                X.append( self._addBackdoor(x))
                Y.append(self.targetClass)
        self.x_backdoor  = np.array(X)
        self.y_backdoor =  keras.utils.to_categorical(Y, self.numClass)
        # print(self.x_backdoor.shape, self.y_backdoor.shape)
        # print("DODODDO")   

    def _addBackdoor(self,x):

        # shape (28,28,1)
        # set a square backdoor
        k = 3
        data = x[:,:,0]
  
        # the corner is either white (1) or black/gray (<1),dependingon the dataset
        #  we will flip the color of a  k-size square
        #for i in range(k):
        #    for j in range(k):
        #        if data[i,j] != data[0,0]:
        #            print(data[0, 0], data[i,j])

        if data[0,0]  == 1:
            data[0:k,0:k] = 0
        else:
            data[0:k,0:k] = 1

        data = np.expand_dims(data,axis=-1)
        return data

        
    def getBackdoorTestX(self):
        return self.x_backdoor

    def getBackdoorTestY(self):
        return self.y_backdoor

    def _backdoorSingleClient(self,index):
        trainX,trainY = self.getClientData()[index]
        poisonedCount = 0
        PDR = self.kwargs.get('adversary').get('PDR')
        non_target_class_sample_count = sum(trainY.argmax(axis=1) != self.targetClass)
        backDoorCount = int(non_target_class_sample_count * PDR)
        X = []
        Y = []
        # print(np.unique(np.argmax(self.y_backdoor,axis=1),return_counts=True))
        poisonedCount = 0
        for x,y in zip(trainX,trainY):
            y_argmax = np.argmax(y)
            if poisonedCount < backDoorCount and y_argmax != self.targetClass:
                X.append( self._addBackdoor(x))
                Y.append(self.targetClass)
                poisonedCount +=1
                
            else:
                X.append(x)
                Y.append(y_argmax)
        assert poisonedCount == backDoorCount, f'backDoorCount : {backDoorCount},poisonedCount : {poisonedCount} '

        trainX = np.array(X)
        trainY = keras.utils.to_categorical(Y, self.numClass)
        self.clientData[index] = (trainX,trainY)


def sample(config,numberNodes,adversarialNodes):
    experimentNumber = config.get('id')

    dataset = Dataset(config, numberNodes,adversarialNodes)
    clientData = dataset.getClientData()
    testX = dataset.getTestX()
    testY = dataset.getTestY()
    backDoorTestX = dataset.getBackdoorTestX()
    backDoorTestY = dataset.getBackdoorTestY()
    for client,id in zip(clientData,range(numberNodes)):  
        trainX,trainY = client

        with open(os.path.join(PROJECT_DIR, f'experiment_{experimentNumber}' ,f'Peers/peer-{id}/x_test.npy'), 'wb') as f_testX:
            np.save(f_testX, testX)
        with open(os.path.join(PROJECT_DIR, f'experiment_{experimentNumber}' ,f'Peers/peer-{id}/y_test.npy'), 'wb') as f_testY:
            np.save(f_testY, testY)
        with open(os.path.join(PROJECT_DIR, f'experiment_{experimentNumber}' ,f'Peers/peer-{id}/x_backdoor_test.npy'), 'wb') as f_backDoorTestX:
            np.save(f_backDoorTestX, backDoorTestX)
        with open(os.path.join(PROJECT_DIR, f'experiment_{experimentNumber}' ,f'Peers/peer-{id}/y_backdoor_test.npy'), 'wb') as f_backDoorTestY:
            np.save(f_backDoorTestY, backDoorTestY)
        with open(os.path.join(PROJECT_DIR, f'experiment_{experimentNumber}' ,f'Peers/peer-{id}/x_train.npy'), 'wb') as f_x:
            np.save(f_x, trainX)
        with open(os.path.join(PROJECT_DIR, f'experiment_{experimentNumber}' ,f'Peers/peer-{id}/y_train.npy'), 'wb') as f_y:
            np.save(f_y, trainY)


    
    


