try:
    import sys
    import numpy as np
    import os
    from tensorflow import keras
    # print("\nEffective group id: ",os.getegid())
    # print("Effective user id: ",os.geteuid())
    import sys
    import json
    from sklearn.model_selection import train_test_split
    import data.models as models
    import random
    import pickle
    roundNumber = sys.argv[1] 
    peerCount = sys.argv[2]
    ADVERSARY  = sys.argv[3] == 'true'
    MODEL_POISONING  = sys.argv[4] == 'true'
    ADVERSARY_EPOCHS  = int(sys.argv[5])
    K_TOLERANT      = sys.argv[6] if (len(sys.argv) > 6 and sys.argv[6] != 'null') else None
    CLIPPING_NORM       = sys.argv[7] if (len(sys.argv) > 7 and sys.argv[7] != 'null') else None
    DEFENSE             = sys.argv[8] if (len(sys.argv) > 8 and sys.argv[8] != 'null') else None
    LOCAL_NORM             = sys.argv[9] if (len(sys.argv) > 9 and sys.argv[9] != 'null') else None
    if K_TOLERANT:
        K_TOLERANT = int(K_TOLERANT)
    if CLIPPING_NORM:
        CLIPPING_NORM = float(CLIPPING_NORM)
    if LOCAL_NORM:
        LOCAL_NORM = float(LOCAL_NORM)

    print('-'* 50)
    print(f'CLIPPING_NORM: {CLIPPING_NORM}')
    print(f'DEFENSE: {DEFENSE}')
    print(f'LOCAL_NORM: {LOCAL_NORM}')
    print('-'* 50)


    num_classes = 10
    with open('/app/data/x_train.npy', 'rb') as f:
        train_X = np.load(f)
    with open('/app/data/y_train.npy', 'rb') as f:
        train_y = np.load(f)
    with open('/app/data/x_test.npy', 'rb') as f:
        test_X = np.load(f)
    with open('/app/data/y_test.npy', 'rb') as f:
        test_y = np.load(f)
    with open('/app/data/x_backdoor_test.npy', 'rb') as f:
        test_backdoor_X = np.load(f)
    with open('/app/data/y_backdoor_test.npy', 'rb') as f:
        test_backdoor_y = np.load(f)

    batch_size = 128
    epochs = 1

    def getArchitecture():
        model = models.getModel()
        return model


    def getModel(path):
        model = getArchitecture()
        model.load_weights(os.path.join('/app/data/',path))
        return model

    
    def clip_weights(round, weights_li, clip_nm=1, local_norm=None):
        # Compute the 2-norm of the difference between the old and new weights for
        # each client.
        old_weights = getModel(f'/app/data/mymodel-{int(round)}.h5').get_weights()
        if local_norm is None:
            norm_li = [np.linalg.norm([np.linalg.norm(
                old - w) for (old, w) in zip(old_weights, user)]) for user in weights_li]

            print(f'Previous Norms:{norm_li}')
            # Clip the norms
            norm_li = [max(1, norm / clip_nm) for norm in norm_li]
            print(f'Clipped norms:{norm_li}')
            # Divide the updated by the clipped norms
            clipped_li = [[(w - old) / norm + old for (old, w) in zip(old_weights, user)]
                        for (user, norm) in zip(weights_li, norm_li)]

        else:
            local_weight = weights_li.pop()
            local_weight_norm = np.linalg.norm([np.linalg.norm(
                old - w) for (old, w) in zip(old_weights, local_weight)])
            conversion_norm_local = max(1, local_weight_norm / local_norm)
            clipped_local = [(w - old) / conversion_norm_local + old for (old, w) in zip(old_weights, local_weight)]
                        
            print(f'local_norm norm is : {local_norm}' )
            print(f'local weight norm is : {local_weight_norm}' )
            print(f'conversion_norm_local is : {conversion_norm_local}' )


            norm_li = [np.linalg.norm([np.linalg.norm(
                old - w) for (old, w) in zip(old_weights, user)]) for user in weights_li]

            print(f'Previous Norms:{norm_li}')
            # Clip the norms
            norm_li = [max(1, norm / clip_nm) for norm in norm_li]
            print(f'Clipped norms:{norm_li}')
            # Divide the updated by the clipped norms
            clipped_li = [[(w - old) / norm + old for (old, w) in zip(old_weights, user)]
                        for (user, norm) in zip(weights_li, norm_li)]
            
            clipped_li.append(clipped_local)

        return clipped_li

    def getModelsByRound(roundnum):
        fileNames = os.listdir('/app/data')
        filterNames = []
        twoPreviousNames = []
        twoPrevious = int(roundnum) -2

        for file in fileNames:
            if file.count('-') != 2:
                continue
            file = file.split('.')[0] # take the part before . i.e 1-4-0
            curr_round = file.split('-')[2]
            if roundnum == curr_round:
                filterNames.append(file+'.h5')
            if twoPrevious == int(curr_round):
                twoPreviousNames.append(file+'.h5')

        if len(filterNames) != int(peerCount):
            print(fileNames)
            print(filterNames)
            print(f'For round: {round} need: {peerCount} arrived : {len(filterNames) }')
            sys.exit(1)

        twoPreviousNames = [os.path.join('/app/data/',path) for path in twoPreviousNames]
        for file in twoPreviousNames:
            os.remove(file)

        other_weights = [getModel(i).get_weights() for i in filterNames]

        if K_TOLERANT:
            print('k tolerant part')
            print(len(other_weights))
            subset =  random.choices(other_weights, k = max(1, int(peerCount) - K_TOLERANT ) )
            print(len(subset))
            print('k tolerant part ends')
            other_weights = subset
        else:
            print('No fault tolerance')

        return other_weights


    def trainSingleRound(localModel,round):
        ## shuffle
        X_train, X_val, y_train, y_val = train_test_split(
        train_X, train_y, test_size=0.33, random_state=42)

        model = getModel(localModel)
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"])
        model.fit(X_train, y_train, batch_size=batch_size,epochs= epochs,  shuffle=True, validation_data=(X_val,y_val),verbose=2)
        return model.get_weights()


    def trainSingleRound_backdoor(localModel, round, boost_factor):
            """ Apply the current round's weights and perform model substitution attack

            This is a boosting model replacement attack as described in 
            https://arxiv.org/pdf/1807.00459.pdf
            The model update required to substitute the general model is
            evil_update = old_weights + boost_factor * (adv_weights - old_weights)
            boost_factor = n_clients / learning_rate

            Args:
                current_weights (list): list of numpy arrays representing the global model weights
                boost_factor (float): factor to boost the poisoned model's weights 

            Returns:
                tuple: final weights, train loss, train accuracy
            """
            X_train, X_val, y_train, y_val = train_test_split(
            train_X, train_y, test_size=0.33, random_state=42)

            model = getModel(localModel)
            current_weights = model.get_weights()

            model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"])
            model.fit(X_train, y_train, batch_size=batch_size,epochs= ADVERSARY_EPOCHS,  shuffle=False, validation_data=(X_val,y_val),verbose=2)

            new_weights=  model.get_weights()

            diff = [new - old for (old, new) in zip(current_weights, new_weights)]
            boosted_weights = [d * boost_factor +
                        old for (old, d) in zip(current_weights, diff)]

            return boosted_weights

    def trimmed_mean(weights_li, mode= 'mean'):

        arr = np.asanyarray(weights_li)
        user_count, layer_count = arr.shape
        trim_size = 1
        trimmed_layers = []
        for layer_idx in range(layer_count):
            layer = arr[:, layer_idx]
            # stack each layer
            a = np.stack([layer[user_count] for user_count in range(user_count)] ,axis=0)
            # sort based on user
            a.sort(axis=0)
            # trim
            a =a[trim_size:user_count-trim_size]
            # take mean or median
            if mode == 'mean':
                a = a.mean(axis=0)
            elif mode == 'median':
                a = np.median(a, axis=0)
            # add layer back
            trimmed_layers.append(a)
        return np.asanyarray(trimmed_layers)

    
    def updateModel(round):

        weights = getModelsByRound(round)
        print('peer models loaded')
        if ADVERSARY and MODEL_POISONING:
            localModel = trainSingleRound_backdoor(f'/app/data/mymodel-{round}.h5',int(round),boost_factor=10)
            # localModel = trainSingleRound_backdoor(f'/app/data/mymodel-{round}.h5',int(round),boost_factor=5)
            print('local training done with model poisoning')

        else:
            localModel = trainSingleRound(f'/app/data/mymodel-{round}.h5',int(round))
            print('local training done')
        
        weights.append(localModel)

        if not ADVERSARY and CLIPPING_NORM:
            print('-'*30)
            print('clipping part')
            weights = clip_weights(round, weights, CLIPPING_NORM, LOCAL_NORM)
            print('-'*30)

        else:
            print('No clipping')

        if not ADVERSARY and DEFENSE is not None:
            if DEFENSE == 'trimmed-mean':
                print('-'*30)
                print('trimmed-mean')
                agg = trimmed_mean(weights, 'mean')
                # with open('/app/data/mypickle.pickle', 'wb') as f:
                #     pickle.dump(weights, f)
                #weights = clip_weights(round, weights, CLIPPING_NORM)

                print('-'*30)

            elif int(round) >  30 and  DEFENSE == 'trimmed-median':

                print('-'*30)
                print('trimmed-median')
                agg = trimmed_mean(weights, 'median')
                # with open('/app/data/mypickle.pickle', 'wb') as f:
                #     pickle.dump(weights, f)
                #weights = clip_weights(round, weights, CLIPPING_NORM)

                print('-'*30)
            else:
                print('Standart averaging')
                agg  = np.mean(weights,axis=0)
                agg = np.asanyarray(agg)     

        else:
            print('Standart averaging')

            agg  = np.mean(weights,axis=0)
            agg = np.asanyarray(agg)     

        model = getArchitecture()
        model.set_weights(agg)
        model.save(f'/app/data/mymodel-{int(round)+1}.h5')
        print(f'Round {int(round)+1} completed merged and saved.')

    def evaluateModel(round):
        model =  getModel(f'/app/data/mymodel-{round}.h5')
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"])
        loss,acc = model.evaluate(x=test_X, y=test_y, verbose=0)
        loss_bacdoor,acc_backdoor =model.evaluate(x=test_backdoor_X, y=test_backdoor_y, verbose=0)
        evalutation_report = {'test_acc':acc, 'test_backdoor_acc':acc_backdoor, 'test_loss':loss, 'test_backdoor_loss':loss_bacdoor}
        print(f'Round {round}, evaluation: {evalutation_report}')
        json_path = f'/app/data/Evaluation/evaluation-{round}.json'
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(evalutation_report, f)
    

    try:
        updateModel(roundNumber)
        evaluateModel(str(int(roundNumber)+1))
        sys.exit(0)
    except (FileNotFoundError,OSError) as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(e)
        sys.exit(2)
except Exception as e:
    print(e)

