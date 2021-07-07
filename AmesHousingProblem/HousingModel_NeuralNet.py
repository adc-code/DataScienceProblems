#
# HousingModel_NeuralNet.py
#

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import itertools

import HousingDataUtils
import BigNums


MAX_EPOCHS      = 2
PATIENCE        = 3
NET_STYLE       = 2

FIND_OPTIMAL_PARAMS = False # True

BASE_OUTPUT_NAME    = 'Results/TestPreds_NN'
RESULTS_FILE        = 'Results_NN.csv'



#
# BuildModel... 
#
def BuildModel (modType, inputShape):

    tf.random.set_seed (42)
    model = keras.models.Sequential ()

    if modType == 1:
        model.add (keras.layers.Dense (80, kernel_initializer='normal', activation='relu', input_shape=inputShape))
        model.add (keras.layers.Dense (40, kernel_initializer='normal', activation='relu'))
        model.add (keras.layers.Dense (20, kernel_initializer='normal', activation='relu'))

    elif modType == 2:
        model.add (keras.layers.Dense (128, kernel_initializer='normal', activation='relu', input_shape=inputShape))
        model.add (keras.layers.Dense (256, kernel_initializer='normal', activation='relu'))
        model.add (keras.layers.Dense (256, kernel_initializer='normal', activation='relu'))
        model.add (keras.layers.Dense (256, kernel_initializer='normal', activation='relu'))

    elif modType == 3:
        model.add (keras.layers.Dense (256, kernel_initializer='normal', activation='relu', input_shape=inputShape))
        model.add (keras.layers.Dense (512, kernel_initializer='normal', activation='relu'))
        model.add (keras.layers.Dense (512, kernel_initializer='normal', activation='relu'))
        model.add (keras.layers.Dense (256, kernel_initializer='normal', activation='relu'))

    model.add (keras.layers.Dense (1, kernel_initializer='normal', activation='linear'))
    model.compile (loss='mean_squared_error', optimizer='nadam', metrics=['mean_absolute_error'])

    return model

#
# MakeFileName
#
def MakeOutputFileName (params):
    
    FileName = BASE_OUTPUT_NAME
    FileName += '_' + str(params[0])
    FileName += '_' + params[1]
    FileName += '_' + params[2]

    if params[3] == True:
        FileName += '_LogT'
    else:
        FileName += '_LogF'

    # note LV -> little variation
    if params[4] == True:
        FileName += '_LVT'
    else:
        FileName += '_LVF'

    # LC -> little correlation
    if params[5] == True:
        FileName += '_LCT'
    else:
        FileName += '_LCF'

    # FE -> Feature engineering
    if params[6] == True:
        FileName += '_FET'
    else:
        FileName += '_FEF'

    # FD -> Feature engineering dependencies
    if params[7] == True:
        FileName += '_FDT'
    else:
        FileName += '_FDF'

    # DO -> drop outliers
    if params[8] == True:
        FileName += '_DOT'
    else:
        FileName += '_DOF'

    FileName += '.csv' 

    return FileName


def MakeResultsOutput (score, params, errMsg):

    output = str(score[0]*score[1]) + ',' + str(score[0]) + ',' + str(score[1]) + ','
    for i in range(9):
        output += str (params[i]) + ','
    output += MakeOutputFileName (params) + ','
    output += errMsg + '\n'
   
    return output 


#
# EvaluateModel... this does it all, that is it cleans the data, builds and trains a model, makes a prediction
#    with the test data.
#
def EvaluateModel (params):

    # Get the Train/Test data... depending on the various parameters, 
    XTrain, YTrain, XTestIDs, XTestValues = HousingDataUtils.GetTrainTestData (CatEncoding      = params[1],
                                                                               ScalingType      = params[2],
                                                                               LogSalesPrice    = params[3],
                                                                               DropLilVarFeats  = params[4], 
                                                                               DropLilCorrFeats = params[5],
                                                                               MakeNewFeatures  = params[6], 
                                                                               DropUsedFeatures = params[7],
                                                                               DropOutliers     = params[8])

    XTrain_split, XValid, YTrain_split, YValid = train_test_split (XTrain, YTrain, random_state=42)

    # Make the model...
    print ('\n>>> Building and Fitting Model...')
    model = BuildModel (params[0], XTrain_split.shape[1:])

    callback = tf.keras.callbacks.EarlyStopping (monitor='loss', patience=PATIENCE)
    history = model.fit (XTrain_split, YTrain_split, epochs=MAX_EPOCHS, validation_data=(XValid, YValid), callbacks=[callback])

    # get the model score... or loss and mean square error
    modelScore = model.evaluate (XTrain, YTrain)
    print ('>>> modelScore = ', modelScore, '   ', model.metrics_names)
    print ()

    # Predict the model
    print ('Predicting the model...')
    predYValues = model.predict (XTestValues).tolist()

    # note that we need to modify them from the TF style...
    tmp = []
    for i in range(len(predYValues)):
        tmp.append (predYValues[i][0])
    predYValues = tmp 

    # output...
    outputFile = MakeOutputFileName (params)
    results, errMsg = HousingDataUtils.MergeResult (XTestIDs, predYValues, params[3], YTrain.mean())
    results.to_csv (outputFile, index=False)

    if errMsg != '':
        print ('>>> Errors')
        print ('   ', errMsg)

    return (modelScore, errMsg)



if FIND_OPTIMAL_PARAMS == True:

    paramCombos = list (itertools.product( [ 1, 2, 3 ],                # 0 - Model Type
                                           [ 'Label', 'OneHot' ],      # 1 - Categorical Encoding Type
                                           [ 'Min', 'Std', 'None' ],   # 2 - Scaling Type
                                           [ True, False ],            # 3 - Take log of sales prices
                                           [ True, False ],            # 4 - Drop features with little variation
                                           [ True, False ],            # 5 - Drop features with little correlation to sales price
                                           [ True, False ],            # 6 - Make new features; i.e. feature engineering
                                           [ True, False ],            # 7 - Drop feature engineering dependencies
                                           [ True, False ] ))          # 8 - Drop outliers

    resultsFile = open (RESULTS_FILE, 'a') 
    resultsFile.write ('Score,Loss,MSE,ModelType,CatEncoding,ScalingType,LogSales,DropLilVar,DropLilCorr,FeatEng,DropFEDepends,DropOuts,ResultsFile,Errors\n')

    bestScore  = 9e9
    bestParams = []

    i = 1
    for params in paramCombos:

        print ('**** Iteration: ')
        print (BigNums.IntToNumFont (i))
        print ('of ', len(paramCombos), '\n')

        print (params)

        modelScore, errMsg = EvaluateModel (params)
       
        score = modelScore[0] * modelScore[1]
        if score < bestScore:
            bestScore  = score
            bestParams = params

        resultsFile.write ( MakeResultsOutput(modelScore, params, errMsg) )

        i += 1


    resultsFile.close ()

    print ()
    print ('*** *** *** *** *** *** *** *** ***')
    print ('Best Score: ', bestScore)
    print ('Best Params: ')
    print ('   Model Type: ', bestParams[0])
    print ('   Categorical Encoding Type: ', bestParams[1])
    print ('   Scaling Type: ', bestParams[2])
    print ('   Log of Sales Price: ', bestParams[3])
    print ('   Drop features with little variation: ', bestParams[4])
    print ('   Drop features with little correlation to sales price: ', bestParams[5])
    print ('   Make new features: ', bestParams[6])
    print ('   Drop feature engineering dependencies: ', bestParams[7])
    print ('   Drop outliers: ', bestParams[8])
    print ('-> ', bestParams)
    print ('*** *** *** *** *** *** *** *** ***')
    print ()

else:
    
    # Evaluate the model once using previously found values...
    EvaluateModel ( (3, 'OneHot', 'Min', True, True, True, True, False, True) ) 


