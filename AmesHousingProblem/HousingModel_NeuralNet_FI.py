#
# HousingModel_NeuralNet_FI.py
#
# Used to calculate the feature importance for a regression model that uses neural networks
#


from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import itertools

import HousingDataUtils
import BigNums


MAX_EPOCHS      = 200
PATIENCE        = 3
NET_STYLE       = 2

BASE_OUTPUT_NAME    = 'Results/TestPreds_NN'

MOD_TYPE    = 1
INPUT_SHAPE = (100,)



#
# BuildModelGV... 
#
def BuildModel ():

    tf.random.set_seed (42)
    model = keras.models.Sequential ()

    if MOD_TYPE == 1:
        model.add (keras.layers.Dense (80, kernel_initializer='normal', activation='relu', input_shape=INPUT_SHAPE))
        model.add (keras.layers.Dense (40, kernel_initializer='normal', activation='relu'))
        model.add (keras.layers.Dense (20, kernel_initializer='normal', activation='relu'))

    elif MOD_TYPE == 2:
        model.add (keras.layers.Dense (128, kernel_initializer='normal', activation='relu', input_shape=INPUT_SHAPE))
        model.add (keras.layers.Dense (256, kernel_initializer='normal', activation='relu'))
        model.add (keras.layers.Dense (256, kernel_initializer='normal', activation='relu'))
        model.add (keras.layers.Dense (256, kernel_initializer='normal', activation='relu'))

    elif MOD_TYPE == 3:
        model.add (keras.layers.Dense (256, kernel_initializer='normal', activation='relu', input_shape=INPUT_SHAPE))
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

    global MOD_TYPE
    global INPUT_SHAPE

    # Get the Train/Test data... depending on the various parameters, 
    XTrain, YTrain, XTestIDs, XTestValues = HousingDataUtils.GetTrainTestData (CatEncoding      = params[1],
                                                                               ScalingType      = params[2],
                                                                               LogSalesPrice    = params[3],
                                                                               DropLilVarFeats  = params[4], 
                                                                               DropLilCorrFeats = params[5],
                                                                               MakeNewFeatures  = params[6], 
                                                                               DropUsedFeatures = params[7],
                                                                               DropOutliers     = params[8])

    #XTrain_split, XValid, YTrain_split, YValid = train_test_split (XTrain, YTrain, random_state=42)

    # Make the model...
    print ('\n>>> Building and Fitting Model...')

    MOD_TYPE = params[0]
    INPUT_SHAPE = XTrain.shape[1:]

    cb = tf.keras.callbacks.EarlyStopping (monitor='loss', patience=PATIENCE)
    model = KerasRegressor (build_fn=BuildModel, validation_split=0.2, epochs=MAX_EPOCHS, verbose=2, callbacks=[cb])

    history = model.fit (XTrain, YTrain)

    # get the model score... or loss and mean square error
    modelScore = model.score (XTrain, YTrain)
    print ('>>> modelScore = ', modelScore)
    print ()

    # feature importance...
    print ('>>> Finding feature importance...')
    featImportanceResults = HousingDataUtils.FindFeatureImportance (model, XTrain, YTrain, 3, 100)

    for i in range(len(featImportanceResults)):
        print (f'{featImportanceResults[i][0]:25} {featImportanceResults[i][1]:14.7f} {featImportanceResults[i][2]:14.7f}')
   
 
    # Predict the model
    print ('Predicting the model...')
    predYValues = model.predict (XTestValues).tolist()

    # output...
    outputFile = MakeOutputFileName (params)
    print ('>>>Writing output to...', outputFile)
    results, errMsg = HousingDataUtils.MergeResult (XTestIDs, predYValues, params[3], YTrain.mean())
    results.to_csv (outputFile, index=False)

    if errMsg != '':
        print ('>>> Errors')
        print ('   ', errMsg)

    return (modelScore, errMsg)



    
# Evaluate the model once using previously found values...
EvaluateModel ( (3, 'OneHot', 'Min', True, True, True, True, False, True) ) 


