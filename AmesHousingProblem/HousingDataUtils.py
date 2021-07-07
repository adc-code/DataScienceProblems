#
# HousingDataUtils.py
#
# Various utility functions used by all the various models
#


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance

import seaborn as sns
import matplotlib.pyplot as plt
import shap

import pandas as pd
import numpy as np
import random
import math


TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE  = 'test.csv'


#
# GetTrainTestData: get train and test data... 
#
def GetTrainTestData (CatEncoding = 'Label',     # Category Encoding: Label or OneHot
                      ScalingType = 'Min',       # Scaling Type: Min, Std, or None
                      LogSalesPrice = False,     # Take log of Sales Price: True or False
                      DropLilVarFeats = False,   # Drop features with little varience 
                      FeatVarTol = 0.9,          # Feature Varience Tolerance: 0 to 1
                      DropLilCorrFeats = True,   # Drop features with little correlation to the sales price
                      FeatCorrTol = 0.35,        # Correlation tolerance with features and sales price
                      MakeNewFeatures = True,    # Make new features... that is feature engineering
                      DropUsedFeatures = True,   # Drop features that are used by the new features
                      DropOutliers = True,       # Drop outliers in train set
                      OutliersNumSD = 3.5):      # Standard deviation to determine outliers

    # Get the data...
    print ('Loading data...')
    trainDF, testDF = LoadData ()

    #
    # merge data
    #
    
    # note we will work with copies of the data...
    combinedDF = trainDF.copy ()
    combinedDF.drop (['Id', 'SalePrice'], axis = 1, inplace = True)

    cpyTestDF = testDF.copy ()
    cpyTestDF.drop (['Id'], axis = 1, inplace = True)

    combinedDF = pd.concat ([combinedDF, cpyTestDF], axis=0, sort=False)

    #
    # fix NAs
    #
    print ('Fixing NAs...')    

    print ('List of NAs before...')
    ListNAValues (combinedDF)

    # Repair the NAs
    FixNAs (combinedDF)

    print ('List of NAs after...')
    ListNAValues (combinedDF)
    print ()

    # Find almost constant features/columns.  Note that this needs to be
    # done before categorical feature encoding since one-hot encoding will
    # lead to a lot of false positive results.
    print ('Checking feature variation...')
    featsNoVar = CheckFeatureVariation (combinedDF, FeatVarTol)
    print ()    

    #
    # feature engineering
    #
    newFeatDepends = []
    if MakeNewFeatures == True:
        print ('Creating new features...')
        newFeatDependencies = CreateNewFeatures (combinedDF)
        print ()

    #
    # drop columns
    # 
    # Note that features should be dropped before any type of categorical encoding,
    # scaling, or outlier removal
    if DropLilVarFeats == True:
        print ('Dropping features with little variation...')
        print ('   Features before: ', combinedDF.shape[1])
        combinedDF.drop (featsNoVar, axis=1, inplace=True)
        print ('   Features after: ', combinedDF.shape[1])
        print ()

    if DropLilCorrFeats == True:
        # note that only the training set is used since the test set does not have any prices... yet
        print ('Dropping features with little correlation with sales price...')
        corrMat = trainDF.corr ()
        poorCorrFeatures = corrMat [abs(corrMat['SalePrice']) < FeatCorrTol].transpose().columns.tolist()
        #print (poorCorrFeatures)
        print ('   Features before: ', combinedDF.shape[1])
        for feat in poorCorrFeatures:
            if feat in combinedDF.columns:
                combinedDF.drop ([feat], axis=1, inplace=True)
        print ('   Features after: ', combinedDF.shape[1])
        print ()

    if MakeNewFeatures == True and DropUsedFeatures == True:
        # note only drop such features if new features were created 
        print ('Dropping features used in new features...')
        print ('   Features before: ', combinedDF.shape[1])
        for feat in newFeatDependencies:
            if feat in combinedDF.columns:
                combinedDF.drop ([feat], axis=1, inplace=True)
        print ('   Features after: ', combinedDF.shape[1])
        print ()

    #
    # Encode categorical data... either use one hot or label encoding (or none)
    #
    print ('Categorical Encoding... ', CatEncoding, '\n')
    if CatEncoding == 'Label':
        DoLabelEncoding (combinedDF)
    elif CatEncoding == 'OneHot':
        combinedDF = DoOneHotEncoding (combinedDF)

    #
    # separate data
    #
    XTrain = combinedDF [ :trainDF.shape[0] ]

    YTrain = trainDF['SalePrice']
    # take log of price?
    if LogSalesPrice:
        print ('\nTaking log of YValues\n')
        YTrain = np.log (YTrain)

    XTestIDs    = testDF['Id']
    XTestValues = combinedDF [ trainDF.shape[0]: ] 

    #
    # get rid of outliers
    #
    if DropOutliers == True:
        print ('Removing some outliers from train set...')
        print ('   Rows before: ', XTrain.shape[0], YTrain.shape[0])
        outlierFeats = ['LotFrontage', 'LotArea', 'GrLivArea', 'GarageArea', 'SalePrice']
        XTrain, YTrain =  RemoveOutliers (XTrain, YTrain, outlierFeats, OutliersNumSD)
        print ('   Rows after: ', XTrain.shape[0], YTrain.shape[0])
        print ()

    #
    # Scale data
    #
    scaler = None
    if ScalingType == 'Min':
        scaler = MinMaxScaler ()
    elif ScalingType == 'Std':
        scaler = StandardScaler ()

    if scaler is not None:
        print ('Scaling data... using ', ScalingType)
        scaler.fit (combinedDF)

        # note that there are a few extra steps being done here so that
        # 1) values are returned as a dataframe (not an ndarray), and 
        # 2) no warnings are displayed since the code wasn't particularly 'pythonic'
        colNames = combinedDF.columns.tolist()

        scaledXTrain      = XTrain.copy()
        scaledXTestValues = XTestValues.copy()
        
        scaledXTrainFeats      = scaledXTrain[colNames]
        scaledXTestValuesFeats = scaledXTestValues[colNames]

        scaledXTrainFeats      = scaler.transform (scaledXTrainFeats.values)
        scaledXTestValuesFeats = scaler.transform (scaledXTestValuesFeats.values)

        XTrain[colNames]      = scaledXTrainFeats
        XTestValues = XTestValues.copy()
        XTestValues[colNames] = scaledXTestValuesFeats
  
    #
    # report some details...
    #
    print ('\nData info...')
    print (f'  Original trainDF.shape = {trainDF.shape}')
    print (f'  Original testDF.shape = {testDF.shape}')
    print (f'  XTrain.shape = {XTrain.shape}')
    print (f'  YTrain.shape = {YTrain.shape}')
    print (f'  XTestIDs.shape = {XTestIDs.shape}')
    print (f'  XTestValues.shape = {XTestValues.shape}')

    return XTrain, YTrain, XTestIDs, XTestValues


#
# LoadData... return the train and test datasets as dataframes
#
def LoadData ():
    
    trainDF = pd.read_csv (TRAIN_DATA_FILE)
    testDF  = pd.read_csv (TEST_DATA_FILE)
    
    return trainDF, testDF


#
# GetCatNumColumns... returns 2 lists of the categorical and numeric columns/features
#
def GetCatNumColumns (DF):

    categoricalCols = []
    numericalCols   = []

    for col in DF.columns:
        if DF.dtypes [col] == 'object':
            categoricalCols.append (col)
        else:
            numericalCols.append (col)
                
    return (categoricalCols, numericalCols)


#
# ListNAValues... list the number of NA values for each column/feature
#
def ListNAValues (DF):

    categoricalCols, numericalCols = GetCatNumColumns (DF)

    total = 0
    for col in DF.columns:
        countNAs = DF[col].isna().sum()
        total += countNAs
        if countNAs > 0:
            colType = 'Categorical'
            if col in numericalCols:
                colType = 'Numerical'
            print (f'  {col:13} --> {countNAs:5}   {colType}')

    print (f'  {"Total":13} --> {total:5}')


#
# FixNAs
#
def FixNAs (DF):
   
    # MSZoning: Identifies the general zoning classification of the sale.
    # -> Use the most common element... RL for Residential Low Density
    DF['MSZoning'].fillna ( GetTopItemForFeature (DF, 'MSZoning'), inplace=True)

    # LotFrontage: Linear feet of street connected to property
    # -> Take the median since the distribution is a bit skewed
    DF['LotFrontage'].fillna (DF['LotFrontage'].median(), inplace=True)

    # Alley: Type of alley access to property
    # -> NA refers to No alley access.... so NA becomes None
    DF['Alley'].fillna ('None', inplace=True)
    
    # Utilities: Type of utilities available
    # -> Use the most common element... All public Utilities (E,G,W,& S)
    # -> Can be dropped since there is little varience in the values
    DF['Utilities'].fillna ( GetTopItemForFeature (DF, 'Utilities'), inplace=True)

    # Exterior1st & Exterior2nd: Exterior covering on house... first and second most used
    # -> Use the most common element for each... Vinyl Siding
    DF['Exterior1st'].fillna ( GetTopItemForFeature (DF, 'Exterior1st'), inplace=True)
    DF['Exterior2nd'].fillna ( GetTopItemForFeature (DF, 'Exterior2nd'), inplace=True)

    # MasVnrType: Masonry veneer type  
    # -> Use the most common value... none
    # MasVnrArea: Masonry veneer area
    # -> Use the median... which is 0
    DF['MasVnrType'].fillna ( GetTopItemForFeature (DF, 'MasVnrType'), inplace=True)
    DF['MasVnrArea'].fillna ( DF['MasVnrArea'].median(), inplace=True)
     
    # BsmtQual: Evaluates the height of the basement
    # -> Use the most common value... TA Typical
    DF['BsmtQual'].fillna ( GetTopItemForFeature (DF, 'BsmtQual'), inplace=True)

    # BsmtCond: Evaluates the general condition of the basement
    # -> Use the most common value... TA Typical - slight dampness allowed
    DF['BsmtCond'].fillna ( GetTopItemForFeature (DF, 'BsmtCond'), inplace=True)

    # BsmtExposure: Refers to walkout or garden level walls
    # -> Use the most common value... No - No Exposure
    DF['BsmtExposure'].fillna ( GetTopItemForFeature (DF, 'BsmtExposure'), inplace=True)
    
    # BsmtFinType1: Rating of basement finished area
    # -> Since this feature does not have a big winner, fill in items randomly based on
    #    their frequency
    MultiFillNAs (DF, 'BsmtFinType1')
    
    # BsmtFinSF1: Type 1 finished square feet
    # -> The value of BsmtFinSF1 depends on the value of BsmtFinType1.  Hence fill in
    #    any NAs with appropriate values
    GetRelaventValue (DF, 'BsmtFinType1', 'BsmtFinSF1')
    
    # BsmtFinType2: Rating of basement finished area (if multiple types)
    # -> The vast majority of values are of one type, then take the most frequent
    DF['BsmtFinType2'].fillna ( GetTopItemForFeature (DF, 'BsmtFinType2'), inplace=True)
    
    # BsmtFinSF2: Type 2 finished square feet
    # -> Again use the parent category to fill in the BsmtFinSF2... it is more than likely 0
    GetRelaventValue (DF, 'BsmtFinType2', 'BsmtFinSF2')
 
    # TotalBsmtSF: Total square feet of basement area
    # -> The values are not quite normally distributed... anyways we shall take the median
    DF['TotalBsmtSF'].fillna ( DF['TotalBsmtSF'].median(), inplace=True)

    # BsmtUnfSF: Unfinished square feet of basement area 
    # -> Subtract the total finished amount from the overall total; and take the average
    DF['BsmtUnfSF'].fillna ( DF['TotalBsmtSF'].mean() - DF['BsmtFinSF1'].mean() - DF['BsmtFinSF2'].mean(),
                             inplace=True)

    # Electrical: Electrical system
    # -> Take the value that occurs the most frequently... SBrkr
    DF['Electrical'].fillna ( GetTopItemForFeature (DF, 'Electrical'), inplace=True)

    # BsmtFullBath: Basement full bathrooms
    # BsmtHalfBath: Basement half bathrooms
    # -> based on the number of bathrooms per finished square footage, estimate the full and half baths
    EstNumBsmtBaths (DF, 'Full')
    EstNumBsmtBaths (DF, 'Half')

    # KitchenQual: Kitchen quality
    # -> Use the most frequent value... 
    DF['KitchenQual'].fillna ( GetTopItemForFeature (DF, 'KitchenQual'), inplace=True)

    # Functional: Home functionality (Assume typical unless deductions are warranted)
    # -> Told to assume typical... hence replace NAs with Typ
    DF['Functional'].fillna ('Typ', inplace=True)

    # FireplaceQu: Fireplace quality
    # -> NA refers to No fireplace... so NA becomes None
    DF['FireplaceQu'].fillna ('None', inplace=True)

    # GarageType: Garage location
    # -> NA refers to No garage... so NA becomes None
    DF['GarageType'].fillna ('None', inplace=True)

    # GarageYrBlt: Year garage was built
    # -> Use the most frequent value... that is the mode
    DF['GarageYrBlt'].fillna ( DF['GarageYrBlt'].mode()[0], inplace=True)

    # GarageFinish: Interior finish of the garage
    # -> NA refers to No garage... so NA becomes None
    DF['GarageFinish'].fillna ('None', inplace=True)

    # GarageCars: Size of garage in car capacity
    # -> Use the most frequent value... 
    DF['GarageCars'].fillna ( GetTopItemForFeature (DF, 'GarageCars'), inplace=True)

    # GarageArea: Size of garage in square feet
    # -> Fill in the the average depending on the number of cars
    FixGarageArea (DF)

    # GarageQual: Garage quality
    # -> NA refers to No garage... so NA becomes None
    DF['GarageQual'].fillna ('None', inplace=True)

    # GarageCond: Garage condition
    # -> NA refers to No garage... so NA becomes None
    DF['GarageCond'].fillna ('None', inplace=True)

    # PoolQC: Pool quality
    # -> NA refers to No pool... so NA becomes None
    DF['PoolQC'].fillna ('None', inplace=True)

    # Fence: Fence quality
    # -> NA refers to No fence... so NA becomes None
    DF['Fence'].fillna ('None', inplace=True)

    # MiscFeature: Miscellaneous feature not covered in other categories
    # -> NA refers to None... 
    DF['MiscFeature'].fillna ('None', inplace=True)

    # Type of sale 
    # -> Use the most frequent value... WD
    DF['SaleType'].fillna ( GetTopItemForFeature (DF, 'SaleType'), inplace=True)
    

#
# GetTopItemForFeature: 
# 
def GetTopItemForFeature (DF, Feature):

    # make the values_counts output into a dictionary, then take the first element (since
    # value counts sorts the values in a decending manor)
    return next (iter (DF[Feature].value_counts().to_dict()))


#
# MultiFillNAs: instead of filling in NA values with the mean/median/zero, use existing
#    values based on how frequently they occur.  For ex. if two values exist in a 50-50
#    ratio, and 10 values are missing, randomly pick one of each value for the NAs
#
def MultiFillNAs (DF, feature):
    
    # first get the totals for each category
    amts = DF[feature].value_counts().to_dict()

    # next find the total... note that this should be the length - the amount of NAs
    total = 0
    for key in amts:
        total += amts[key]

    # make a list to pick from.  The amount of entries is based on the frequency of each
    # category
    fillInTypes = []
    for key in amts:
        multipleAmnt = int (1000 * amts[key] / total)
        fillInTypes.extend ( [key] * multipleAmnt )
   
    # randomly shuffle the list... and yet we shall still randomly pick from it 
    random.shuffle(fillInTypes)
   
    # get all the NAs 
    NAs = DF[feature].isna ()

    # for each NA value, fill it in with something randomly choosen from the list
    for i in range (len(NAs)):
        if NAs.iloc[i] == True:
            index = math.floor (len(fillInTypes) * random.random())
            DF.iloc[i, DF.columns.get_loc(feature)] = fillInTypes[index]
            #DF[feature].iloc[i] = fillInTypes[index]


#
# GetRelaventValue: 
#
def GetRelaventValue (DF, classFeature, Feature):

    # get the list of keys from the feature class
    keys = list (DF[classFeature].value_counts().to_dict().keys())

    # for each key in the feature class, get the associated average value
    meanValues = {}
    for key in keys:
        selected = DF[classFeature] == key
        total    = (selected * DF[Feature]).sum()
        count    = sum (selected)
    
        meanValues [key] = total / count

    # get the NAs with the feature column
    NAs = DF[Feature].isna()

    for i in range(len(NAs)):
        if NAs.iloc[i] == True:
            #DF[Feature].iloc[i] = meanValues [DF[classFeature].iloc[i]]
            DF.iloc[i, DF.columns.get_loc(Feature)] = meanValues [DF.iloc[i, DF.columns.get_loc(classFeature)]]


#
# EstNumBsmtBaths: use the total finished basement square footage to guess the number
#    of bathrooms should be.
#
def EstNumBsmtBaths (DF, bathType):

    FeatureName = 'BsmtFullBath'
    if bathType == 'Half':
        FeatureName = 'BsmtHalfBath'

    # find the average finished basement square footage for each bathroom
    BsmtSFPerBath = {}
    for i in range(4):
        BsmtSFPerBath[i] = []

    # group the total number of finished square feet per bathroom
    for i in range (DF.shape[0]):
        BsmtFinSFTotal = DF['BsmtFinSF1'].iloc[i] + DF['BsmtFinSF2'].iloc[i]

        if DF['BsmtFullBath'].iloc[i] == 0.0:
            BsmtSFPerBath[0].append (BsmtFinSFTotal)
        elif DF['BsmtFullBath'].iloc[i] == 1.0:
            BsmtSFPerBath[1].append (BsmtFinSFTotal)
        elif DF['BsmtFullBath'].iloc[i] == 2.0:
            BsmtSFPerBath[2].append (BsmtFinSFTotal)
        elif DF['BsmtFullBath'].iloc[i] == 3.0:
            BsmtSFPerBath[3].append (BsmtFinSFTotal)
        
    # then calculate the average for each grouping
    for i in range(4):
        BsmtSFPerBath[i] = sum (BsmtSFPerBath[i]) / len (BsmtSFPerBath[i])
    
    # now go through the NAs and looking at the finished area, guess a number for the bathrooms 
    NAs = DF[FeatureName].isna()

    for i in range (len(NAs)):
        if NAs.iloc[i] == True:
            # now use the total area to estimate the number of bathrooms
            totalArea = DF['BsmtFinSF1'].iloc[i] + DF['BsmtFinSF2'].iloc[i]

            # find the closest element
            closestDiff = 9999
            closestIndex = 0
            for j in range(4):
                diff = math.fabs (BsmtSFPerBath[j] - totalArea)
                if diff < closestDiff:
                    closestDiff  = diff
                    closestIndex = j
        
            #DF[FeatureName].iloc[i] = float(closestIndex)
            DF.iloc[i, DF.columns.get_loc(FeatureName)] = float(closestIndex)


#
# FixGarageArea: this function should be a bit more generic... anyways it is used to fill in
#    any NAs with the garage area feature using the average garage area and the number of cars
#    that are present
#
def FixGarageArea (DF):

    GarageSFPerCar = {}
    for i in range (6):
        GarageSFPerCar[i] = []
    
    # convert from float to integer    
    DF['GarageCars'] = DF['GarageCars'].astype('int32')

    for i in range (DF.shape[0]):
        numOfCars  = DF['GarageCars'].iloc[i]
        garageArea = DF['GarageArea'].iloc[i]
        if not math.isnan (garageArea):
            GarageSFPerCar[numOfCars].append (garageArea)

    for i in range(6):
        GarageSFPerCar[i] = sum (GarageSFPerCar[i]) / len (GarageSFPerCar[i])
    
    # print (GarageSFPerCar)

    NAs = DF['GarageArea'].isna()

    for i in range(len(NAs)):
        if NAs.iloc[i] == True:
            #print (combinedDF['GarageCars'].iloc[i])
            #DF['GarageArea'].iloc[i] = GarageSFPerCar[DF['GarageCars'].iloc[i]]
            DF.iloc[i, DF.columns.get_loc('GarageArea')] = GarageSFPerCar[DF.iloc[i, DF.columns.get_loc('GarageCars')]]


#
# CheckFeatureVariation: used to find possible features that have very little variation and hence don't 
#    provide much information to differentiate various items
#
def CheckFeatureVariation (DF, FeatVarTol):

    featList = []

    for col in DF.columns:

        counts = DF[col].value_counts().tolist()
        total  = sum (counts)

        for i in range(len(counts)):
            frac = counts[i] / total
            if frac > FeatVarTol:
                print (f'  {col:15}  {frac:0.8f}')
                featList.append (col)

    return featList 


#
# DoLabelEncoding: encode the categorical columns using label encoding; that is replace categorical
#    names with numbers... 0 to n-1
#
def DoLabelEncoding (DF):
   
    categoricalCols, numericalCols = GetCatNumColumns (DF)
 
    for col in categoricalCols:
        DF[col] = DF[col].astype ('category')
        DF[col] = DF[col].cat.codes 


#
# DoOneHotEncoding: perform one-hot encoding on the categorical based features
#
def DoOneHotEncoding (DF):

    categoricalCols, numericalCols = GetCatNumColumns (DF)

    for col in categoricalCols:
        DF = pd.get_dummies (DF, columns=[col], prefix=[col])

    return DF


#
# CreateNewFeatures: perform some feature engineering, that is make new features from
#    existing features.  Note that the dependencies of the new features are returned.
#
def CreateNewFeatures (DF):

    DF['SqFtPerRoom'] = DF['GrLivArea'] / ( DF['TotRmsAbvGrd'] +
                                            DF['FullBath'] +
                                            DF['HalfBath'] +
                                            DF['KitchenAbvGr'] )

    DF['OverallCondQual'] = DF['OverallQual'] * DF['OverallCond']

    DF['TotalBaths'] = ( DF['FullBath'] + 
                         DF['BsmtFullBath'] +
                         (0.5 * DF['HalfBath']) +
                         (0.5 * DF['BsmtHalfBath']) )

    DF['AboveGradeSF'] = DF['1stFlrSF'] + DF['2ndFlrSF']

    return [ 'GrLivArea', 'TotRmsAbvGrd', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'OverallQual', 
             'OverallCond', 'FullBath', 'BsmtFullBath', 'HalfBath', 'BsmtHalfBath' ]


#
# RemoveOutliers... remove rows that are a certain number of standard deviations above
#    the mean for a particular feature.  Note that only values above the mean are considered,
#    those below the mean are not considered.
#
def RemoveOutliers (DF, yValues, columns, NumSDs):

    tmpDF = DF.copy()
    tmpDF.append (yValues)
    tmpDF['SalePrice'] = yValues

    categoricalCols, numericalCols = GetCatNumColumns (tmpDF)

    for col in numericalCols:
        if col in columns:
            mean   = tmpDF[col].mean()
            stdDev = tmpDF[col].std()

            tmpDF.drop (tmpDF[tmpDF[col] > (mean + NumSDs*stdDev)].index, inplace=True)

    newYValues = tmpDF['SalePrice']
    tmpDF.drop (['SalePrice'], axis=1, inplace=True)

    return tmpDF, newYValues


#
# MergeResults... make the results ready for output, that is add the Ids and take the
#    exponent if necessary
#
def MergeResult (IDs, YValues, takeExp, defaultValue):

    errMsg = ''

    if takeExp:

        # Go through all the values to check if any negative or large values are present.
        # If so, use the default value and report an error message
        for i in range(len(YValues)):
            if YValues[i] < 0:
                # reminder... we can't take logs of negative numbers
                errMsg += 'Negative value (' + str(YValues[i]) + ');'
                YValues[i] = defaultValue
            elif YValues[i] > np.log(np.finfo('d').max):
                # note... the above line checks if the value is above a maximum value, 
                # which will produce overflow errors
                errMsg += 'Overflow error (' + str(YValues[i]) + ');'
                YValues[i] = defaultValue

        YValues = np.exp (YValues)

    return (pd.DataFrame ({'Id':IDs,'SalePrice':YValues}), errMsg)


#
# FindFeatureImportance... using the permutation importance, the overall importance of all the features 
#    is determined.  The two reasons why 'permutation importance' is used is because: 1) it can be used
#    with all of the various models that were used; and 2) it can gauge both categorical and numeric
#    features.  
#
def FindFeatureImportance (model, XData, YData, repetitions=5, scaler=1):

    # first find the importance results...
    importanceResults = permutation_importance (model, XData, YData, n_repeats=repetitions, random_state=0)

    # and then clean them up a bit
    overallResults = []

    # note that for some models, the mean values sum to one.  So if there are a lot of features present,
    # then each value can be rather small.  The scaler value is used to make dealing with the values a little
    # less painfull

    for i in range(len(XData.columns)):
        overallResults.append ( [ XData.columns[i],                                # Feature
                                  importanceResults.importances_mean[i] * scaler,  # Mean
                                  importanceResults.importances_std[i]   ] )       # Standard deviation  

    return overallResults


