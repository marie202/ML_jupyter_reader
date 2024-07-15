# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:59:14 2020

@author: Andreas Wunsch
"""
#reproducability
from numpy.random import seed
seed(1+347823)
import tensorflow as tf
tf.random.set_seed(1+63493)

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
# from bayes_opt.util import load_logs #needed if logs are already available
import os
import pandas as pd
import datetime
from scipy import stats
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from uncertainties import unumpy


gpus = tf.config.experimental.list_physical_devices('GPU')

# =============================================================================
#### Functions 
# =============================================================================

def load_GW_and_HYRAS_Data(i):
    #define where to find the data
    pathGW = "Data/GWData"
    pathHYRAS = "Data/HYRAS"
    pathconnect = "/"
    
    #load a list of all sites
    well_list = pd.read_csv("./list.txt")
    Well_ID = well_list.ID[i]
    
    #load and merge the data
    GWData = pd.read_csv(pathGW+pathconnect+Well_ID+'_GW-Data.csv', 
                          parse_dates=['Date'],index_col=0, dayfirst = True, 
                          decimal = '.', sep=',')
    HYRASData = pd.read_csv(pathHYRAS+pathconnect+Well_ID+'_weeklyData_HYRAS.csv',
                            parse_dates=['Date'],index_col=0, dayfirst = True,
                            decimal = '.', sep=',')
    data = pd.merge(GWData, HYRASData, how='inner', left_index = True, right_index = True)
    
    return data, Well_ID

def split_data(data, GLOBAL_SETTINGS):
    #split the test data from the rest
    dataset = data[(data.index < GLOBAL_SETTINGS["test_start"])] #Testdaten abtrennen
    
    #split remaining time series into three parts 80%-10%-10%
    TrainingData = dataset[0:round(0.8 * len(dataset))]
    StopData = dataset[round(0.8 * len(dataset))+1:round(0.9 * len(dataset))]
    StopData_ext = dataset[round(0.8 * len(dataset))+1-GLOBAL_SETTINGS["seq_length"]:round(0.9 * len(dataset))] #extend data according to dealys/sequence length
    OptData = dataset[round(0.9 * len(dataset))+1:]
    OptData_ext = dataset[round(0.9 * len(dataset))+1-GLOBAL_SETTINGS["seq_length"]:] #extend data according to dealys/sequence length
    
    TestData = data[(data.index >= GLOBAL_SETTINGS["test_start"]) & (data.index <= GLOBAL_SETTINGS["test_end"])]
    TestData_ext = pd.concat([dataset.iloc[-GLOBAL_SETTINGS["seq_length"]:], TestData], axis=0) # extend Testdata to be able to fill sequence later                                              

    return TrainingData, StopData, StopData_ext, OptData, OptData_ext, TestData, TestData_ext

def to_supervised(data, GLOBAL_SETTINGS):
    #make the data sequential
    #modified after Jason Brownlee and machinelearningmastery.com
    
    X, Y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + GLOBAL_SETTINGS["seq_length"]
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_idx, 1:], data[end_idx, 0]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)

class MCDropout(tf.keras.layers.Dropout):
    #define Monte Carlo Dropout Layer, where training state is always true (even during prediction)
    def call(self, inputs):
        return super().call(inputs, training=True)
    
def predict_distribution(X, model, n):
    preds = [model(X) for _ in range(n)]
    return np.hstack(preds)

def gwmodel(ini,GLOBAL_SETTINGS,X_train, Y_train,X_stop, Y_stop):
    # define model
    seed(ini+872527)
    tf.random.set_seed(ini+87747)

    inp = tf.keras.Input(shape=(GLOBAL_SETTINGS["seq_length"], X_train.shape[2]))
    cnn = tf.keras.layers.Conv1D(filters=GLOBAL_SETTINGS["filters"],
                                         kernel_size=GLOBAL_SETTINGS["kernel_size"],
                                         activation='relu',
                                         padding='same')(inp)
    cnn = tf.keras.layers.MaxPool1D(padding='same')(cnn)
    cnn = MCDropout(0.5)(cnn)
    cnn = tf.keras.layers.Flatten()(cnn)
    cnn = tf.keras.layers.Dense(GLOBAL_SETTINGS["dense_size"], activation='relu')(cnn)
    output1 = tf.keras.layers.Dense(1, activation='linear')(cnn)
    
    # tie together
    model = tf.keras.Model(inputs=inp, outputs=output1)
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=GLOBAL_SETTINGS["learning_rate"],
                                         epsilon=10E-3, clipnorm=GLOBAL_SETTINGS["clip_norm"])
    
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    
    
    # early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', 
                                          verbose=0, patience=15,restore_best_weights = True)
    
    # fit network
    history = model.fit(X_train, Y_train, validation_data=(X_stop, Y_stop), 
                        epochs=GLOBAL_SETTINGS["epochs"], verbose=0,
                        batch_size=GLOBAL_SETTINGS["batch_size"], callbacks=[es])
    
    return model, history


def bayesOpt_function(pp,densesize, seqlength, batchsize, filters):
    #basically means conversion to rectangular function
    densesize_int = int(densesize)
    seqlength_int = int(seqlength)
    batchsize_int = int(batchsize)
    filters_int = int(filters)
    
    pp = int(pp)
    
    return bayesOpt_function_with_discrete_params(pp, densesize_int, seqlength_int, batchsize_int, filters_int)

def bayesOpt_function_with_discrete_params(pp,densesize_int, seqlength_int, batchsize_int, filters_int):
    
    assert type(densesize_int) == int
    assert type(seqlength_int) == int
    assert type(batchsize_int) == int
    assert type(filters_int) == int
    #[...]
    
    # fixed settings for all experiments
    GLOBAL_SETTINGS = {
        'pp': pp,
        'batch_size': batchsize_int, #16-128
        'kernel_size': 3, #ungerade!
        'dense_size': densesize_int, 
        'filters': filters_int, 
        'seq_length': seqlength_int,
        'clip_norm': True,
        'clip_value': 1,
        'epochs': 100,
        'learning_rate': 1e-3,
        'test_start': pd.to_datetime('02012012', format='%d%m%Y'),
        'test_end': pd.to_datetime('28122015', format='%d%m%Y')
    }
    
    ## load data
    data, Well_ID = load_GW_and_HYRAS_Data(GLOBAL_SETTINGS["pp"])
    
    #modify test period if data ends earlier
    if GLOBAL_SETTINGS["test_end"] > data.index[-1]:
        GLOBAL_SETTINGS["test_end"] = data.index[-1]
        GLOBAL_SETTINGS["test_start"] = GLOBAL_SETTINGS["test_end"] - datetime.timedelta(days=(365*4))
    
    #scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl.fit(pd.DataFrame(data['GWL']))
    data_n = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

    #split data
    TrainingData, StopData, StopData_ext, OptData, OptData_ext, TestData, TestData_ext = split_data(data, GLOBAL_SETTINGS)
    TrainingData_n, StopData_n, StopData_ext_n, OptData_n, OptData_ext_n, TestData_n, TestData_ext_n = split_data(data_n, GLOBAL_SETTINGS)
    
    #sequence data
    X_train, Y_train = to_supervised(TrainingData_n.values, GLOBAL_SETTINGS)
    X_stop, Y_stop = to_supervised(StopData_ext_n.values, GLOBAL_SETTINGS) 
    X_opt, Y_opt = to_supervised(OptData_ext_n.values, GLOBAL_SETTINGS)
    X_test, Y_test = to_supervised(TestData_ext_n.values, GLOBAL_SETTINGS) 

    #build and train model with idifferent initializations
    os.chdir(basedir)
    inimax = 3
    optresults_members = np.zeros((len(X_opt), inimax))
    for ini in range(inimax):
        print("(pp:{}) BayesOpt-Iteration {} - ini-Ensemblemember {}".format(pp,len(optimizer.res)+1, ini+1))
        
        model,history = gwmodel(ini,GLOBAL_SETTINGS,X_train, Y_train, X_stop, Y_stop)  
        opt_sim_n = model.predict(X_opt)
        opt_sim = scaler_gwl.inverse_transform(opt_sim_n)
        optresults_members[:, ini] = opt_sim.reshape(-1,)
    
    opt_sim_median = np.median(optresults_members,axis = 1)    
    sim = np.asarray(opt_sim_median.reshape(-1,1))
    obs = np.asarray(scaler_gwl.inverse_transform(Y_opt.reshape(-1,1)))
    err = sim-obs
    meanTrainingGWL = np.mean(np.asarray(TrainingData['GWL']))
    meanStopGWL = np.mean(np.asarray(StopData['GWL']))
    err_nash = obs - np.mean([meanTrainingGWL, meanStopGWL])
    r = stats.linregress(sim[:,0], obs[:,0])
    
    
    print("total elapsed time = {}".format(datetime.datetime.now()-time1))
    print("(pp = {}) elapsed time = {}".format(pp,datetime.datetime.now()-time_single))

    return (1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2)))) + r.rvalue ** 2 #NSE+R²: (max = 2)

def simulate_testset(pp,densesize_int, seqlength_int, batchsize_int, filters_int):
    
    # fixed settings for all experiments
    GLOBAL_SETTINGS = {
        'pp': pp,
        'batch_size': batchsize_int, #16-128
        'kernel_size': 3, #ungerade!
        'dense_size': densesize_int, 
        'filters': filters_int, 
        'seq_length': seqlength_int,
        'clip_norm': True,
        'clip_value': 1,
        'epochs': 100,
        'learning_rate': 1e-3,
        'test_start': pd.to_datetime('02012012', format='%d%m%Y'),
        'test_end': pd.to_datetime('28122015', format='%d%m%Y')
    }
    
    ## load data
    data, Well_ID = load_GW_and_HYRAS_Data(GLOBAL_SETTINGS["pp"])
    
    #modify test period if data ends earlier
    if GLOBAL_SETTINGS["test_end"] > data.index[-1]:
        GLOBAL_SETTINGS["test_end"] = data.index[-1]
        GLOBAL_SETTINGS["test_start"] = GLOBAL_SETTINGS["test_end"] - datetime.timedelta(days=(365*4))
        
    #scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl.fit(pd.DataFrame(data['GWL']))
    data_n = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

    #split data
    TrainingData, StopData, StopData_ext, OptData, OptData_ext, TestData, TestData_ext = split_data(data, GLOBAL_SETTINGS)
    TrainingData_n, StopData_n, StopData_ext_n, OptData_n, OptData_ext_n, TestData_n, TestData_ext_n = split_data(data_n, GLOBAL_SETTINGS)
    
    #sequence data
    X_train, Y_train = to_supervised(TrainingData_n.values, GLOBAL_SETTINGS)
    X_stop, Y_stop = to_supervised(StopData_ext_n.values, GLOBAL_SETTINGS) 
    X_opt, Y_opt = to_supervised(OptData_ext_n.values, GLOBAL_SETTINGS)
    X_test, Y_test = to_supervised(TestData_ext_n.values, GLOBAL_SETTINGS) 

    #build and train model with different initializations
    inimax = 10
    sim_members = np.zeros((len(X_test), inimax))
    sim_members[:] = np.nan
    
    sim_std = np.zeros((len(X_test), inimax))
    sim_std[:] = np.nan
    f = open('./traininghistory_CNN_'+Well_ID+'.txt', "w")
    for ini in range(inimax):
        model,history = gwmodel(ini,GLOBAL_SETTINGS,X_train, Y_train, X_stop, Y_stop)  
        
        loss = np.zeros((1, 100))
        loss[:,:] = np.nan
        loss[0,0:np.shape(history.history['loss'])[0]] = history.history['loss']
        val_loss = np.zeros((1, 100))
        val_loss[:,:] = np.nan
        val_loss[0,0:np.shape(history.history['val_loss'])[0]] = history.history['val_loss']
        print('loss', file = f)
        print(loss.tolist(), file = f)
        print('val_loss', file = f)
        print(val_loss.tolist(), file = f)
        
        #make prediction 100 times for each ini
        y_pred_distribution = predict_distribution(X_test, model, 100)
        sim = scaler_gwl.inverse_transform(y_pred_distribution)
        sim_members[:, ini], sim_std[:, ini]= sim.mean(axis=1), sim.std(axis=1)

    f.close()
    sim_members_uncertainty = unumpy.uarray(sim_members,1.96*sim_std) #1.96 because of sigma rule for 95% confidence
    
    sim_mean = np.nanmedian(sim_members,axis = 1)
    
    sim_mean_uncertainty = np.sum(sim_members_uncertainty,axis = 1)/inimax
    
   
    # get scores
    sim = np.asarray(sim_mean.reshape(-1,1))
    obs = np.asarray(scaler_gwl.inverse_transform(Y_test.reshape(-1,1)))

    err = sim-obs
    err_rel = (sim-obs)/(np.max(data['GWL'])-np.min(data['GWL']))
    err_nash = obs - np.mean(np.asarray(data['GWL'][(data.index < GLOBAL_SETTINGS["test_start"])]))

    NSE = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2)))
    r = stats.linregress(sim[:,0], obs[:,0])
    R2 = r.rvalue ** 2
    RMSE =  np.sqrt(np.mean(err ** 2))
    rRMSE = np.sqrt(np.mean(err_rel ** 2)) * 100
    Bias = np.mean(err)
    rBias = np.mean(err_rel) * 100

    scores = pd.DataFrame(np.array([[NSE, R2, RMSE, rRMSE, Bias, rBias]]),
                   columns=['NSE','R2','RMSE','rRMSE','Bias','rBias'])
    print(scores)
    
    sim1=sim
    obs1=obs

    errors = np.zeros((inimax,6))
    errors[:] = np.nan
    for i in range(inimax):
        sim = np.asarray(sim_members[:,i].reshape(-1,1))
        err = sim-obs
        err_rel = (sim-obs)/(np.max(data['GWL'])-np.min(data['GWL']))
        errors[i,0] = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2)))
        r = stats.linregress(sim[:,0], obs[:,0])
        errors[i,1] = r.rvalue ** 2
        errors[i,2] =  np.sqrt(np.mean(err ** 2))
        errors[i,3] = np.sqrt(np.mean(err_rel ** 2)) * 100
        errors[i,4] = np.mean(err)
        errors[i,5] = np.mean(err_rel) * 100
    

    return scores, TestData, sim1, obs1, inimax, sim_members, Well_ID, errors, sim_members_uncertainty, sim_mean_uncertainty

class newJSONLogger(JSONLogger) :
      def __init__(self, path):
            self._path=None
            super(JSONLogger, self).__init__()
            self._path = path if path[-5:] == ".json" else path + ".json"
            
