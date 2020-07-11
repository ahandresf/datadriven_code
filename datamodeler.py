import math
import numpy as np
import time
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import save_model
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
from env_data_modeler import env_nn_modeler
from env_data_modeler import env_gb_modeler
from env_data_modeler import env_lstm_modeler
from env_data_modeler import env_poly_modeler
from env_data_modeler import create_nn_model_wrapper
from env_data_modeler import create_lstm_model_wrapper
import h5py
import argparse
import pickle
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from conf_params_datadriven import STATE_SPACE_DIM, ACTION_SPACE_DIM, MARKOVIAN_ORDER, POLYNOMIAL_DEGREE, DROPOUT_RATE
from conf_params_datadriven import INPUT_DATASET, OUTPUT_DATASET, ACTION_NAMES,STATE_NAMES
from conf_params_datadriven import default_nn_config
from conf_params_datadriven import EPOCHS
import matplotlib.pyplot as plt
import os

#create a time stamp, Month_Day_HourMinuteSeconds
time_stamp=time.strftime('%B_%d_%H%M%S')
#create directory
os.makedirs('./models/'+time_stamp)
#creating pathnames for saving output
x_scalerdir=('./models/%s/scaler_x_set.pkl'%(time_stamp))
y_scalerdir=('./models/%s/scaler_y_set.pkl'%(time_stamp))


parser = argparse.ArgumentParser()
parser.add_argument("--use-gb", type=bool, default=False,help="choose gradient boosting as a model")
parser.add_argument("--grid-search", type=bool, default=False, help="use grid search to tune hyperparameter tuning, if applicable")
parser.add_argument("--genetic-searcg", type=bool, default=False, help="use genetic algorithn to tune hyperparameter if applicable")
parser.add_argument("--use-lstm", type=bool, default=False, help="using lstm model after hyperparameter tuning")
parser.add_argument("--use-nn", type=bool, default=False, help="choose multilayer perceptron as a model")
parser.add_argument("--use-poly", type=bool, default=False, help="choose polynominal fitting as a model")
#parser.add_argument("tune-ga", type=bool, default=False, help="uses genetic algorithm and TPOT library for hyperparameter tuning")
parser.add_argument("--tune-rs", type=bool, default=False, help="uses random search from scikitlearn for hyperparameter tuning")

def read_env_data():
    try:
        print('load action and state names')
        state_names = np.load(STATE_NAMES, allow_pickle=True)
        action_names = np.load(ACTION_NAMES, allow_pickle=True)
    except:
        print('action_names.npy and state_names.npy arrays should be should be found in env_data folder')

    try:
        with open(INPUT_DATASET, 'rb') as f:
            x_set = pickle.load(f)
            print("I load x_set")
        with open(OUTPUT_DATASET, 'rb') as f:
            y_set = pickle.load(f)
            print("I load y_set")
    except:
        print('No data was available. Note: x_set.pickle and y_set.pickle should be found in env_data folder')

    return x_set, y_set, state_names, action_names

def plot_accuracy(features_names,accuracy_list,store_data=False):
    plt.xlabel('Features')
    plt.ylabel('Accuracy')
    plt.barh(features_names,accuracy_list)
    plt.show()


if __name__=="__main__":

    args=parser.parse_args()
    markovian_order=int(MARKOVIAN_ORDER)
    state_space_dim=int(STATE_SPACE_DIM)
    action_space_dim=int(ACTION_SPACE_DIM)
    polydegree=int(POLYNOMIAL_DEGREE)
    score_list=[] #for storing accuracy results
    time_tag=str(int(time.time()))

    randomsearch_dist_lstm = {
        "activation": ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
        "dropout_rate": [0,0.1,0.5],
        "num_neurons": np.random.randint(2, 11, size=1),
        "num_hidden_layers": np.random.randint(2,11, size=1),
        "learning_rate": np.random.choice([10**-1,10**-3], size=1),
        "num_lstm_units":np.random.randint(2,101,size=1),
        "decay": np.random.uniform(10**-3,10**-9, size=1),
        "markovian_order":[markovian_order],
        "state_space_dim":[state_space_dim],
        "action_space_dim":[action_space_dim]}

    random_search_nn = {
        "activation": ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
        "dropout_rate": [0,0.1,0.5],
        "num_neurons": np.random.randint(2, 30, size=1),
        "num_hidden_layers": np.random.randint(2,20, size=1),
        "learning_rate": np.random.choice([10**-1,10**-3], size=1),
        "decay": np.random.uniform(10**-3,10**-9, size=1),
        "state_space_dim":[state_space_dim],
        "action_space_dim":[action_space_dim]}

    random_search_gb={
        "loss":["ls", "lad", "huber"],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "min_samples_split": [2,10] ,#np.linspace(0.1, 0.5, 10),
        "min_samples_leaf": [1,2,10] ,#np.linspace(0.1 0.5, 10),
        "max_depth":[3,5],
        "max_features":["log2","sqrt"],
        "criterion": ["friedman_mse",  "mae"],
        "subsample":[0.5, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0,1.0,1.0],
        "n_estimators":[50,100,200]
    }

    x_set, y_set, state_names, action_names=read_env_data()

    if args.use_nn==True:
        scaler_x_set=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(x_set)
        scaler_y_set=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(y_set)
        joblib.dump(scaler_x_set, x_scalerdir)
        joblib.dump(scaler_y_set, y_scalerdir)
        x_set=scaler_x_set.transform(x_set)
        y_set=scaler_y_set.transform(y_set)

    if args.use_lstm==True:
        l=x_set.shape[0]
        m=x_set.shape[1]
        n=x_set.shape[2]
        print('reshaping data for normalization ..')
        print('shape of original inputs', x_set.shape, y_set.shape)
        x_set=x_set.reshape(l, m*n)
        scaler_x_set=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(x_set)
        scaler_y_set=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(y_set)

        joblib.dump(scaler_x_set, x_scalerdir)
        joblib.dump(scaler_y_set, y_scalerdir)

        x_set=scaler_x_set.transform(x_set)
        y_set=scaler_y_set.transform(y_set)

        x_set=x_set.reshape((l,m,n))

    args = parser.parse_args()
    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.33, random_state=42)


    if args.tune_rs==True:
        if args.use_lstm==True:
            model=KerasRegressor(build_fn=create_lstm_model_wrapper,epochs=EPOCHS, batch_size=1024, verbose=1)
            random_search = RandomizedSearchCV(estimator=model, param_distributions=randomsearch_dist_lstm, n_iter=50, n_jobs=-1, cv=5)
            result = random_search.fit(x_train, y_train)
            print("Best: %f using %s" % (result.best_score_, result.best_params_))
            filename='./models/'+time_stamp+'/tune_lstm_random_search_results_'+str(100*result.best_score_)+'.pkl'
            #filename='./models/lstm_random_search_results_'+str(100*result.best_score_)+'.pkl'
            joblib.dump(result.best_params_, filename)

        elif args.use_nn==True:
            model=KerasRegressor(build_fn=create_nn_model_wrapper,epochs=EPOCHS, batch_size=1024, verbose=1)
            random_search = RandomizedSearchCV(estimator=model, param_distributions=random_search_nn, n_iter=50, n_jobs=-1, cv=5)
            result = random_search.fit(x_train, y_train)
            print("Best: %f using %s" % (result.best_score_, result.best_params_))
            filename='./models/'+time_stamp+'/tune_nn_random_search_results_'+str(100*result.best_score_)+'.pkl'
            #filename='./models/nn_random_search_results_'+str(100*result.best_score_)+'.pkl'
            joblib.dump(result.best_params_, filename)
            #new code from branch small fixes
            config={"epochs": 1000,
                "batch_size": 512,
                "activation":result.best_params_["activation"],
                "n_layer": result.best_params_["num_hidden_layers"],
                "n_neuron": result.best_params_["num_neurons"],
                "lr": result.best_params_["learning_rate"],
                "decay": result.best_params_["decay"],
                "dropout": result.best_params_["dropout_rate"]
                    }
            nn_estimator=env_nn_modeler(state_space_dim=state_space_dim,action_space_dim=action_space_dim)
            nn_estimator.create_model(config)
            nn_estimator.train_nn_model(x_train,y_train,config["epochs"],config["batch_size"])
            nnmodel=nn_estimator.model
            nn_estimator.evaluate_nn_model(x_test, y_test,config["batch_size"])
            test_score=nn_estimator.score[1]*100
            randomsample=np.random.random_integers(0,10,1)
            x_sample=x_set[randomsample]
            print('random sample:', x_sample)
            predict_sample=nnmodel.predict(x_sample)
            print('estimator prediction: ', predict_sample)
            print('actual value:', y_set[randomsample])

            #Save model
            modelname='./models/'+time_stamp+'/nnmodel'+str(int(test_score))+'.h5'
            nnmodel.save(modelname)
            modelname2='./models/'+time_stamp+'/nnmodel.h5'
            nnmodel.save(modelname2)

        time.sleep(10)

    if args.use_gb==True and args.tune_rs==True:
        accuracy_file=('./accuracy_results/gb/%s_gb_scores.npy'%(time_tag))
        for i in range (0, y_set.shape[1]):
            gb_estimator=env_gb_modeler(state_space_dim=state_space_dim,action_space_dim=action_space_dim)
            gb_estimator.create_gb_model()
            gb_estimator.train_gb_model(x_train,y_train[:,i])
            score=gb_estimator.evaluate_gb_model(x_test, y_test[:,i])
            score_list.append(score)
            print('evaluation score for default is:', score)
            model=GradientBoostingRegressor()
            random_search = RandomizedSearchCV(estimator=model, param_distributions=random_search_gb, n_iter=10, n_jobs=-1, cv=3, verbose=0)
            result = random_search.fit(x_train, y_train[:,i])
            print("Best: %f using %s" % (result.best_score_, result.best_params_))
            filename='./models/'+time_stamp+'/tune_gb_random_search_results_'+str(i)+'th'+str(100*result.best_score_)+'.pkl'
            #filename='./models/gb_random_search_results_'+str(i)+'th'+str(100*result.best_score_)+'.pkl'
            joblib.dump(result.best_params_, filename)
            model_opt=GradientBoostingRegressor(result.best_params_)
            modelname='./models/gbmodel'+str(int(i))+'.sav'
            joblib.dump(model_opt, modelname)
        np.save(accuracy_file,score_list)
        print(accuracy_file)

    elif args.use_gb==True and args.tune_rs==False:
        accuracy_file=('./accuracy_results/poly/%s_gb_scores.npy'%(time_tag))
        print('using gradient boost regressor ....')
        for i in range (0, y_set.shape[1]):
            gb_estimator=env_gb_modeler()
            gb_estimator.create_gb_model()
            gb_estimator.train_gb_model(x_train,y_train[:,i])
            score=gb_estimator.evaluate_gb_model(x_test, y_test[:,i])
            score_list.append(score)
            print('evaluation score is:', score)
            modelname='./models/'+time_stamp+'/gbmodel'+str(int(i))+'.sav'
            #modelname='./models/gbmodel'+str(int(i))+'.sav'
            joblib.dump(gb_estimator.model, modelname)
        np.save(accuracy_file,score_list)
        print(accuracy_file)

    if args.use_poly==True:
        print('using polynomial fitting ....')
        accuracy_file=('./accuracy_results/poly/%s_%s_poly_training.npy'%(time_tag,POLYNOMIAL_DEGREE))
        for i in range (0, y_set.shape[1]):
            poly_estimator=env_poly_modeler()
            poly_estimator.create_poly_model(degree=polydegree)
            poly_estimator.train_poly_model(x_train,y_train[:,i])
            score=poly_estimator.evaluate_poly_model(x_test, y_test[:,i])
            print('evaluation score i=%s is: %s'% (i,score))
            modelname='./models/'+time_stamp+'/polymodel'+str(int(i))+'.sav'
            estimator_name='./models/'+time_stamp+'/polydegree.sav'
            #modelname='./models/polymodel'+str(int(i))+'.sav'
            joblib.dump(poly_estimator.model, modelname)
            joblib.dump(poly_estimator.poly, estimator_name)
            randomsample=np.random.random_integers(0,10,1)
            x_sample=x_set[randomsample]
            #print('random sample:', x_sample)
            predict_sample=poly_estimator.predict_poly_model(x_sample)
            print('estimator prediction: ', predict_sample)
            print('actual value:', y_set[randomsample,i])
            score_list.append(score)
        #plot_accuracy(features_names=state_names,accuracy_list=score_list)
        #Save results of the training accuracy
        np.save(accuracy_file,score_list)
        print('Accuracy file:',accuracy_file)

# use the hyperparamter tuned network
##default neural network without hyperparamter tuning
    if  args.tune_rs==False and args.use_lstm==True:
        the_lstm_estimator=env_lstm_modeler()
        config={"epochs": 100,
                "batch_size": 512,
                "activation":'linear',
                "n_hidden_layer": 1,
                "n_neuron": 12,
                "lr": 10**-1,
                "decay": 10**-3,
                "dropout": 0.5,
                "markovian_order":markovian_order, # should be >1 , zero order==>1
                "num_lstm_units": 10}
        the_lstm_estimator.create_model(config)
        the_lstm_estimator.train_nn_model(x_train,y_train,config["epochs"],config["batch_size"])
        lstmmodel=the_lstm_estimator.model
        the_lstm_estimator.evaluate_nn_model(x_test, y_test,config["batch_size"])
        test_score=the_lstm_estimator.score[1]*100
        randomsample=np.random.random_integers(0,10,1)
        x_sample=x_set[randomsample]
        print('random sample:', x_sample)
        predict_sample=lstmmodel.predict(x_sample)
        print('estimator prediction: ', predict_sample)
        print('actual value:', y_set[randomsample])
        modelname='./models/'+time_stamp+'/lstmmodel'+str(int(test_score))+'.h5'
        #modelname='./models/lstmmodel'+str(int(test_score))+'.h5'
        print(modelname)
        lstmmodel.save(modelname)
        modelname2='./models/'+time_stamp+'/lstmmodel.h5'
        #modelname2='./models/lstmmodel.h5'
        lstmmodel.save(modelname2)

    if args.tune_rs==False and args.use_nn==True:
        nn_estimator=env_nn_modeler(state_space_dim=state_space_dim,action_space_dim=action_space_dim)
        config=default_nn_config #modify using default
        nn_estimator.create_model(config)
        nn_estimator.train_nn_model(x_train,y_train,config["epochs"],config["batch_size"])
        nnmodel=nn_estimator.model
        nn_estimator.evaluate_nn_model(x_test, y_test,config["batch_size"])
        test_score=nn_estimator.score[1]*100
        randomsample=np.random.random_integers(0,10,1)
        x_sample=x_set[randomsample]
        print('random sample:', x_sample)
        predict_sample=nnmodel.predict(x_sample)
        print('estimator prediction: ', predict_sample)
        print('actual value:', y_set[randomsample])
        modelname='./models/'+time_stamp+'/nnmodel'+str(int(test_score))+'.h5'
        #modelname='./models/nnmodel'+str(int(test_score))+'.h5'
        nnmodel.save(modelname)
        modelname2='./models/'+time_stamp+'/nnmodel.h5'
        #modelname2='./models/nnmodel.h5'
        nnmodel.save(modelname2)
    else:
        pass
    print('Input dataset:',INPUT_DATASET)
    print('X set shape:', x_set.shape)
    print('\nOutput dataset:',OUTPUT_DATASET)
    print('Y set shape:', y_set.shape)
    print("End of training, you model has been store at './models/"+time_stamp)
