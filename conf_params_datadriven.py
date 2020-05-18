#!/bin/usr/env python
'''
Author: Andres Felipe Alba Hernandez
v-analba@microsoft.com
Andres.AlbaHernandez@us.sogeti.com
Load configuration variables on the data driven model
instead of modifying parameters at the env_data_modeler
we can set the values here an import them.
'''

'''
INPUT DATASETS
'''
#Reduce
INPUT_DATASET ='./his_data/short_x_cut.pickle'
OUTPUT_DATASET ='./his_data/short_y_cut.pickle'

'''
ACTION AND STATE NAMES
'''
#Reduce
STATE_NAMES = './his_data/short_actions.npy'
ACTION_NAMES = './his_data/short_states.npy'
'''
ACTION AND STATE SPACE DIMENSIONS
Number of variables in the state an action space eg. (velocity, position) will
be dimension 2 for the state.
'''
#STATE_SPACE_DIM = 30
#ACTION_SPACE_DIM = 8

STATE_SPACE_DIM = 9
ACTION_SPACE_DIM = 4

'''
Polynomial methods parameters
'''
POLYNOMIAL_DEGREE=1

'''
GENERAL PARAMETERS OF NEURONAL NETWORKS
'''
DROPOUT_RATE = 0.1 #how many neurons you drop during training. It seems to help reducing bias.
EPOCHS = 1000
BATCH_SIZE = 512
ACTIVATION = 'tanh'
N_LAYER = 3
N_NEURON = 19
LEARNING_RATE = 0.001
DECAY = 0.00022459583969984315
DROPOUT = 0

default_nn_config={"epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "activation": ACTIVATION,
        "n_layer": N_LAYER,
        "n_neuron": N_NEURON,
        "lr": LEARNING_RATE,
        "decay": DECAY,
        "dropout": DROPOUT}

'''
EPOCHS = 100
BATCH_SIZE = 512
ACTIVATION = 'linear'
N_LAYER = 5
N_NEURON = 12
LEARNING_RATE = 10**-5
DECAY = 10**-7
DROPOUT = 0.5
# Original in script
config={"epochs": 100,
        "batch_size": 512,
        "activation": 'linear',
        "n_layer": 5,
        "n_neuron": 12,
        "lr": 10**-5,
        "decay": 10**-7,
        "dropout": 0.5}
'''

'''
This paramaters determine the dependance of the history. e.g.(a markovian order
of two will depend on the current state and the previous state)
It is use by the LSTM.
'''
MARKOVIAN_ORDER = 2

'''
Output Folder
'''


'''
OLD STUFF
'''

'''
ACTION AND STATE NAMES
'''
#STATE_NAMES = './env_data/state_names.npy'
#ACTION_NAMES = './env_data/action_names.npy'

#STATE_NAMES = 'C:/Users/aalbaher/dataset_pttgc/data_preprocess/May_06_104606/short_states.npy'
#ACTION_NAMES = 'C:/Users/aalbaher/dataset_pttgc/data_preprocess/May_06_104606/short_actions.npy'

#STATE_NAMES = 'C:/Users/aalbaher/dataset_pttgc/data_preprocess/May_13_165417/short_states.npy'
#ACTION_NAMES = 'C:/Users/aalbaher/dataset_pttgc/data_preprocess/May_13_165417/short_actions.npy'

'''
INPUT DATASETS
'''

#Reduce dataset 9s 4a
#INPUT_DATASET='C:/Users/aalbaher/dataset_pttgc/data_preprocess/May_13_165417/short_x_cut.pickle'
#OUTPUT_DATASET='C:/Users/aalbaher/dataset_pttgc/data_preprocess/May_13_165417/short_y_cut.pickle'

#Reduce dataset 9s 4a
#INPUT_DATASET='C:/Users/aalbaher/dataset_pttgc/data_preprocess/May_13_165417/short_x.pickle'
#OUTPUT_DATASET='C:/Users/aalbaher/dataset_pttgc/data_preprocess/May_13_165417/short_y.pickle'

#Standard
##INPUT_DATASET='./env_data/x_set.pickle'
##OUTPUT_DATASET='./env_data/y_set.pickle'

#Total dataset
##INPUT_DATASET='./env_data/total/x_set_total.pickle'
##OUTPUT_DATASET='./env_data/total/y_set_total.pickle'

#Differencial Data
#INPUT_DATASET='./env_data/diff/x_set_diff.pickle'
#OUTPUT_DATASET='./env_data/diff/y_set_diff.pickle'

#Differencial Data with shift=10
#INPUT_DATASET='./data_parser_output/x_set_diff_s_10.pickle'
#OUTPUT_DATASET='./data_parser_output/y_set_diff_s_10.pickle'

#INPUT_DATASET='./data_parser_output/x_set_diff_subset.pickle'
#OUTPUT_DATASET='./data_parser_output/y_set_diff_subset.pickle'

#GCG2 Neuronal Networks
#INPUT_DATASET = './data_parser_output/April_21_091929/x_set_diff_s_10.pickle'
#OUTPUT_DATASET = './data_parser_output/April_21_091929/y_set_diff_s_10.pickle'

#GCG1 Linear Model April 27
###INPUT_DATASET='./data_parser_output/April_27_184109/x_set_sv5.pickle'
###OUTPUT_DATASET='./data_parser_output/April_27_184109/y_set_sv5.pickle'

#Reduce dataset with 10s 4a
#INPUT_DATASET='C:/Users/aalbaher/dataset_pttgc/data_preprocess/May_06_104606/hort_x_cut.pickle'
#OUTPUT_DATASET='C:/Users/aalbaher/dataset_pttgc/data_preprocess/May_06_104606/hort_y_cut.pickle'
