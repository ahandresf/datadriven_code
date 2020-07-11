import numpy as np
import pandas as pd


'''
* We get a minibatch 60, 30 normal and 30 anomaly videos
* Each video will have 32 segments.
    *Va=>32*30=960 anomaly and Vn=>960 normal videos
* loss function must received (y_true,y_pred) as arguments
    * y_true= label (1 for anomaly, 0 for normal) per segment
    * y_pred= score (0-1) per segment
    * We compute loss function over a set of (Vn,Va).
        * Vn = y_pred if (y_true==0)
        * Va = y_pred if (y_true==1)
'''
def _smothness(Va):
    smothness_term=np.sum(Va)
    return smothness_term

def _sparsity(Va):
    diff=Va[0:-1]-Va[1::]
    sparsity_term=np.square(diff).sum()
    return sparsity_term

def _loss_custom(Va,Vn):
    '''
    Va: Predicted score anomaly (positive label, 1)
    Vn: Predicted score normal (negative label, 0)
    '''
    Va=np.array(Va)
    Vn=np.array(Vn)
    l1=8e-5
    l2=8e-5
    loss_value=max(0,1-max(Va)+max(Vn))+l1*_smothness(Va)+l2*_sparsity(Va)
    return(loss_value)

# Define custom loss
def loss_func(y_true,y_pred):
    labels=np.array(y_true)
    scores=np.array(y_pred)
    df=pd.DataFrame({'labels':labels,'scores':scores})
    Va=df[df['labels']==1]['scores']
    Vn=df[df['labels']==0]['scores']
    loss_value=_loss_custom(Va=Va,Vn=Vn)
    return loss_value
