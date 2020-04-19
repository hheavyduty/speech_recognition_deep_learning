# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:51:29 2018

@author: USER
"""

import numpy as np
import config as cfg
import prepare_dev_data as pp_dev_data
'''from hat.preprocessing import pad_trunc_seqs, sparse_to_categorical, mat_2d_to_3d
from hat.models import Sequential
from hat.layers.core import InputLayer, Dense, Dropout, Flatten
from hat.callbacks import SaveModel, Validation
from hat.optimizers import Adam
import hat.backend as K'''
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,LSTM,Reshape
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.utils import np_utils
from sklearn.metrics import precision_score,recall_score,f1_score
from keras import metrics as kp



### hyper-params        # can be 1,2,3 or 4
type = 'home'
#type = 'home'   # can be 'home' or 'resi'
agg_num = 40
hop = 20
n_hid = 500

#print (cfg.dev_fe_mel_home_fd)
### train model from cross validation data
def train_cv_model(fold):
    # init path
    if type=='home':
        fe_fd = cfg.dev_fe_mel_home_fd
        labels = cfg.labels_home
        lb_to_id = cfg.lb_to_id_home
        tr_txt = cfg.dev_evaluation_fd + '/home_fold' + str(fold) + '_train.txt'
        te_txt = cfg.dev_evaluation_fd + '/home_fold' + str(fold) + '_evaluate.txt'
        print (te_txt)
        print (tr_txt)		   
    if type=='resi':
        fe_fd = cfg.dev_fe_mel_resi_fd
        labels = cfg.labels_resi
        lb_to_id = cfg.lb_to_id_resi
        tr_txt = cfg.dev_evaluation_fd + '/residential_area_fold' + str(fold) + '_train.txt'
        te_txt = cfg.dev_evaluation_fd + '/residential_area_fold' + str(fold) + '_evaluate.txt'
        
    #n_out = len( labels )
    
    # load data to list
    #tr_X, tr_y = pp_dev_data.LoadAllData( cfg.dev_fe_mel_home_fd, cfg.dev_evaluation_fd + '/home_fold' + str(fold) + '_train.txt',cfg.lb_to_id_home, agg_num, hop )
    tr_X,tr_y = pp_dev_data.LoadAllData( cfg.dev_fe_mel_home_fd,cfg.dev_evaluation_fd + '/home_fold' + str(fold) + '_train.txt' , cfg.lb_to_id_home,40, 20 )
    labelencoder_train = LabelEncoder()
    labelencoder_train.fit(tr_y)
    tr_y1 = labelencoder_train.transform(tr_y)
    tr_y1 = np_utils.to_categorical(tr_y)
    
    #print (tr_X.shape)
    #print (tr_y.shape)
    #n_freq = tr_X.shape[2]
    return tr_X,tr_y1,tr_y
    
    # build model
def model(train_x,train_y,test_x,y4):
    train_xx=np.expand_dims(train_x,axis=3)
    test_xx=np.expand_dims(test_x,axis=3)
    seq = Sequential()
    #seq.add( InputLayer( (agg_num, n_freq) ) )
    #seq.add( Flatten(input_shape=(11,40)) )
    seq.add(Conv2D(80,kernel_size=(5,5), activation='relu',input_shape=(40,40,1)))
    seq.add(MaxPooling2D(pool_size=(2,2)))
    seq.add(Conv2D(80,kernel_size=(5,5), activation='relu'))
    seq.add(MaxPooling2D(pool_size=(1,1)))
    seq.add(Conv2D(80,kernel_size=(5,5), activation='relu'))
    seq.add(MaxPooling2D(pool_size=(1,1)))
    seq.add(Conv2D(80,kernel_size=(5,5), activation='relu'))
    seq.add(MaxPooling2D(pool_size=(1,1)))
    seq.add(Dropout(0.25))
    seq.add(Reshape((80,-1)))
    seq.add(LSTM(80,activation='tanh'))
    seq.add(Dropout(0.25))
    seq.add(Reshape((80,-1)))
    seq.add(LSTM(80,activation='tanh'))
    seq.add(Dropout(0.25))
    seq.add(Reshape((80,-1)))
    seq.add(LSTM(80,activation='tanh'))
    seq.add(Reshape((1,-1)))
    seq.add(Flatten())
    seq.add( Dense( n_hid,activation='relu' ) )
    seq.add( Dropout( 0.25 ) )
    seq.add( Dense( n_hid, activation='relu' ) )
    seq.add( Dropout( 0.25 ) )
    #seq.add( Dense( n_hid, activation='relu' ) )
    #seq.add( Dropout( 0.1 ) )
    seq.add( Dense(7,activation ='sigmoid' ) )
    #md = seq.combine()
    
    # print summary info of model
    seq.summary()
    
    # optimization method
    #optimizer = Adam(1e-3)
    
    # callbacks (optional)
    # save model every n epoch
    #pp_dev_data.CreateFolder( cfg.dev_md_fd )
    #save_model = SaveModel( dump_fd=cfg.dev_md_fd, call_freq=5 )
    
    # validate model every n epoch
    #validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=None, te_y=None, metrics=['binary_crossentropy'], call_freq=1, dump_path=None )
    
    # callbacks function
    #callbacks = [validation, save_model]
    seq.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    
    # train model
    seq.fit(train_xx,train_y, batch_size=1,epochs=1 )
    #print (seq.evaluate(test_x,test_y,batch_size=1))
    y_pred=seq.predict_classes(test_xx)
    #f1_score(y4,y_pred)
    return y_pred
	


### main function
if __name__ == '__main__':
    tr_x_1,tr_y_1,y1=train_cv_model(1)
    tr_x_2,tr_y_2,y2=train_cv_model(2)
    tr_x_3,tr_y_3,y3=train_cv_model(3)
    tr_x_4,tr_y_4,y4=train_cv_model(4)
    train_x = np.vstack([tr_x_1,tr_x_2,tr_x_3])
    train_y = np.vstack([tr_y_1,tr_y_2,tr_y_3])
    print (train_x.shape)
    print (train_y.shape)
    y_pred=model(train_x,train_y,tr_x_4,tr_y_4)
    print(f1_score(y4,y_pred,average='micro'))
    
    

		
    
	  
