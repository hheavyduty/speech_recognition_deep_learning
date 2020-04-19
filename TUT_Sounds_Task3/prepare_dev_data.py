# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 11:48:41 2018

@author: ARKADIP GHOSH
"""

'''SUMMARY:  prepare data for development data
AUTHOR:   Qiuqiang Kong
Created:  2016.06.26
Modified: 2016.10.11 Modify variable name
--------------------------------------
'''
import sys
sys.path.append( 'activity_detection_lib' )
import numpy as np
import config as cfg
import wavio
import os
from scipy import signal
#from scipy.io import wavfile
import librosa
import pickle as cPickle
import matplotlib.pyplot as plt
#from hat.preprocessing import mat_2d_to_3d
from activity_detection import activity_detection
import sed_eval
from keras.layers import Dense,Dropout,Flatten
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.utils import np_utils


### readwav
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

###
# calculate mel feature
def mat_2d_to_3d(x, agg_num, hop):
    
    # Pad to at least one block. 
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
        
    # Segment 2d to 3d. 
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop
    return np.array(x3d)
def GetMel( wav_fd, fe_fd, n_delete ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.wav') ]
    names = sorted(names)
    for na in names:
        print (na)
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )
        assert fs==cfg.fs
        ham_win = np.hamming(cfg.win)
        [f, t, X] = signal.spectral.spectrogram( wav, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        
        # define global melW, avoid init melW every time, to speed up. 
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( fs, n_fft=cfg.win, n_mels=40, fmin=0., fmax=22100 )
            melW /= np.max(melW, axis=-1)[:,None]
            
        X = np.dot( X, melW.T )
        X = X[:, n_delete:]
        
        # DEBUG. print mel-spectrogram
        #plt.matshow(np.log(X.T), origin='lower', aspect='auto')
        #plt.show()
        #pause
        
        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )

### Without background
# load training data and label
def LoadAllData( fe_fd, txt_file, lb_to_id, agg_num, hop ):
    # add acoustic sound and id to Xlist, ylist
    fr = open( txt_file, 'r' )
    Xlist, ylist = [], []
    for line in fr.readlines():
        line_list = line.split('\t')
        
        # parse info
        path, scene, bgn, fin, lb = line_list[0], line_list[1], float(line_list[2]), float(line_list[3]), line_list[4].split('\n')[0]
        
        # load whole feature
        fe_path = fe_fd + '/' + path.split('/')[-1][0:4] + '.f'
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        # get sub feature
        ratio = cfg.fs / cfg.win
        X = X[ int(bgn*ratio):int(fin*ratio), : ]
        
        # aggregate feature
        X3d = mat_2d_to_3d( X, agg_num, hop )
        #print (X3d.shape)
        
        Xlist.append( X3d )
        	   
        #print (lb_to_id[lb])
        ylist +=[ cfg.lb_to_id_home[lb] ] * len(X3d)
    Xlist = np.array(Xlist)	    
    fr.close()
    
    
    return np.concatenate( Xlist, axis=0 ), ylist 


### 
# return names to be detect from .txt
def GetWavNamesFromTxt( txt_file ):
    fr = open( txt_file, 'r')
    names = []
    for line in fr.readlines():
        line_list = line.split('\t')
        
        # parse info
        path, scene, bgn, fin, label = line_list[0], line_list[1], float(line_list[2]), float(line_list[3]), line_list[4].split('\r')[0]
        na = path.split('/')[-1][0:4]
        if na not in names:
            names.append( na )
        
    return names

# load annoation file, return list of dict
def LoadGtAnn( txt_file ):
    fr = open( txt_file, 'r')
    index = 0
    
    lists = []
    for line in fr.readlines():
        line_list = line.split('\t')
        
        # parse info
        bgn, fin, label = float(line_list[0]), float(line_list[1]), line_list[2].split('\n')[0]
        lists.append( { 'event_label':label, 'event_onset':bgn, 'event_offset':fin } )
        
    return lists

###
# get out_list from scores
def OutMatToList( scores, thres, id_to_lb ):
    n_smooth = 10
    N, n_class = scores.shape
    
    lists = []
    for i1 in range( n_class ):
        bgn_fin_pairs = activity_detection( scores[:,i1], thres, n_smooth )
        for i2 in range( len(bgn_fin_pairs) ): 
            lists.append( { 'event_label':id_to_lb[i1], 
                            'event_onset':bgn_fin_pairs[i2]['bgn'] / (44100./1024.), 
                            'event_offset':bgn_fin_pairs[i2]['fin'] / (44100./1024.) } )
    return lists

# print lists to txt
def PrintListToTxt( lists, path ):
    f = open( path, 'w' )
    for li in lists:
        f.write( str(li['event_onset']) + '\t' + str(li['event_offset']) + '\t' + li['event_label'] + '\n' )
    f.close()
    print ('Write out detection result to', path, 'successfully!')


### print score (arranged from http://tut-arg.github.io/sed_eval/tutorial.html)
''' 
file_list should be:
------ begin file ------
[ { 'reference_file': 'xxx.ann', 'estimated_file': 'xxx_detect.ann' }, 
  { 'reference_file': 'yyy.ann', 'estimated_file': 'yyy_detect.ann' }, 
  ...
]
------ edn file ------
'''
def PrintScore( file_list, labels ):
    pairs = []
    
    # Get used event labels
    for file_pair in file_list:
        reference_event_list = sed_eval.io.load_event_list(file_pair['reference_file'])
        estimated_event_list = sed_eval.io.load_event_list(file_pair['estimated_file'])
        pairs.append({'reference_event_list': reference_event_list,
                    'estimated_event_list': estimated_event_list})

    # Start evaluating
    event_labels = labels
    
    # Create metrics classes, define parameters
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(event_label_list=event_labels,
                                                                    time_resolution=1)
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(event_label_list=event_labels,
                                                                t_collar=0.250)
    
    # Go through files
    for list_pair in pairs:
        segment_based_metrics.evaluate(list_pair['reference_event_list'],
                                    list_pair['estimated_event_list'])
        event_based_metrics.evaluate(list_pair['reference_event_list'],
                                    list_pair['estimated_event_list'])
    
    # Get only certain metrics
    overall_segment_based_metrics = segment_based_metrics.results_overall_metrics()
    print ("Accuracy:", overall_segment_based_metrics['accuracy']['accuracy'])
    
    # Or print all metrics as reports
    print (segment_based_metrics)
    #print event_based_metrics


### 
# create an empty folder
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
if __name__ == "__main__":
    CreateFolder( cfg.dev_fe_fd )
    CreateFolder( cfg.dev_fe_mel_fd )
    CreateFolder( cfg.dev_fe_mel_home_fd )
    CreateFolder( cfg.dev_fe_mel_resi_fd )
    
    # calculate all features
    GetMel( cfg.dev_wav_home_fd, cfg.dev_fe_mel_home_fd, n_delete=0 )
GetMel( cfg.dev_wav_resi_fd, cfg.dev_fe_mel_resi_fd, n_delete=0 )








'''fold = 1        # can be 1,2,3 or 4
#type = 'resi'
type = 'home'   # can be 'home' or 'resi'
agg_num = 11
hop = 5
n_hid = 500


tr_X_1,tr_Y_1 = LoadAllData( cfg.dev_fe_mel_home_fd,cfg.dev_evaluation_fd + '/home_fold' + str(fold) + '_train.txt' , cfg.lb_to_id_home, 11, 5 )
print (tr_X_1.shape)
print (len(tr_Y_1))


tr_X_2,tr_Y_2 = LoadAllData( cfg.dev_fe_mel_home_fd,cfg.dev_evaluation_fd + '/home_fold' + str(2) + '_train.txt' , cfg.lb_to_id_home, 11, 5 )
print (tr_X_2.shape)
print (len(tr_Y_2))




tr_X_3,tr_Y_3 = LoadAllData( cfg.dev_fe_mel_home_fd,cfg.dev_evaluation_fd + '/home_fold' + str(3) + '_train.txt' , cfg.lb_to_id_home, 11, 5 )
print (tr_X_3.shape)
print (len(tr_Y_3))




tr_X_4,tr_Y_4 = LoadAllData( cfg.dev_fe_mel_home_fd,cfg.dev_evaluation_fd + '/home_fold' + str(4) + '_train.txt' , cfg.lb_to_id_home, 11, 5 )
print (tr_X_4.shape)
print (len(tr_Y_4))




train_x=np.vstack([tr_X_1,tr_X_2,tr_X_3])
print (train_x.shape)



train_y=tr_Y_1+tr_Y_2+tr_Y_3
print (np.array(train_y).shape)



labelencoder_train = LabelEncoder()
labelencoder_train.fit(train_y)
tr_y_1 = labelencoder_train.transform(train_y)
tr_y = np_utils.to_categorical(tr_y_1)  
print (tr_y.shape) 


train_xx = train_x.reshape(17107,440)
print (train_xx.shape)


model = Sequential()
model.add(Dense(500,activation='relu',input_dim=440))
model.add(Dropout(0.1))
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(11,activation='sigmoid'))



model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(train_xx,tr_y,batch_size=1,epochs=2,verbose=1)
'''       




   
   