import librosa as lb
import librosa.display
from glob import glob
import os 
import numpy as np
import pandas as pd
import csv

def create_csv():
    '''
    This function with create the csv file with all the audio file path and machine type
    Parameters:
        path : path for location where the all audio file present
        file_name : name of the csv file in which you want to save data
    '''
    header = ['filename', 'class']

    with open('assets/data/audio_file_list.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name in glob('assets/data/audio_data/**/id_00/*/*.wav', recursive=True):
            data = []
            arr = name.split('/')
            print(arr)
            data.append(name)
            data.append(arr[-4]+"_"+arr[-2])
            writer.writerow(data)
            
def create_validation_data_csv():
    '''
    This function with create the csv file with all the audio file path and machine type
    Parameters:
        path : path for location where the all audio file present
        file_name : name of the csv file in which you want to save data
    '''
    header = ['filename', 'class']

    with open('assets/data/validation_file_list.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name in glob('assets/data/audio_data/validation/*/*.wav', recursive=True):
            data = []
            arr = name.split('/')
            print(arr)
            data.append(name)
            data.append(arr[-2]+"_abnormal")
            writer.writerow(data)

            
def mfcc_array(i:int,data:object, files)->object:
    '''
    This function will help to get audio file feature and return it
    Parameters:
        i : file index
        data : audio data to calculate feature
    '''
    filename = files['filename'][i]
    librosa_load = lb.load(str(filename),sr=None)
    audio, sample_rate = librosa_load
    mfcc = lb.feature.mfcc(audio,sr=sample_rate,n_mfcc=2)
#     mfcc = mfcc.flatten()
    mean = mfcc.mean(axis=1)
    print("mean shape",mean.shape)
    print("data shape",data.shape)
    if len(data) ==0:
        data = mean
    else:
        data = np.vstack((data,mean))
    
    return data


def get_target(files_list):
    '''This function will return the target'''
    lst = []
    for i in files_list.index:
        lst.append(files_list['class'][i])
    y =np.array(lst)
    y = np.reshape(y,(files_list.index.stop,-1))
    return y

#hstacking the features and the targets
def fuse_target_mfcc(data:object,y:object)->object:
    '''This function will combine the feature and target'''
    data = np.hstack((data, y))
    return data

#looping through csv file to calculate the mfcc from every file
def manageMfcc(files_list)->object:
    '''This function will get the feature for all the file list in csv'''
    lst=[]
    data =np.array(lst)
    for i in files_list.index:
        data = mfcc_array(i,data, files_list)
          
    return data


# Create Dataset to create model
# create_csv()
audio_files = pd.read_csv('assets/data/audio_file_list.csv')
data = fuse_target_mfcc(manageMfcc(audio_files),get_target(audio_files))
df1 = pd.DataFrame(data)

df1.to_csv("assets/data/audio_dataset.csv", header=None, index=False)

# Create Dataset for model validation
# create_validation_data_csv()
validation_audio_files = pd.read_csv('assets/data/validation_file_list.csv')
validation_data = fuse_target_mfcc(manageMfcc(validation_audio_files),get_target(validation_audio_files))
df2 = pd.DataFrame(validation_data)
    
df2.to_csv("assets/data/validation_dataset.csv", header=None, index=False)

