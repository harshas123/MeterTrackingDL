close all
clear all
clc

% Data base has about 996.9834 minutes of data
% roughly 16 hours of data

path_audio_dbase = '../Database/CMR_dataset_full/audio/';
audio_files = dir([path_audio_dbase '*.wav']);

num_files = length(audio_files);
len=0;
for i=1:num_files
   [y, Fs] = audioread([path_audio_dbase audio_files(i).name]);
   len = len+length(y)/(Fs*60);
   disp(i)
end

