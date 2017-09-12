close all
clear all
clc

%% Parameters
path_audio_dbase = '../Database/CMR_dataset_full/audio/';
path_annote_dbase = '../Database/CMR_dataset_full/annotations/beats/';
wav_files = dir([path_audio_dbase '*.wav']);
num_files = length(wav_files);
Fs_reqd = 16000;
Fs = 44100;%Bit of a hack as I am assuming all audio files to be sampled at 44.1 kHz 
% Auditory spectrogram Parameters
frmlen = 100; %in ms
frame_conv_fac = 10^3/frmlen;
tc = 0;
fac = -1;%-2: Linear; -1: HWR
sr_ratio = Fs_reqd / 16000;
shft = log2(sr_ratio);
PARAS	= [frmlen, tc, fac, shft];
loadload;
op_stage = 5;
% Some partioning percentages
seen_perc = 95;
unseen_perc = 5;
train_seen_perc = 60;
train_dev_seen_perc = 10;
dev_seen_perc = 15;
test_seen_perc = 15;
dev_unseen_perc = 15;
test_unseen_perc = 15;
train_unseen_perc = 70;

num_unseen_files = round(unseen_perc*num_files/100);
unseen_file_id = randint_gen_unique(num_unseen_files,[1, num_files]);
seen_file_id = setdiff(1:num_files,unseen_file_id);
num_seen_files = length(seen_file_id);

%% Seen Main Loop: For each .wav file
% Pre-define the containers for different partitions
train_X = [];
train_Y = [];
train_dev_X = [];
train_dev_Y = [];
dev_X = [];
dev_Y = [];
test_X = [];
test_Y = [];

for file_id = 99:num_seen_files
   fname = wav_files(seen_file_id(file_id)).name(1:end-4);
   [y, Fs] = audioread([path_audio_dbase fname '.wav']);
   y = y(:,1);
   if Fs ~= Fs_reqd
      y = resample(y,Fs_reqd, Fs);
   end
%    Get the annotation file
   beat_annote = csvread([path_annote_dbase fname '.beats']);
   %    Convert the time stamps in first column to frame numbers
   beat_annote(:,1) = ceil(beat_annote(:,1)*frame_conv_fac);
   

%  Auditory Spectrogram  
   aud_spec = wav2aud(unitseq(y),PARAS,'p',0,op_stage); %Returns num_frames X 128
   aud_spec = aud_spec';
   [DIM, num_frames] = size(aud_spec);
   labels = zeros(2,num_frames);
   labels(2,beat_annote(:,1)) = 1;
   train_len = round(train_seen_perc* num_frames/ 100);
   train_dev_len = round(train_dev_seen_perc* num_frames/ 100);
   dev_len = round(dev_seen_perc* num_frames/ 100);
   train_X = [train_X aud_spec(:,1:train_len)];
   train_Y = [train_Y labels(:,1:train_len)];
   
   train_dev_X = [train_dev_X aud_spec(:,train_len+1:train_len+train_dev_len)];
   train_dev_Y = [train_dev_Y labels(:,train_len+1:train_len+train_dev_len)];
   
   dev_X = [dev_X aud_spec(:,train_len+train_dev_len+1:train_len+train_dev_len+dev_len)];
   dev_Y = [dev_Y labels(:,train_len+train_dev_len+1:train_len+train_dev_len+dev_len)];
   
   test_X = [test_X aud_spec(:,train_len+train_dev_len+dev_len+1:end)];
   test_Y = [test_Y labels(:,train_len+train_dev_len+dev_len+1:end)];
   disp(file_id);
end
%% UnSeen Main Loop: For each .wav file
train_unseen_X = [];
train_unseen_Y = [];

for file_id = 1:num_unseen_files
   fname = wav_files(unseen_file_id(file_id)).name(1:end-4);
   [y, Fs] = audioread([path_audio_dbase fname '.wav']);
   y = y(:,1);
   if Fs ~= Fs_reqd
      y = resample(y,Fs_reqd, Fs);
   end
%    Get the annotation file
   beat_annote = csvread([path_annote_dbase fname '.beats']);
   beat_annote(:,1) = round(beat_annote(:,1)*frame_conv_fac);
   
%    Convert the time stamps in first column to frame numbers
%  Auditory Spectrogram  
   aud_spec = wav2aud(unitseq(y),PARAS,'p',0,op_stage); %Returns num_frames X 128
   aud_spec = aud_spec';
   [DIM, num_frames] = size(aud_spec);
   labels = zeros(2,num_frames);
   labels(2,beat_annote(:,1)) = 1;
   
   train_len = round(train_unseen_perc* num_frames/ 100);
   dev_len = round(dev_unseen_perc* num_frames/ 100);
   train_unseen_X = [train_unseen_X aud_spec(:,1:train_len)];
   train_unseen_Y = [train_unseen_Y labels(:,1:train_len)];
   
   dev_X = [dev_X aud_spec(:,train_len+1:train_len+dev_len)];
   dev_Y = [dev_Y labels(:,train_len+1:train_len+dev_len)];
   
   test_X = [test_X aud_spec(:,train_len+dev_len+1:end)];
   test_Y = [test_Y labels(:,train_len+dev_len+1:end)];
   disp(file_id)
end

%% Save all the partitions if -v7.3 .mat files for reading using h5py in Python
save('../Data/Train.mat','train_X','train_Y','-v7.3');
save('../Data/Train-Dev.mat','train_dev_X','train_dev_Y','-v7.3');
save('../Data/Dev.mat','dev_X','dev_Y','-v7.3');
save('../Data/Test.mat','test_X','test_Y','-v7.3');
save('../Data/Train-Unseen.mat','train_unseen_X','train_unseen_Y','-v7.3');
