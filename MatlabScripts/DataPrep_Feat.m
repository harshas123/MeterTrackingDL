close all
clear all
clc

% NOTE: Make sure you have the desired features in "path_features"
% directory before running this file. Feature in "path_features" can be
% created using "gen_HTK_log_melspec_feat.m". To change feature extraction
% parameters llok at the HTK config file at "../HTKTextFiles/config_44100.con"
%% Parameters
path_features = '../Features/';
path_annote_dbase = '../Database/CMR_dataset_full/annotations/beats/';
feat_files = dir([path_features '*.mfc']);%mfc is a misnomer; 
%  although I have melspec features, I have named it mfc...generic
%  convenience I guess ....

num_files = length(feat_files);
hop_size = 10;% in ms...
frame_conv_fac = 10^3/hop_size;

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

for file_id = 1:num_seen_files
   fname = feat_files(seen_file_id(file_id)).name(1:end-4);
   %  Get the already computed Features
   [obs_vec] = readhtk_new([path_features fname '.mfc']);
   obs_vec = obs_vec';
   [DIM, num_frames] = size(obs_vec);
   
%    Get the annotation file and generate Gaussain smoothed Labels
   beat_annote = csvread([path_annote_dbase fname '.beats']);
   %    Convert the time stamps in first column to frame numbers
   beat_annote(:,1) = round(beat_annote(:,1)*frame_conv_fac);
%  Get  Gaussain smoothed Labels
   labels = genLabels(beat_annote, num_frames,5,1.5);	% Hardcoded smoothing params
  
%    Set the data and labels in the required format
   train_len = round(train_seen_perc* num_frames/ 100);
   train_dev_len = round(train_dev_seen_perc* num_frames/ 100);
   dev_len = round(dev_seen_perc* num_frames/ 100);
   train_X = [train_X obs_vec(:,1:train_len)];
   train_Y = [train_Y labels(:,1:train_len)];
   
   train_dev_X = [train_dev_X obs_vec(:,train_len+1:train_len+train_dev_len)];
   train_dev_Y = [train_dev_Y labels(:,train_len+1:train_len+train_dev_len)];
   
   dev_X = [dev_X obs_vec(:,train_len+train_dev_len+1:train_len+train_dev_len+dev_len)];
   dev_Y = [dev_Y labels(:,train_len+train_dev_len+1:train_len+train_dev_len+dev_len)];
   
   test_X = [test_X obs_vec(:,train_len+train_dev_len+dev_len+1:end)];
   test_Y = [test_Y labels(:,train_len+train_dev_len+dev_len+1:end)];
   disp(file_id);
end
%% UnSeen Main Loop: For each .wav file
train_unseen_X = [];
train_unseen_Y = [];

for file_id = 1:num_unseen_files
   fname = feat_files(unseen_file_id(file_id)).name(1:end-4);
   %  Get the already computed Features
   [obs_vec] = readhtk_new([path_features fname '.mfc']);
   obs_vec = obs_vec';
   [DIM, num_frames] = size(obs_vec);
%    Get the annotation file
   beat_annote = csvread([path_annote_dbase fname '.beats']);
%    Convert the time stamps in first column to frame numbers
   beat_annote(:,1) = round(beat_annote(:,1)*frame_conv_fac);
   
   %  Get  Gaussain smoothed Labels
   labels = genLabels(beat_annote, num_frames,5,1.5);	% Hardcoded smoothing params
   
%    Store the data and labels in required format
   train_len = round(train_unseen_perc* num_frames/ 100);
   dev_len = round(dev_unseen_perc* num_frames/ 100);
   train_unseen_X = [train_unseen_X obs_vec(:,1:train_len)];
   train_unseen_Y = [train_unseen_Y labels(:,1:train_len)];
   
   dev_X = [dev_X obs_vec(:,train_len+1:train_len+dev_len)];
   dev_Y = [dev_Y labels(:,train_len+1:train_len+dev_len)];
   
   test_X = [test_X obs_vec(:,train_len+dev_len+1:end)];
   test_Y = [test_Y labels(:,train_len+dev_len+1:end)];
   disp(file_id)
end

%% Save all the partitions if -v7.3 .mat files for reading using h5py in Python
save('../Data/Train.mat','train_X','train_Y','-v7.3');
save('../Data/Train-Dev.mat','train_dev_X','train_dev_Y','-v7.3');
save('../Data/Dev.mat','dev_X','dev_Y','-v7.3');
save('../Data/Test.mat','test_X','test_Y','-v7.3');
save('../Data/Train-Unseen.mat','train_unseen_X','train_unseen_Y','-v7.3');
