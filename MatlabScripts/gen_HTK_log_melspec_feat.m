close all
clear all
clc

% Create the Log-Melspec features using HTK
path_audio_dbase = '../Database/CMR_dataset_full/audio/';
path_features = '../Features/';
path_txt_files = '../HTKTextFiles/';
path_config = [path_txt_files 'config_44100.con'];
[status1, result1] = HCopy_bulk(path_audio_dbase,path_config,path_txt_files);
if status1
   error(result1) 
end
[status, result] = system(['mv ' path_audio_dbase '*.mfc ' path_features]);

if status
   error(result) 
end
