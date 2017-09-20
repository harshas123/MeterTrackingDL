% close all
clear all
clc


% The objective of this code is to compare the output mel spectrogram from
% HTK and melspec and adjust the parameters in HTK to match melspec output
% as desired by Ajay. 

% HTK Config file: ./config_44100.con
path_config = './config_44100.con';
path_dbase = '../Database/CMR_dataset_full/audio/';
wav_files = dir([path_dbase '*.wav']);

% Load the Audio Data from the first wav_files
fname_wav = [path_dbase wav_files(1).name];
fname_htk = ['./' wav_files(1).name(1:end-4) 'melspec'];
[y, Fs] = audioread(fname_wav);

% MELSPEC from melspec
[mel_spec, cntrs] = melspec(y(:,1), Fs, 10, 46.44, [60 16000], 80);

% Using HTK
tmp_fname_wav = 'Test_HTK_melspec.wav';
tmp_fname_htk = 'Test_HTK_melspec.melspec';

audiowrite(tmp_fname_wav,y(:,1),Fs);
[status, result] = system(['HCopy -T 1 -C ' path_config ' ' fname_wav ' ' fname_htk]);
if status
   error(result) 
end
[obs_vec] = readhtk_new(tmp_fname_htk);
obs_vec = obs_vec';
% Compare the Dimensions
DIM = table;

DIM.melspec = size(mel_spec)';
DIM.htk = size(obs_vec)';
DIM.Properties.RowNames = {'Rows','Columns'};
disp(DIM)

% Visual Comparison
figure;
subplot(211);
imagesc(mel_spec);
axis xy;
title('melspec');
set(gca,'FontSize',30);
xlabel('Frame Index');
ylabel('Freq. Bin Index');

subplot(212);
imagesc(obs_vec);
axis xy;
title('HTK');
set(gca,'FontSize',30);
xlabel('Frame Index');
ylabel('Freq. Bin Index');

disp('As you can see...Outputs from either are nearly similar')