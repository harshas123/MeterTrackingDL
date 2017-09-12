close all
clear all
clc

% This script is to visualize and verify if the data used for training the
% network is correct.

path_data = '../Data/';
partition_type = 'Train';% 'Train', 'Train-Dev', 'Dev', 'Test'
load([path_data partition_type '.mat']);
if strcmp(partition_type, 'Train')
    X = train_X;
    Y = train_Y;
    clear train_X train_Y
elseif strcmp(partition_type, 'Train-Dev')
    X = train_dev_X;
    Y = train_dev_Y;
    clear train_dev_X train_dev_Y
elseif strcmp(partition_type, 'Dev')
    X = dev_X;
    Y = dev_Y;
    clear dev_X dev_Y
elseif strcmp(partition_type, 'Test')
    X = test_X;
    Y = test_Y;
    clear test_X test_Y
end



