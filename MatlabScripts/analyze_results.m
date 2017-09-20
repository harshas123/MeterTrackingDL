close all
clear all
clc

load ../Results/Parameters_NN_config_NumLayers_3num_iter_10.mat
train_Y = LSTM_data_format_to_spectro(permute(train_Y,[2,1,3]));
train_dev_Y = LSTM_data_format_to_spectro(permute(train_dev_Y,[2,1,3]));
dev_Y = LSTM_data_format_to_spectro(permute(dev_Y,[2,1,3]));
test_Y = LSTM_data_format_to_spectro(permute(test_Y,[2,1,3]));

figure;
plot(dev_Y(:,2),'LineWidth',4);
hold on;
plot(dev_pred(:,2));
grid on;
xlabel('Time Index');
set(gca,'FontSize',25);
hl = legend('Labels','Predictions');
set(hl,'Location','Best')
xlim([0 1000])