clc 
clear all

model_path='I_O_HMM_T3_B4.mat'; %%% model file Path 
trainData_path='train_data_processed_std.mat'; %%% train data path 
testData_path='test_data.mat'; %%% test data path 
nT=3; %%% look back steps  
nB=4; %%% number of hidden states 
[train_Y,train_prediction,test_Y,test_prediction] = modelInfer_I_O_HMM_one_input(model_path,trainData_path,testData_path,3,4);


%%% algorithm evaluation %%%
MAE_train=sum(abs(train_Y-train_prediction))/length(train_prediction) %%% calculate mean absolute error for training data
MAE_test=sum(abs(test_Y-test_prediction))/length(test_prediction)%%% calculate mean absolute error for test data
max_test=max(test_Y)  %%% maximum range of the test data 
min_test=min(test_Y)  %%% minimum range of the test data 
range_test=max_test-min_test %%% the range of the test data 



%%%% plot %%%%
figure (3)
plot(1:length(train_Y),train_Y,'b');
hold on; 
plot(1:length(train_Y),train_prediction,'r'); 
title('Groud truth and predction (training data)'); 
xlabel('time sequence'); 
ylabel('value'); 
legend('Ground truth','Prediction ')


figure (4) 
plot(1:length(test_Y),test_Y,'b');
hold on; 
plot(1:length(test_Y),test_prediction,'r'); 
title('Groud truth and predction (test data)'); 
xlabel('time sequence'); 
ylabel('value'); 
legend('Ground truth','Prediction ')


figure (5)
plot(train_Y,train_prediction,'bo');
hold on;
plot(-1:6,-1:6,'r');
title('Groud truth v.s predction (train data)'); 
xlabel('prediction'); 
ylabel('ground truth'); 
legend('Ground truth v.s prediction','optimal prediction')




figure (6) 
plot(test_Y,test_prediction,'bo');
hold on;
plot(-1:6,-1:6,'r');

title('Groud truth v.s predction (test data)'); 
xlabel('prediction'); 
ylabel('ground truth'); 
legend('Ground truth v.s prediction','optimal prediction')
