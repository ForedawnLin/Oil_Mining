clc 
clear all
[train_Y,train_prediction,test_Y,test_prediction,train_input_feature,test_input_feature] = ModelInfer_I_O_HMM_one_input('I_O_HMM_one_input/I_O_HMM_T3_B4.mat',3,4);
MAE_train=sum(abs(train_Y'-train_prediction))/length(train_prediction)
MAE_test=sum(abs(test_Y'-test_prediction))/length(test_prediction)
max_test=max(test_Y)
min_test=min(test_Y)
range_test=max_test-min_test

%%%% collect data for model stacking %%%% 
data=struct(); 
data.train_Y=train_Y; 
data.train_prediction=train_prediction'; 
data.train_input_feature=train_input_feature;
data.test_Y=test_Y;
data.test_prediction=test_prediction';
data.test_input_feature=test_input_feature;
save("I_O_HMM_one_input/Stack_Data_train_prediction_T3_B4.mat","data")

%%%%


%%% add SVR model %%%
% SVR_train_X=predicted_val_train_set; 
% SVR_train_Y=Y_train(T+1:end); 
% 
% SVR_model=fitrsvm(SVR_train_X',SVR_train_Y,'KernelFunction','linear','BoxConstraint',20); 
% 
% SVR_train_predict=predict(SVR_model,SVR_train_X');
% 
% 
% SVR_test_X=Y_test(T+1:end);
% SVR_test_predict=predict(SVR_model,SVR_test_X);
% 
% SVR_test_Y=predicted_val_test_set;


%%% Linear Regression model 
% LR_train_X=predicted_val_train_set;
% LR_train_Y=Y_train(T+1:end); 
% LR_model=polyfit(LR_train_X',LR_train_Y,1); 
% 
% LR_train_predict=LR_model(1)*LR_train_X'+LR_model(2);
% 
% 
% LR_test_X=predicted_val_test_set;
% LR_test_Y=Y_test(T+1:end);
% LR_test_predict=LR_model(1)*LR_test_X+LR_model(2);




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

figure (6) 
plot(test_Y,test_prediction,'bo');
hold on;
plot(-1:6,-1:6,'r');

title('Groud truth v.s predction (test data)'); 
xlabel('prediction'); 
ylabel('ground truth'); 
legend('Ground truth v.s prediction','optimal prediction')
