clc 
clear all


[train_Y,train_prediction,test_Y,test_prediction] = ModelInfer_I_O_HMM_one_input('I_O_HMM_one_input/I_O_HMM_T3_B4_STD.mat',3,4);
MAE_train=sum(abs(train_Y'-train_prediction))/length(train_prediction)
MAE_test=sum(abs(test_Y'-test_prediction))/length(test_prediction)
max_test=max(test_Y)
min_test=min(test_Y)
range_test=max_test-min_test



%%% model averge %%% 
% [train_Y1,train_prediction1,test_Y1,test_prediction1] = ModelInfer_I_O_HMM_one_input('I_O_HMM_one_input/I_O_HMM_T3_B4_STD.mat',2,4);
% [train_Y2,train_prediction2,test_Y2,test_prediction2] = ModelInfer_I_O_HMM_one_input('I_O_HMM_one_input/I_O_HMM_T3_B3_STD.mat',2,3);
% [train_Y3,train_prediction3,test_Y3,test_prediction3] = ModelInfer_I_O_HMM_one_input('I_O_HMM_one_input/I_O_HMM_T3_B2_STD.mat',2,2);
% [train_Y4,train_prediction4,test_Y4,test_prediction4] = ModelInfer_I_O_HMM_one_input('I_O_HMM_one_input/I_O_HMM_T3_B5_STD.mat',2,5);
% 
% 
% MAE_train1=sum(abs(train_Y1'-train_prediction1))/length(train_prediction1);
% MAE_train2=sum(abs(train_Y2'-train_prediction2))/length(train_prediction2);
% MAE_train3=sum(abs(train_Y3'-train_prediction3))/length(train_prediction3);
% MAE_train4=sum(abs(train_Y4'-train_prediction4))/length(train_prediction4);
% MAE_sum=MAE_train1+MAE_train2+MAE_train3+MAE_train4; 
% w1=MAE_train1/MAE_sum;
% w2=MAE_train2/MAE_sum;
% w3=MAE_train3/MAE_sum;
% w4=MAE_train4/MAE_sum;


% test_prediction=(w1*test_prediction1+w2*test_prediction2+w3*test_prediction3+w4*test_prediction4); 


%MAE_train=sum(abs(train_Y'-train_prediction))/length(train_prediction)
% MAE_test=sum(abs(test_Y'-test_prediction))/length(test_prediction)
% max_test=max(test_Y)
% min_test=min(test_Y)
% range_test=max_test-min_test











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
