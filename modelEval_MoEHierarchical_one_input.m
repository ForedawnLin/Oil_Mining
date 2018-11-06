clc 
clear all
[train_Y,train_prediction,test_Y,test_prediction] = ModelInfer_MoEHierarchical_one_input('EOMHierarchical_one_input/bnet2_B4_D4_STD.mat',4,4);

MAE_train=sum(abs(train_Y-train_prediction'))/length(train_Y)
MAE_test=sum(abs(test_prediction-test_Y'))/length(test_Y)
max_test=max(test_Y)
min_test=min(test_Y)
range_test=max_test-min_test


%%%% plot %%%%
figure (1) 
plot(1:length(train_Y),train_Y,'b'); 
hold on; 
plot(1:length(train_Y),train_prediction,'r');
title('Ground truth and predction (training data)'); 
xlabel('time sequence'); 
ylabel('value'); 
legend('Ground truth','Prediction ')


figure (2) 
plot(1:length(test_Y),test_Y,'b'); 
hold on; 
plot(1:length(test_Y),test_prediction,'r');
title('Ground truth and predction (test data)'); 
xlabel('time sequence'); 
ylabel('value'); 
legend('Ground truth','Prediction ')



figure (3) 
plot(test_prediction,test_Y,'bo'); 
hold on;
plot(-1:6,-1:6,'r');
title('Ground truth v.s predction (test data)'); 
xlabel('prediction'); 
ylabel('ground truth'); 
legend('Ground truth v.s prediction','optimal prediction')
