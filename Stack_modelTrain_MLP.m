clc
clear all 

data=load('I_O_HMM_one_input/Stack_Data_train_prediction_T3_B4.mat'); 
data=data.data; 

%%%% get train and test data for NN %%%%
train_Y=data.train_Y;
train_predict=data.train_prediction; 
train_feature=data.train_input_feature; 

test_Y=data.test_Y;
test_predict=data.test_prediction; 
test_feature=data.test_input_feature; 

train_residual=train_Y-train_predict;  %%% train target 
test_residual=test_Y-test_predict;  %%% test target 


%%% shuffle data %%% 
SIZE=size(train_feature);
input_dim=SIZE(2);
train_num=SIZE(1);
INDEX=randperm(train_num); 
train_feature=train_feature(INDEX,:);
train_residual=train_residual(INDEX);




train_feature_cell=cell(train_num,1);
for i=1:train_num
    train_feature_cell{i}=train_feature(i,:)';
end


SIZE=size(test_feature);
test_num=SIZE(1);
test_feature_cell=cell(test_num,1);
for i=1:test_num
    test_feature_cell{i}=test_feature(i,:)';
end 



%%% MLP model %%%
layer = sequenceInputLayer(input_dim,'Name','residual_model'); 
layers = [ ...
    layer
    fullyConnectedLayer(30)
    leakyReluLayer
    fullyConnectedLayer(10)
    leakyReluLayer 
    fullyConnectedLayer(1)
    regressionLayer];

maxEpochs = 100000;
miniBatchSize = 64;

options = trainingOptions('adam',...
    'ExecutionEnvironment', 'cpu',...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',miniBatchSize,...
    'Plots','none')

net = trainNetwork(train_feature_cell,num2cell(train_residual),layers,options);





