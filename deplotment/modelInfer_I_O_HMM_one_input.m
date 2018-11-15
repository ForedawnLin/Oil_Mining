function [Y_in_train,train_prediction,Y_in_test,test_prediction] = ModelInfer_I_O_HMM_one_input(modelFile,train_Data,test_Data,nT,nB)
%%% modelFile: DBN model file path
%%% input: 
%%% train_Data: path for training data 
%%% test_Data: path for test data 
%%% nT: number of look back step 
%%% nB: number of discrete node states

%%% output: 
%%% Y_in_train: ground truth for training data  
%%% train_prediction: prediction on training data  
%%% Y_in_test: ground truth for test data 
%%% test_prediction: prediction on test data 
    

%%%% get standardized training data %%%%
FILE=load(train_Data);
train_data=FILE.train_processed;
Y_train=train_data.Y;
train_feature=train_data.feature;



%%% get testing data %%%%
FILE2=load(test_Data);
test_data=FILE2.test_data; 
Y_test=test_data(:,end);


%%% Apply standardization and basis (PCA) on test_feature
test_feature=(test_data(:,train_data.feature_num)-train_data.feature_mean)./train_data.feature_std;  %%% get selected features and std train data mean 
test_feature=test_feature*train_data.PCs; %%% apply PCA basis



%%% get basic data information 
train_data_SIZE=size(train_feature); 
n_sample_train=train_data_SIZE(1); 
test_data_SIZE=size(test_feature); 
n_sample_test=test_data_SIZE(1); 


%%% model setup
T=nT;  %%% look back step 

%%% I/O HMM structure %%% 
A=1;B=2;C=3;  
n_nodes=3;  

%%% create sample %%%

cases=cell(1,n_sample_train-T); 
for i=1:n_sample_train-T 
    cases{i}=cell(n_nodes,T);
    for j=1:T
        cases{i}(A,j)={train_feature(i+j-1,:)'}; 
        cases{i}(C,j)={Y_train(i+j-1)};
    end 
end 


%%% Load trained model and specifiy algorithm for inference %%%
FILE=load(modelFile); 
bnet2=FILE.bnet2;
engine=smoother_engine(jtree_2TBN_inf_engine(bnet2)); %%% use junction tree algorithm 


%%% create a suitable format to load train data for inference  
infer_train_Y=cell(1,T); %%% initialize train target val for inference
infer_train_Y(1,:)=cases{1}(3,:);

predicted_val_train_set=zeros(n_sample_train-T,1); %%% %%% initilize result varibale for test data 

%%% evaluate algorithm on training data %%%
for i=1:n_sample_train-T
%%%%%%%%% inference %%%%%%% 
%%% input data %%%
    evidence=cell(3,T);
    evidence(1,:)=cases{i}(1,:); %%% no need to update input 
    evidence(3,:)=infer_train_Y(1,:);
    [engine,ll]=enter_evidence(engine,evidence);
%%% inference %%%
    marg=marginal_nodes(engine,B,T); %%% node_num, time slice 


%%%%%%%% prediction %%%%%%%%% 
    input_feature=train_feature(i+T,:);  %%% 1*m, will be input
    [~,T_th_state]=max(marg.T); %%% choose the T_th state 
    %%% get the corresponding CPD
    softmax_node_CPD= struct(bnet2.CPD{4});  
    softmax_set=softmax_node_CPD.glim{T_th_state}; 

    %%%calculate prediced state numb at t+1 
    softmax_element_values=input_feature*softmax_set.w1+softmax_set.b1;%%% no need to use full expression of softmax. since monotonic increase for each element and we only need the max  
    [~,predicted_state]=max(softmax_element_values); 

    %%% calculate predicted output  
    emission_CPD_set=struct(bnet2.CPD{3}); 
    emission_CPD_mean=emission_CPD_set.mean(predicted_state); % 1*1 
    emission_CPD_weight=emission_CPD_set.weights(:,:,predicted_state)'; %%% m*1
    predicted_val_train=input_feature*emission_CPD_weight+emission_CPD_mean;
    predicted_val_train_set(i)=predicted_val_train; %%% results  
    %%% keep rolling the observed input data 
    infer_train_Y(1)=[];
    infer_train_Y{T}=predicted_val_train; %%% output output value as previously observed output 
end 



%%% Big assumption: we know the first T points in test data set 
%%% create a suitable format to load test data for inference 
infer_test_feature=cell(1,T); %%% initialize test feature for inference
for i=1:T 
    infer_test_feature(i)={test_feature(i,:)'};
end
infer_test_Y=cell(1,T); %%% initialize test target val for inference
infer_test_Y(1,:)=num2cell(Y_test(1:T));


%%%%%%% test data prediciton %%%%%%%%%
predicted_val_test_set=zeros(n_sample_test-T,1); %%% initilize result varibale for test data 
for i=1:n_sample_test-T
%%%%%%%%% inference %%%%%%% 
  %%% input data %%%
    evidence=cell(3,T);
    evidence(1,:)=infer_test_feature; %%% no need to update input 
    evidence(3,:)=infer_test_Y;
    [engine,ll]=enter_evidence(engine,evidence);
  %%% inference %%%
    marg=marginal_nodes(engine,B,T); %%% node_num, time slice 
    input_feature=test_feature(i+T,:);  %%% 1*m, will be input     
    [~,T_th_state]=max(marg.T); %%% choose the T_th state 
    %%% get the corresponding CPD
    softmax_node_CPD= struct(bnet2.CPD{4});  
    softmax_set=softmax_node_CPD.glim{T_th_state}; 

    %%%calculate prediced state number at t+1 
    softmax_element_values=input_feature*softmax_set.w1+softmax_set.b1;%%% no need to use full expression of softmax. since monotonic increase for each element and we only need the max  
    [~,predicted_state]=max(softmax_element_values); 

    %%% calculate predicted output  
    emission_CPD_set=struct(bnet2.CPD{3}); 
    emission_CPD_mean=emission_CPD_set.mean(predicted_state); % 1*1 
    emission_CPD_weight=emission_CPD_set.weights(:,:,predicted_state)'; %%% m*1
    predicted_val_test=input_feature*emission_CPD_weight+emission_CPD_mean;
    predicted_val_test_set(i)= predicted_val_test;
    %%% keep rolling the observed input data 
    infer_test_feature(1)=[]; 
    infer_test_feature{T}=test_feature(i+T,:)'; %%% get new input 
    infer_test_Y(1)=[];
    infer_test_Y{T}=predicted_val_test; %%% output output value as previously observed output    
end

%%% output %%%%
Y_in_train=Y_train(T+1:end);
train_prediction=predicted_val_train_set;
Y_in_test=Y_test(T+1:end);
test_prediction=predicted_val_test_set;

end

