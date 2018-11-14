function [Y_in_train,train_prediction,Y_in_test,test_prediction,train_input_feature_set,test_input_feature_set] = ModelInfer_I_O_HMM_one_input(modelFile,nT,nB)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
%%% modelFile: DBN model file path
%%% nT: number of look back step 
%%% nB: number of discrete node states

std_test_data=1; %%% if use standlized data 


if std_test_data==0
%%%% get training data %%%%
    FILE=load('data/train_data_processed.mat');
    train_data=FILE.train_processed;
    Y_train=train_data.Y;
    train_feature=train_data.feature;
    train_feature=train_feature/100; 



    %%% get testing data %%%%
    FILE2=load('data/test_data.mat');
    test_data=FILE2.test_data; 
    Y_test=test_data(:,end);


    %%% Apply basis (PCA) on test_feature
    test_feature=test_data(:,train_data.feature_num)-train_data.feature_mean;  %%% get selected features and substract train data mean 
    test_feature=test_feature*train_data.PCs; %%% apply PCA basis 
    test_feature=test_feature/100; 
elseif std_test_data==1
    
    %%%% get std training data %%%%
    FILE=load('data/train_data_processed_std.mat');
    train_data=FILE.train_processed;
    Y_train=train_data.Y;
    train_feature=train_data.feature;



    %%% get testing data %%%%
    FILE2=load('data/test_data.mat');
    test_data=FILE2.test_data; 
    Y_test=test_data(:,end);


    %%% Apply std and basis (PCA) on test_feature
    test_feature=(test_data(:,train_data.feature_num)-train_data.feature_mean)./train_data.feature_std;  %%% get selected features and std train data mean 
    test_feature=test_feature*train_data.PCs; %%% apply PCA basis
else 
        %%%% get std training data %%%%
    FILE=load('data/train_data_processed_PCA_before_std.mat');
    train_data=FILE.train_processed;
    Y_train=train_data.Y;
    train_feature=train_data.feature;



    %%% get testing data %%%%
    FILE2=load('data/test_data.mat');
    test_data=FILE2.test_data; 
    Y_test=test_data(:,end);


    %%% Apply basis (PCA) and std on test_feature
    test_feature_PCed=test_data(:,train_data.feature_num)*train_data.PCs; %%% apply PCA basis      
    test_feature=(test_feature_PCed-train_data.feature_mean)./train_data.feature_std;  %%% get selected features and std train data mean  
end 




train_data_SIZE=size(train_feature); 
n_sample_train=train_data_SIZE(1); 
input_dim=train_data_SIZE(2); 
test_data_SIZE=size(test_feature); 
n_sample_test=test_data_SIZE(1); 

%%% training settings 
T=nT;  %%% look back step 
max_iter=1000; %%% max iter to train 
thresh_em=0.0000001; %%% EM threshold


%%% I/O HMM structure %%% 
A=1;B=2;C=3;  
n_nodes=3; 
intra=zeros(n_nodes); 
intra(A,[B,C])=1;
intra(B,C)=1; 
ns=ones(1,n_nodes); 
ns(A)=input_dim;
ns(B)=nB; 
dNodes=B; 
oNodes=[A C];
inter=zeros(n_nodes); 
inter(B,B)=1; 

%%% define CPDs for two-slice nodes and tie parameters %%%%
eclass1=[1 2 3]; 
eclass2=[1 4 3]; 
elcass=[eclass1 eclass2]; 
bnet=mk_dbn(intra,inter,ns,'discrete',dNodes,'observed',oNodes,'eclass1',eclass1,'eclass2',eclass2); 

bnet.CPD{1}=gaussian_CPD(bnet,A,'cov_type','diag'); 
%bnet.CPD{1}=root_CPD(bnet,A);
bnet.CPD{2}=softmax_CPD(bnet,B,'clamped',0,'max_iter',10);
bnet.CPD{3}=gaussian_CPD(bnet,C,'cov_type','diag');
bnet.CPD{4}=softmax_CPD(bnet,5,'clamped',0,'max_iter',10);


%data=sample_dbn(bnet,5) 


% 
%%% create sample %%%

cases=cell(1,n_sample_train-T); 
for i=1:n_sample_train-T 
    cases{i}=cell(n_nodes,T);
    for j=1:T
        cases{i}(A,j)={train_feature(i+j-1,:)'}; 
        cases{i}(C,j)={Y_train(i+j-1)};
    end 
end 

% T=5;
% ncases=2;
% cases=cell(1,ncases);
% for i=1:ncases 
%     ev=sample_dbn(bnet,T); 
%     cases{i}=cell(3,T); 
%     cases{i}(1,:)=ev(1,:); 
%     cases{i}(3,:)=ev(3,:);  
% end 



%%% train DBN 
%engine = jtree_dbn_inf_engine(bnet); 
%engine = jtree_unrolled_dbn_inf_engine(bnet,T); 
%engine = hmm_inf_engine(bnet,T);

engine=smoother_engine(jtree_2TBN_inf_engine(bnet));
%[bnet2,LLtrace]=learn_params_dbn_em(engine,cases,'max_iter',max_iter,'thresh',thresh_em); 


FILE=load(modelFile); 
bnet2=FILE.bnet2;
% save('I_O_HMM_one_input/I_O_HMM_T2_B4','bnet2');
% 

infer_train_Y=cell(1,T); %%% initialize train target val for inference
infer_train_Y(1,:)=cases{1}(3,:);
train_input_feature_set=zeros(n_sample_train-T,input_dim); %%% store for model stacking 
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
    train_input_feature_set(i,:)=input_feature;

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
infer_test_feature=cell(1,T); %%% initialize test feature for inference
for i=1:T 
    infer_test_feature(i)={test_feature(i,:)'};
end 

infer_test_Y=cell(1,T); %%% initialize test target val for inference
infer_test_Y(1,:)=num2cell(Y_test(1:T));



test_input_feature_set=zeros(n_sample_test-T,input_dim); %%% store for model stacking 

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
    test_input_feature_set(i,:)=input_feature; 
    
    
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
    predicted_val_test=input_feature*emission_CPD_weight+emission_CPD_mean;
    predicted_val_test_set(i)= predicted_val_test;
    %%% keep rolling the observed input data 
    infer_test_feature(1)=[]; 
    infer_test_feature{T}=test_feature(i+T,:)'; %%% get new input 
    infer_test_Y(1)=[];
    infer_test_Y{T}=predicted_val_test; %%% output output value as previously observed output    
end

Y_in_train=Y_train(T+1:end);
train_prediction=predicted_val_train_set;
Y_in_test=Y_test(T+1:end);
test_prediction=predicted_val_test_set;

end

