function [result]=Pred_I_O_HMM(bnet,nT,B,data)

infer_train_Y=cell(1,nT); %%% initialize train target val for inference
infer_train_Y(1,:)=data{1}(3,:); %%% big assumption: know first T data points 
predicted_val_train_set=zeros(length(data)-1,1); %%% init prediction result; m*1 data points 
engine=smoother_engine(jtree_2TBN_inf_engine(bnet));

for i=1:length(data)-1 %%(length(data)-1?)
%%%%%%%%% inference %%%%%%% 
%%% input data %%%
    evidence=cell(3,nT);
    evidence(1,:)=data{i}(1,:); %%% no need to update input feature 
    evidence(3,:)=infer_train_Y(1,:);
    [engine,ll]=enter_evidence(engine,evidence);
%%% inference %%%
    marg=marginal_nodes(engine,B,nT); %%% node_num, time slice 


%%%%%%%% prediction %%%%%%%%% 
    input_feature=data{i+1}{1}';  %%% 1*m, will be input
    [~,T_th_state]=max(marg.T); %%% choose the T_th state 
    %%% get the corresponding CPD
    softmax_node_CPD= struct(bnet.CPD{4});  
    softmax_set=softmax_node_CPD.glim{T_th_state}; 

    %%%calculate prediced state numb at t+1 
    softmax_element_values=input_feature*softmax_set.w1+softmax_set.b1;%%% no need to use full expression of softmax. since monotonic increase for each element and we only need the max  
    [~,predicted_state]=max(softmax_element_values); 

    %%% calculate predicted output  
    emission_CPD_set=struct(bnet.CPD{3}); 
    emission_CPD_mean=emission_CPD_set.mean(predicted_state); % 1*1 
    emission_CPD_weight=emission_CPD_set.weights(:,:,predicted_state)'; %%% m*1
    predicted_val_train=input_feature*emission_CPD_weight+emission_CPD_mean;
    predicted_val_train_set(i)=predicted_val_train; %%% results  
    %%% keep rolling the observed input data 
    infer_train_Y(1)=[];
    infer_train_Y{nT}=predicted_val_train; %%% output output value as previously observed output 
end
result=predicted_val_train_set;
end