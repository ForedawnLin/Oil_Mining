clc
clear all

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
test_feature=test_data(:,train_data.feature_num)-train_data.feature_mean;  %%% get selected features and substract train data mean 
test_feature=test_feature/100; 



train_data_SIZE=size(train_feature); 
n_sample_train=train_data_SIZE(1); 
input_dim=train_data_SIZE(2); 
test_data_SIZE=size(test_feature); 
n_sample_test=test_data_SIZE(1); 




%%% EM train setting %%%
max_iter=1000;  %%% max iteration 
epsilon=0.0000001; %%% max error 



%%% create BN %%%
A=1;  %%% input 
B=2;D=4; %%% hidden units 
C=3;  %%% output 

n_node=4; 
ns=ones(1,n_node); 
ns(A)=input_dim; 
ns(B)=8; %%% hidden state num 
ns(D)=8; %%% hidden state num


dag=zeros(n_node); 
dag(A,[B C D])=1; 
dag(B,[C D])=1;
dag(D,C)=1; 


bnet=mk_bnet(dag,ns,'discrete',[B D],'observed',[A C]); 
seed=0; 
rand('state',seed); 

bnet.CPD{A}=gaussian_CPD(bnet,A,'cov_type','diag');
bnet.CPD{B}=softmax_CPD(bnet,B,'clamped',0, 'max_iter', 10);
%bnet.CPD{C}=gaussian_CPD(bnet,C,'mean',[0 0 0],'cov','diag');  
bnet.CPD{C}=gaussian_CPD(bnet,C,'cov_type','diag');
bnet.CPD{D}=softmax_CPD(bnet,D,'clamped',0, 'max_iter', 10);



%%% obtain data for BN %%%
samples=cell(n_node,n_sample_train);
for i=1:n_sample_train
    node1_data=train_feature(i,:)';
    node3_data=Y_train(i,:);
    samples([1 3],i)={[node1_data];[node3_data]};
end 


%% train BNT 
engine=jtree_inf_engine(bnet); 
[bnet2,LLtrace]=learn_params_em(engine,samples,max_iter,epsilon);



 
% save('bnet2.mat','bent2');
% bnet2=load('bnet2.mat'); 
% bnet2=bnet2.bnet2; 

%% Inferene 
engine = jtree_inf_engine(bnet2); 
evidence=cell(1,n_node);
% n_test_sample=1000;
% x2=rand(1,n_test_sample)*20;
for i=1:n_sample_train
    evidence{A}=[train_feature(i,:)'];
%     evidence{A}=[test_data(i,1:9)'];
    [engine,ll]=enter_evidence(engine,evidence); 
    marg=marginal_nodes(engine,C);
    %mpe=find_mpe(engine,evidence);
    Y_pred_train(i)=marg.mu;
    Y_eval_train(i)=Y_train(i);
    
end 

for i=1:n_sample_test
    evidence{A}=[test_feature(i,:)'];
%     evidence{A}=[test_data(i,1:9)'];
    [engine,ll]=enter_evidence(engine,evidence); 
    marg=marginal_nodes(engine,C);
    %mpe=find_mpe(engine,evidence);
    Y_pred_test(i)=marg.mu;
    Y_eval_test(i)=Y_test(i);
    
end 



%%%% plot %%%%
figure (1) 
plot(1:n_sample_train,Y_eval_train); 
hold on; 
plot(1:n_sample_train,Y_pred_train);



figure (2) 
plot(1:n_sample_test,Y_eval_test); 
hold on; 
plot(1:n_sample_test,Y_pred_test);

