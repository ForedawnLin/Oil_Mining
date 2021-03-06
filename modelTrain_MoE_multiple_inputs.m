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
A=1;B=2;C=3;D=4;E=5;F=6;G=7;H=8;  %%% input 
I=9; K=10%%% hidden units 
J=11; %%% output 

n_node=11; 
ns=ones(1,n_node); 
ns(I)=3; %%% hidden state num


dag=zeros(n_node); 
dag([A B C D E F G H],K)=1;
dag(K,I)=1; 
dag([A B C D E F G H],J)=1;
dag(I,J)=1; 



bnet=mk_bnet(dag,ns,'discrete',[I],'observed',[A B C D E F G H J]); 
seed=0; 
rand('state',seed); 


bnet.CPD{A}=root_CPD(bnet,A);
bnet.CPD{B}=root_CPD(bnet,B);
bnet.CPD{C}=root_CPD(bnet,C);
bnet.CPD{D}=root_CPD(bnet,D);
bnet.CPD{E}=root_CPD(bnet,E);
bnet.CPD{F}=root_CPD(bnet,F);
bnet.CPD{G}=root_CPD(bnet,G);
bnet.CPD{H}=root_CPD(bnet,H);


bnet.CPD{I}=softmax_CPD(bnet,I,'clamped',0, 'max_iter', 10);
bnet.CPD{K}=gaussian_CPD(bnet,K);  
bnet.CPD{J}=gaussian_CPD(bnet,J,'mean',[0 0 0],'cov_type','diag');



%%% obtain data for BN %%%
samples=cell(n_node,n_sample_train);
for i=1:n_sample_train
    samples([1:input_dim],i)=num2cell(train_feature(i,:));
    samples([n_node],i)={Y_train(i,:)};
end 


%% train BNT 
engine=jtree_inf_engine(bnet); 
[bnet2,LLtrace]=learn_params_em(engine,samples,max_iter,epsilon);



 
% save('EOMHierarchical_one_input/bnet2_B8_D8.mat','bnet2');
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

