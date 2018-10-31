clc
clear all
FILE=load('filtered_matrix.mat');
feature_matrix=FILE.features_goodPCed;
FILE2=load('filtered_Y.mat');
Y=FILE2.Y;




%%% train/test 
matrix_size=size(feature_matrix); 
data_num=matrix_size(1); 
train_test_ratio=4; 
number=floor(data_num*train_test_ratio/(train_test_ratio+1)); 
train_data=feature_matrix(1:number,:);
test_data=feature_matrix(number+1:end,:);

n_sample=number; %%% BNT sample number 




%%% create BN %%%
A=1;B=2;C=3; 
n_node=3; 
ns=ones(1,n_node); 
ns(A)=9; 
ns(B)=3; 

dag=zeros(n_node); 
dag(A,B)=1; 
dag(A,C)=1;
dag(B,C)=1; 

bnet=mk_bnet(dag,ns,'discrete',[B],'observed',[A C]); 
seed=0; 
rand('state',seed); 

bnet.CPD{A}=root_CPD(bnet,A);
bnet.CPD{B}=softmax_CPD(bnet,B,'clamped',0, 'max_iter', 10);
%bnet.CPD{C}=gaussian_CPD(bnet,C,'mean',[0 0 0],'cov','diag');  
bnet.CPD{C}=gaussian_CPD(bnet,C,'mean',[0 0 0],'cov_type','diag');



%%% obtain data for BN %%%
samples=cell(n_node,n_sample);
for i=1:n_sample
    node1_data=feature_matrix(i,:)'/100;
    node3_data=Y(i,:);
    samples([1 3],i)={[node1_data];[node3_data]};
end 


%% train BNT 
engine=jtree_inf_engine(bnet); 
max_iter=1000; 
epsilon=0.0000001;
[bnet2,LLtrace]=learn_params_em(engine,samples,max_iter,epsilon);



%% Inferene 
engine = jtree_inf_engine(bnet2); 
evidence=cell(1,n_node);
% n_test_sample=1000;
% x2=rand(1,n_test_sample)*20;
for i=1:data_num-number
    evidence{A}=[test_data(i,:)'];
    [engine,ll]=enter_evidence(engine,evidence); 
    marg=marginal_nodes(engine,C);
    %mpe=find_mpe(engine,evidence);
    Y_pred(i)=marg.mu;
    %y_pred(i)=mpe{2};
end 



