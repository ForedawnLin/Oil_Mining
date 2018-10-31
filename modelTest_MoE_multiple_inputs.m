clc
clear all
FILE=load('filtered_matrix.mat');
feature_matrix=FILE.features_goodPCed;
FILE2=load('filtered_Y.mat');
Y=FILE2.Y;




%%% train/test 
matrix_size=size(feature_matrix); 
data_num=matrix_size(1); 
feature_num=matrix_size(2);
train_test_ratio=4; 
number=floor(data_num*train_test_ratio/(train_test_ratio+1)); 
train_data=feature_matrix(1:number,:)/100;
test_data=feature_matrix(number+1:end,:)/100;

n_sample=number; %%% BNT sample number 




%%% create BN %%%
A=1;B=2;C=3;D=4;E=5;F=6;G=7;H=8;I=9; %%% inputs 
J=10; %%% output 
K=11; %%% hidden units 


n_node=11; 
ns=ones(1,n_node);  
ns(K)=3; 

dag=zeros(n_node); 
dag([A B C D E F G H I],J)=1; 
dag([A B C D E F G H I],K)=1;
dag(K,J)=1; 
% dag(A,C)=1;
% dag(B,C)=1; 

bnet=mk_bnet(dag,ns,'discrete',[K],'observed',[A B C D E F G H I]); 
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
bnet.CPD{I}=root_CPD(bnet,I);



bnet.CPD{K}=softmax_CPD(bnet,K,'clamped',0, 'max_iter', 10);
%bnet.CPD{C}=gaussian_CPD(bnet,C,'mean',[0 0 0],'cov','diag');  
bnet.CPD{J}=gaussian_CPD(bnet,J,'mean',[0 0 0],'cov_type','diag');



%%% obtain data for BN %%%
samples=cell(n_node,n_sample);
for i=1:n_sample
%      node1_data=feature_matrix(i,:)';
%     node3_data=Y(i,:);
%     samples([1 3],i)={[node1_data];[node3_data]};
      samples([1 2 3 4 5 6 7 8 9 10],i)=num2cell([feature_matrix(i,:)';Y(i)]);
end 

% 
%% train BNT 
engine=jtree_inf_engine(bnet); 
max_iter=1000; 
epsilon=0.0000001;
[bnet2,LLtrace]=learn_params_em(engine,samples,max_iter,epsilon);




% 
% bnet2=load('bnet2.mat'); 
% bnet2=bnet2.bnet2; 
%% Inferene 
engine = jtree_inf_engine(bnet2); 
evidence=cell(1,n_node);
% n_test_sample=1000;
% x2=rand(1,n_test_sample)*20;
for i=1:number
    for j=1:feature_num
        evidence{j}=train_data(i,j);
    end
    [engine,ll]=enter_evidence(engine,evidence); 
    marg=marginal_nodes(engine,J);
    %mpe=find_mpe(engine,evidence);
    Y_pred(i)=marg.mu;
    y_eval(i)=Y(i);
end 


%%%% plot 
figure (1) 
plot(1:3277,Y_pred);
hold on 
plot(1:3277,y_eval,'r');

