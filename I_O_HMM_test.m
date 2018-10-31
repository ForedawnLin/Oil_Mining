clc 
clear all 

FILE=load('filtered_matrix.mat');
feature_matrix=FILE.features_goodPCed;
FILE2=load('filtered_Y.mat');
Y=FILE2.Y;


%%% training settings 
max_iter=100;

%%% create train/test 
matrix_size=size(feature_matrix); 
data_num=matrix_size(1); 
train_test_ratio=4; 
number=floor(data_num*train_test_ratio/(train_test_ratio+1)); 
train_data=feature_matrix(1:number,:)/100;
Y_train=Y(1:number);
test_data=feature_matrix(number+1:end,:)/100;
Y_test=Y(number+1:end,:);
n_sample=number; %%% BNT sample number 







%%% I/O HMM structure %%% 
A=1;B=2;C=3;  
n_nodes=3; 
intra=zeros(n_nodes); 
intra(A,[B,C])=1;
intra(B,C)=1; 
ns=ones(1,n_nodes); 
ns(A)=9;
ns(B)=4; 
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
T=5; 
cases=cell(1,number-T); 
for i=1:number-T 
    cases{i}=cell(n_nodes,T);
    for j=1:T
        cases{i}(A,j)={train_data(i+j-1,:)'}; 
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
[bnet2,LLtrace]=learn_params_dbn_em(engine,cases,'max_iter',max_iter,'thresh',0.001); 


%%%%%%%%% inference %%%%%%% 
%%% input data %%%
evidence=cell(3,T);
evidence(1,:)=cases{1}(1,:);
%evidence(2,1:4)={[1],[2],[3],[2]};
evidence(3,:)=cases{1}(3,:);
[engine,ll]=enter_evidence(engine,evidence);
%%% inference %%%
marg=marginal_nodes(engine,2,5); %%% node_num, time slice 
marg.T
%%% To be continued: get input at t+1, and discrete node at t+1, then infer
%%% output 
%%% 


