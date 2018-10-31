




%%% I/O HMM structure %%% 
A=1;B=2;C=3;  
n_nodes=3; 
intra=zeros(n_nodes); 
intra(A,[B,C])=1;
intra(B,C)=1; 
ns=ones(n_nodes); 
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

bnet.CPD{1}=root_CPD(bnet,A); 
bnet.CPD{2}=softmax_CPD(bnet,B,'clamped',0,'max_iter',10);
bnet.CPD{3}=gaussian_CPD(bnet,C,'cov_type','diag');
bnet.CPD{4}=softmax_CPD(bnet,5,'clamped',0,'max_iter',10);

