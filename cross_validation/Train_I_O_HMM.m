function [model]=Train_I_O_HMM(A,B,C,nT,nB,n_nodes,input_dim,train_data)
    for nT=nT:nT  %%% grid search for time step 
        for nB=nB:nB %%% grid search for hidden state choices 

            %%% training settings 
            T=nT;  %%% look back step 
            max_iter=1000; %%% max iter to train 
            thresh_em=0.01; %%% EM threshold


            %%% I/O HMM structure %%% 
            A=A;B=B;C=C;
            n_nodes=n_nodes;
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

                    
            engine=smoother_engine(jtree_2TBN_inf_engine(bnet));
            [bnet2,LLtrace]=learn_params_dbn_em(engine,train_data,'max_iter',max_iter,'thresh',thresh_em); 
            model=bnet2;
        end 
    end 
end