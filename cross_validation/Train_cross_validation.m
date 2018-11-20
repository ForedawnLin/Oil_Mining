clc 
clear all 

%%%% data pre process %%%%
File=load('../data/train_data_processed_std.mat'); 
train_processed=File.train_processed; 

%%%% get std training data %%%%
%FILE=load('data/train_data_processed_std.mat');
train_data=train_processed;%FILE.train_processed;
Y_train=train_data.Y;
train_feature=train_data.feature;
train_data_SIZE=size(train_feature); 
n_sample_train=train_data_SIZE(1); 
input_dim=train_data_SIZE(2); 


nT=3;
nB=4; 
n_nodes=3; 
A=1;B=2;C=3;  
       
       

%%% make appropriate format of train data for trainig  
cases=cell(1,n_sample_train-nT); 
for i=1:n_sample_train-nT 
    cases{i}=cell(n_nodes,nT);
    for j=1:nT
        cases{i}(A,j)={train_feature(i+j-1,:)'}; 
        cases{i}(C,j)={Y_train(i+j-1)};
    end 
end 



%%% cross validation set %%% 
k_fold=5; 
n_seq=length(cases);
random_index=randperm(n_seq);
one_fold_num=floor(n_seq/k_fold); 
cs_data=cell(1,k_fold); 
shuffle_train_set=0; %%% 1 is to shuffle the training set 
%rand_cases=cases(random_index);
for i=1:k_fold
    if i~=k_fold 
        cs_data{i}=cases((i-1)*one_fold_num+1:i*one_fold_num); %%% one_fold_num sequences
    else 
        cs_data{i}=cases((i-1)*one_fold_num+1:end); %%% one_fold_num sequences
    end 
end 

cs_index_pool=1:k_fold; %%%% initialize cross_validaiton index pool
train_data_rand=[]; %%%% init random training sequences  
for cs_iter=1:k_fold
    train_index_pool=cs_index_pool(cs_index_pool~=cs_iter);
    train_data=[];
    for k_fold_index=1:length(train_index_pool)
        train_data=[train_data cs_data{train_index_pool(k_fold_index)}]; 
    end 
    
    if shuffle_train_set==1
        rand_index=randperm(length(train_data));
        train_data_rand=train_data(rand_index);
    else 
        train_data_rand=train_data;
    end 
    %train_data_SIZE=size(train_data);
    %n_sample_train=train_data_SIZE(1);  
    %train_feature=train_data(:,1:8);
    %Y_train=train_data(:,end); 


    valid_data=[cs_data{cs_iter}];  

    %%% validate combined model %%%
%     model=load('k_folder_models/final_model.mat');
%     model=model.f_model;
    
    
    [model]=Train_I_O_HMM(A,B,C,nT,nB,n_nodes,input_dim,train_data_rand);
    %%% save model %%%
    %savePath=['k_folder_models/' 'model' num2str(cs_iter) '.mat'];
    %save(savePath,'model')
    %%% add prediction function here %%%%
    [result]=Pred_I_O_HMM(model,nT,B,valid_data);
    valid_gt=zeros(length(valid_data)-1,1); %%% init validation gt as coln vect 
    for i=1:length(valid_gt) 
        valid_gt(i)=valid_data{i+1}{n_nodes,1};    
    end 
    MAE_valid=sum(abs(valid_gt-result))/length(result)
end 
% MAE_test=sum(abs(Y_valid(T+1:end)'-predicted_val_test_set))/length(Y_valid(T+1:end))
% 
% 
% %%%% plot %%%%
figure (3)
plot(1:length(result),valid_gt,'b');
hold on; 
plot(1:length(result),result,'r'); 
title('Groud truth and predction (training data)'); 
xlabel('time sequence'); 
ylabel('value'); 
legend('Ground truth','Prediction ')
% 
% 
% figure (4) 
% plot(1:n_sample_valid-T,Y_valid(T+1:end),'b');
% hold on; 
% plot(1:n_sample_valid-T,predicted_val_test_set,'r'); 
% title('Groud truth and predction (test data)'); 
% xlabel('time sequence'); 
% ylabel('value'); 
% legend('Ground truth','Prediction ')
% 
% 
% figure (5)
% plot(Y_train(T+1:end),predicted_val_train_set,'bo');
% 
% figure (6) 
% plot(Y_valid(T+1:end),predicted_val_test_set,'bo');
% hold on;
% plot(-1:6,-1:6,'r');
% 
% title('Groud truth v.s predction (test data)'); 
% xlabel('prediction'); 
% ylabel('ground'); 
% legend('Ground truth v.s prediction','optimal prediction')
