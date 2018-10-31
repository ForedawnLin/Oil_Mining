clc
clear all
% fileName='MiningProcess_Flotation_Plant_Database.csv'; 
% M=readtable(fileName); 
% 
% %%%% organize data %%% 
% table_size=size(M);
% X_feature=table_size(2)-1; %%% didn't include data (first coln) 
% data_num=table_size(1); %%% total number of data 



%%%% load feature matrix %%%%
FILE=load('data/Mining_raw_data.mat'); 
feature_matrix=FILE.feature_matrix; %%% 23 colns feature matrix (the last coln is the target value) 
FILE2=load('data/time_raw_data.mat'); 
time=FILE2.time; 

%%% group data based on the same time points %%%
[index,group]=findgroups(time);

FM_meaned=splitapply(@mean,feature_matrix,index); %%% mean each group 

%%% Split into train and test set


%%% train/test 
matrix_size=size(FM_meaned); 
data_num=matrix_size(1); 
train_test_ratio=4; 
number=floor(data_num*train_test_ratio/(train_test_ratio+1)); 
train_data=FM_meaned(1:number,:);
test_data=FM_meaned(number+1:end,:);
save('data/train_data','train_data'); 
save('data/test_data','test_data');
