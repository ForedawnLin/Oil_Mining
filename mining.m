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
FILE=load('Mining.mat'); 
feature_matrix=FILE.feature_matrix; %%% 23 colns feature matrix (the last coln is the target value) 
FILE2=load('time.mat'); 
time=FILE2.time; 

%%% group data based on the same time points %%%
[index,group]=findgroups(time);

FM_meaned=splitapply(@mean,feature_matrix,index); %%% mean each group 







%%%% find correlation between input features and %Silica 
covariance=cov(FM_meaned);
[R,p]=corrcoef(FM_meaned);
p_th=0.05; 
[sig_features_ind,~]=find(p(:,23)<=p_th); %%% find significantly correlated input features w.r.t output (23th coln)  
good_features=FM_meaned(:,sig_features_ind); %%% Good features
Y=FM_meaned(:,23); % Silica 

%%% PCA calcualtion to get rid of correlation btw inputs %%% 
feature_SIZE=size(good_features); 
data_number=feature_SIZE(1);
feature_mean=mean(good_features);
good_features_centered=good_features-feature_mean; 
%covariance_matrix=good_features_centered'*good_features_centered/(data_numner-1); 
[U,S,V]=svd(good_features_centered);
features_PCed=good_features_centered*V; %%% n*Num_features

singular_th=0.1*10^4;  %%% set singular value threshold (PC component signidicance); 
good_PCs_ind=find(max(S)>singular_th); 
features_goodPCed=features_PCed(:,good_PCs_ind); %%% n*Num_goodPCsFeatures  

save('filtered_matrix','features_goodPCed');
save('filtered_Y','Y');

%%%% BN setup %%%% 





%%% plot %%%
% 
% figure(1)
% Y_t1_1=Y(1:end-1);
% Y_t1_2=Y(2:end);
% scatter(Y_t1_1,Y_t1_2);
% 
% 
% figure(2)
% Y_t2_1=Y(1:end-2);
% Y_t2_2=Y(3:end);
% scatter(Y_t2_1,Y_t2_2);
% 
% 
% figure(3)
% Y_t3_1=Y(1:end-3);
% Y_t3_2=Y(4:end);
% scatter(Y_t3_1,Y_t3_2);



%%% check %%%%
% PCA_corr_check=corr(features_goodPCed); 


% scatter(log(FM_meaned(:,3)),FM_meaned(:,23));
% plot(FM_meaned(:,3),FM_meaned(:,23));




% feature_matrix_hasComma=M{:,2:end}; 
% feature_matrix_no_comma=cellfun(@(z) strrep(z,',','.'), feature_matrix_hasComma,'UniformOutput',false); %% convert comma to decimal point 
% feature_matrix=cellfun(@str2num, feature_matrix_no_comma); %% convert comma to decimal point 
