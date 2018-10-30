### Summary
This dataset is about a flotation plant which is a process used to concentrate the iron ore. This process is very common in a mining plant.


### Goal of this dataset
The target is to predict the % of Silica in the end of the process, which is the concentrate of iron ore and its impurity (which is the % of Silica).


### Why prediction is needed
Although the % of Silica is measured (last column), its a lab measurement, which means that it takes at least one hour for the process engineers to have this value. So if it is posible to predict the amount of impurity in the process. 


### Original data set website 
https://www.kaggle.com/edumagalhaes/quality-prediction-in-a-mining-process 

### Oiginal data set
MiningProcess_Flotation_Plant_Database.csv




Above are the official description 
-----------------------------------------------------------------------------------------------------------
Below are the data processing by Tong 


### "time.mat" and "Mining.mat" description 
1. In order to speed up data loading, I saved the "MiningProcess_Flotation_Plant_Database.csv" into the two ".mat" format. 
2. "time.mat" includes the time (first coln in .csv). I convert the data in ".csv" and save it in "time.mat" so that I can perform grouping. 
3. "Mining.mat" includes the data of the rest of the colns in .csv. These data are features and % of Silica (last coln).


### Data pre-processing 
Since there are more than 700,000 data points, I have pre-processed the data to reduce the number. A lot of data are collected at the same time. I averaged all the data that are collected at the same time. The collected data have 4097 data points.  


### run "mining.m" to get processed data  
1. "mining.m" does the data pre-procesing descirbed above. 
2. "FM_meaned" is the processed data in "Mining.mat", which has 4097 data points. 
3. Now you can simply treat the time as a 4097 dimensional vector. The time step is 1. Thus, time=1:1:4097 
   

