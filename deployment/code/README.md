### Files in the folder  
1. modelEval_I_O_HMM_one_input.m: This is the main file to run. 
2. modelInfer_I_O_HMM_one_input.m: This script contains the inference algoirthm for IOHMM model, which is called in the modelEval_I_O_HMM_one_input.m. 
3. I_O_HMM_T3_B4.mat: This is a traind IOHMM model. It uses 3 look back steps(T) and 4 hidden states(B). 
4. train_data_processed_std,mat: This is the training dataset. The dataset has been pre-preocessed so that it is standardized and only contains 8 features. 
5. test_data: This is the raw test data set，which has 22 features and 1 target value.
6. slide: This slide is from Prof.Litster. He has briefly described the work that he has been working on. 

### How to run
1. To run the algorithm, you only need to open modelEval_I_O_HMM_one_input.m and run. To uderstand the meaning of inputs and outputs, please check modelInfer_I_O_HMM_one_input.m 

