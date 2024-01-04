%CIMML: Base code framework from LSML
clear all;
addpath(genpath('.'));
clc

[optmParameter, modelparameter] =  initialization;% parameter settings for CIMML
model_CIMML.optmParameter = optmParameter;
model_CIMML.modelparameter = modelparameter;  %cv_num
model_CIMML.tuneThreshold = 0; %0;% tune the threshold for mlc
modelparameter.repetitions = 1;
fprintf('*** run CIMML for multi-label learning with missing labels ***\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load the dataset
%load('medical.mat');

%misRate = {'0.1', '0.2', '0.3', '0.4', '0.5'};
%datasets={'medical.mat', 'chess.mat', 'chemistry.mat', 'cs.mat', 'cooking.mat', 'genbase.mat', 'rcv1subset1_top944.mat', 'rcv1subset2_top944.mat', 'rcv1subset3_top944.mat', 'CAL500.mat'};
rho = {1, 2, 8, 4, 4, 1.0, 8, 2, 1.0, 1.0};
%datasets={'yeast.mat'};
datasets = {'foodtruck.mat'};
misRate = {'0.6', '0.8'};
%loop over missing rate
for mr=1:numel(misRate)
    model_CIMML.misRate = str2double(misRate{mr}); % missing rate of positive  class labels

for dc=1:numel(datasets)
    load(datasets{dc});
    target = target'; %for foodtruck.mat
    optmParameter.rho = rho{dc};
    fprintf("Current dataset: %s\n", datasets{dc});

if exist('train_data','var')==1 %'var' means check only for variables
    data    = [train_data;test_data];
    target  = [train_target,test_target];
end
clear train_data test_data train_target test_target;

%target(target == -1) = 0;
target(target == 0) = -1;

data      = double (data);
num_data  = size(data,1);
temp_data = data + eps; 
%%sanjay: normalization - row wise?
temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
if sum(sum(isnan(temp_data)))>0
    temp_data = data+eps;
    temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
end
temp_data = [temp_data,ones(num_data,1)]; %sanjay: for bias add x0=1

rng(0); %Play around with this.
randorder = randperm(num_data);
cvResult  = zeros(16,modelparameter.cv_num); % 16: number of metric?

for i = 1:modelparameter.repetitions       
    for j = 1:modelparameter.cv_num
        fprintf('- Repetition - %d/%d,  Cross Validation - %d/%d', i, modelparameter.repetitions, j, modelparameter.cv_num);
        [cv_train_data,cv_train_target,cv_test_data,cv_test_target ] = generateCVSet( temp_data,target',randorder,j,modelparameter.cv_num );
       %size(cv_train_data)
       %size(cv_train_target)
        if model_CIMML.misRate > 0
             temptarget = cv_train_target;
             [IncompleteTarget, ~, ~, realpercent]= getIncompleteTarget(cv_train_target, model_CIMML.misRate,1); 
             fprintf('\n-- Missing rate:%.1f, Real Missing rate %.3f\n',model_CIMML.misRate, realpercent); 
        end
       %% Training
        modelCIMML  = CIMML2( cv_train_data, IncompleteTarget,optmParameter); 

       %% Prediction and evaluation
        Outputs = (cv_test_data*modelCIMML.W)';
        Pre_Labels = sign(Outputs); 

        fprintf('-- Evaluation\n');
        tmpResult = EvaluationAll(Pre_Labels,Outputs,cv_test_target');
        cvResult(:,j) = cvResult(:,j) + tmpResult;
    end
end
cvResult = cvResult./modelparameter.repetitions; %Not needed
Avg_Result      = zeros(16,2);
Avg_Result(:,1) = mean(cvResult,2);
Avg_Result(:,2) = std(cvResult,1,2);

PrintResults(Avg_Result);

filename='resultcimml.xlsx';
resultToSave = Avg_Result([1, 6, 11:16], 1 );
xlColumn = {'A', 'B', 'C', 'D', 'E', 'F', 'G'};
xlLocation = [xlColumn{mr} num2str((8*(dc-1))+1)]; 
Sheet = 'cimml-1';
xlswrite(filename, resultToSave, Sheet, xlLocation);
%writematrix(resultToSave, filename, 'Sheet', Sheet, 'Range', xlLocation);

end
end

% Thresholding, two ways
%Find a way to use targets with missing labels and make use of it.
