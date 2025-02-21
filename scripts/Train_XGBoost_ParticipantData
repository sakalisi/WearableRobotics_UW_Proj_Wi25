%% First Trial to Use XGBoost to Predict Hip Moment from a Variety of Parameters

% Import Patient Data File
% xtrain = [ ]
% ytrain = true hip moment
params = struct;
params.booster           = 'gbtree';
params.objective         = 'binary:logistic';
params.eta               = 0.05; % From Paper
params.min_child_weight  = 1;
params.subsample         = 1; % 0.9
params.colsample_bytree  = 1;
params.num_parallel_tree = 1;
params.max_depth         = 6; %From paper
num_iters                = 3;

eval_metric = 'Accuracy';

predictHipMom = xgboost_train(Xtrain,ytrain,params,max_num_iters,eval_metric,model_filename)

