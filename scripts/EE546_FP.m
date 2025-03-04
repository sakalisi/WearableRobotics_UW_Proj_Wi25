%% First Trial to Use XGBoost to Predict Hip Moment from a Variety of Parameters
    % Results from Original Paper
    % ~10 features required for convergence of MAE within 5% error
    % Average MAE (0 ms): 0.06 +/- 0.02 Nm/kg
    % Average MAE (250 ms): 0.11 +/- 0.04 Nm/kg
%% Load Participant Files
clc; clear
init(); def=defaults;
SUBJECT='AB07';
AMBULATION='treadmill';
TRIAL='treadmill*.mat';
allfiles=f.fileList('Subject',SUBJECT,'Mode',AMBULATION,'Trial',TRIAL);
trials=f.EpicToolbox(allfiles);

%% Define sensors and features
window = 0.250;
signals = {'emg','gon','imu'};
signalFs = [1000 1000 200]; %Hz
extractor=FeatureExtractor('TD','true','AR','true','AROrder',6);
ytrain = [];
features = [];
%% Segment data and extract features
for ii = 1:length(trials)
    trial=trials{ii};
    Tstart = trial.conditions.trialStarts + window;
    Tend = trial.conditions.trialEnds;
    gon_deriv = [];
    for kk = 2:6
        gon_deriv = [gon_deriv, derivative(trial.gon.Header(:), table2array(trial.gon(:,kk)))];
    end
    Gon_deriv = array2table(gon_deriv, 'VariableNames', strcat("d_", trial.gon.Properties.VariableNames(2:end)));
        trial.gon = [trial.gon Gon_deriv];
    for jj = find(abs(trial.id.Header(:)-Tstart)<0.001):find(abs(trial.id.Header(:)-Tend)<0.001)
        ytrain = [ytrain; trial.id.hip_flexion_r_moment(jj)];
        EMGwin = find(abs(trial.emg.Header-(trial.id.Header(jj)-window))<0.0005):(find(abs(trial.emg.Header-(trial.id.Header(jj)))<0.0005)-1);
        EMGfeatures = extractor.extract(table2array(trial.emg(EMGwin,2:end)));
        IMUwin = find(abs(trial.imu.Header-(trial.id.Header(jj)-window))<0.0005):(find(abs(trial.imu.Header-(trial.id.Header(jj)))<0.0005)-1);
        IMUfeatures = extractor.extract(table2array(trial.imu(IMUwin,2:end)));
        GONwin = find(abs(trial.gon.Header-(trial.id.Header(jj)-window))<0.0005):(find(abs(trial.gon.Header-(trial.id.Header(jj)))<0.0005)-1);
        GONfeatures = extractor.extract(table2array(trial.gon(GONwin,2:end)));
        features = [features; EMGfeatures, IMUfeatures,GONfeatures];
    end
end

%% use TD_extract function for each window 
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

predictHipMom = xgboost_train(Xtrain,ytrain,params,max_num_iters,eval_metric,model_filename);