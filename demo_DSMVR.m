% this code is a demo for nonnegative CP decomposition by DS-MVR algorithm


clear; clc; close all

load('Indian_pines.mat');
X_data = indian_pines;

I_vec = size(X_data);
% Problem setup
X = {};              % Input tensor
F = 30;            % Rank
bs = 10;                  % Number of fibers
iter_mttkrp = 50;         % Number of MTTKRP
num_trial=2;                % Number of trials
ops.b0_f = 0.15;             % stepsize of fixed algorithms
ops.lambda = 0.1;           % parameter of regular terms
ops.eta = 1;
X_data = mapminmax(X_data(:)', 0, 1);
X_data = reshape(X_data',I_vec(1),I_vec(2),I_vec(3));
X_data = tensor(X_data);

for trial = 1:num_trial
    disp('=============================================')

    for i = 1:length(I_vec)
        I{i} = I_vec(i);
    end
    % Generate the true latent factors
    for i=1:length(I_vec)
        A{i} = (rand(I{i},F));
    end
    A_gt = A;
    % Initialize the latent factors
    for d = 1:length(I_vec)
        Hinit{d} = rand( I{d}, F )/10;
    end
    X_data_O = double(X_data);
    %% example
    for i =1:length(I_vec)
        ops.constraint{i} = 'nn'; %'nonnegative';
    end
    ops.n_mb = bs;
    ops.A_ini = Hinit;
    ops.A_gt=A_gt; 
    ops.tol= eps^2;

    %%%%%%%%
    ops.n_mb = bs/2;
    itt = round(prod(size(X_data))/mean(size(X_data)));
    ops.max_it = iter_mttkrp*round(itt/bs);
    ops.out_iter = round(ops.max_it/50);

    [ A_DS_M1_VR_t, MSE_DS_M1_VR_t, NRE_DS_M1_VR_t, TIME_DS_M1_VR_t] = DS_MVR_re(X_data,ops);
    MSE_DSM1VR(trial,:)= MSE_DS_M1_VR_t;
    NRE_DSM1VR(trial,:)= NRE_DS_M1_VR_t;
    TIME_DSM1VR(trial,:)=TIME_DS_M1_VR_t;
    P = ktensor(A_DS_M1_VR_t);
    PP = tensor(P);
    A_DSM1VR = double(PP);

end

NRE_DS_M1VR = mean(NRE_DSM1VR,1)/prod(size(X_data));

