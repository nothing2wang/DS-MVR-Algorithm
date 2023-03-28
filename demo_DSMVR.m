% this code is a demo for nonnegative CP decomposition by DS-MVR algorithm


clear; clc; close all

load('Indian_pines.mat');
X_data = indian_pines;

I_vec = size(X_data);
% Problem setup
X = {};              % Input tensor
F = 20;            % Rank
bs = 10;                  % Number of fibers
iter_mttkrp = 50;         % Number of MTTKRP
num_trial=2;                % Number of trials
ops.b0_f = 0.15;             % stepsize of fixed algorithms
ops.lambda = 0.1;           % parameter of regular terms
ops.eta = 1;

X_data = normed_data(X_data, 0, 1);
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

    [ A_DS_MVR_t, MSE_DS_MVR_t, NRE_DS_MVR_t, TIME_DS_MVR_t] = DS_MVR(X_data,ops);
    MSE_DSMVR(trial,:)= MSE_DS_MVR_t;
    NRE_DSMVR(trial,:)= NRE_DS_MVR_t;
    TIME_DSMVR(trial,:)=TIME_DS_MVR_t;
    P = ktensor(A_DS_MVR_t);
    PP = tensor(P);
    A_DSM1VR = double(PP);

end

NRE_DS_MVR = mean(NRE_DSMVR,1)/prod(size(X_data));


%%%%%%%% plot
figure(101)
semilogy([0:(size(NRE_DS_MVR,2)-1)],NRE_DS_MVR,'-p','linewidth',1.5,'color', [0,0,1],'MarkerIndices', 1:10:50);
legend('DS-MVR');
xlabel('Number of MTTKRP computed')
ylabel('LOSS');
axis([0 50, 2*10^(-4) 10^(-1)]);
set(gca,'fontsize',14);
grid on 
