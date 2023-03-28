function [ A, MSE_A ,NRE_A, TIME_A] = DS_MVR(X,ops)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DS-MVR:  Qingsong Wang, Chunfeng Cui, Deren Han. ``Accelerated doubly stochastic gradient 
% descent for tensor CP decomposition''. Journal of Optimization Theory and Applications. Accept, 2023.
% ============== input =====================================
% X  : the data tensor
% ops: algorithm parameters
%   o 'constraint'          - Latent factor constraints
%   o 'b0'                  - Initial stepsize
%   o 'n_mb'                - Number of fibers
%   o 'max_it'              - Maximum number of iterations
%   o 'A_ini'               - Latent factor initializations
%   o 'A_gt'                - Ground truth latent factors (for MSE computation only)
%   o 'tol'                 - stopping criterion
% =============================================================
% ============= output ========================================
% A: the estimated factors
% MSE_A : the MSE of A at different iterations
% NRE_A : the cost function at different iterations
% TIME_A: the walltime at different iterations
% =============================================================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Code
% Get the algorithm parameters
dim = length(size(X));
A       = ops.A_ini;
b0      = ops.b0_f;
n_mb    = ops.n_mb;
max_it  = ops.max_it;
A_gt    = ops.A_gt;
tol     = ops.tol;
out_iter = ops.out_iter;
% Get initial parametrs
dim_vec = size(X);
PP = tensor(ktensor(A));
XX = tensor(X);
err_e = 0.5*norm(XX(:) - PP(:),2)^2;
NRE_A(1) = (err_e);
MSE_A(1)=0;
for dim_i=1:dim
    MSE_A(1)=MSE_A(1)+MSE_measure(A{dim_i},A_gt{dim_i});
end
MSE_A(1) = (1/dim)*MSE_A(1);
mm = 1;
a=tic;
TIME_A(1)=toc(a);
A_old = A;
A_new = A;
tA = A;
% Run the algorithm until the stopping criterion
for it = 1:max_it
    % step size
    in_par = 1-0.9*(it-1)/(it+2);
    alpha = b0/((it)^(1e-6));

    if it==1 % compute the stochastic gradient for all factors
        %randomly permute the dimensions
        block_vec = randperm(dim);
        for idx=1:dim
            % select the block variable to update.
            d_update = block_vec(idx);
            % sampling fibers and forming the X_{d}=H_{d} A_{d}^t least squares
            [tensor_idx, factor_idx] = sample_fibers(n_mb, dim_vec, d_update);
            % reshape the tensor from the selected samples
            X_sample = reshape(X(tensor_idx), dim_vec(d_update), [])';
            % perform a sampled khatrirao product
            ii=1;
            for i=[1:d_update-1,d_update+1:dim]
                A_unsel{ii}= A{i};
                ii=ii+1;
            end
            H{d_update} = sampled_kr(A_unsel, factor_idx);
            Gra{d_update} = (A_old{d_update}*(H{d_update}'*H{d_update})-X_sample'*H{d_update})/n_mb;
            Dk{d_update}=Gra{d_update};  
        end
    else
        %randomly permute the dimensions
        block_vec = randperm(dim);
        % select the block variable to update.
        d_update = block_vec(1);
        % sampling fibers and forming the X_{d}=H_{d} A_{d}^t least squares
        [tensor_idx, factor_idx] = sample_fibers(n_mb, dim_vec, d_update);
        % reshape the tensor from the selected samples
        X_sample = reshape(X(tensor_idx), dim_vec(d_update), [])';
        % perform a sampled khatrirao product
        ii=1;
        for i=[1:d_update-1,d_update+1:dim]
            A_unsel{ii}= A{i};
            A_unsel_old{ii}= A_old{i};
            ii=ii+1;
        end
        H{d_update} = sampled_kr(A_unsel, factor_idx);
        H_old{d_update} = sampled_kr(A_unsel_old, factor_idx);
        Gra{d_update} = ((A{d_update}*H{d_update}')*H{d_update}-X_sample'*H{d_update})/n_mb;
        Gra_old{d_update} = ((A_old{d_update}*H{d_update}')*H{d_update}-X_sample'*H{d_update})/n_mb;
        Dk{d_update} = Gra{d_update}+(1-in_par)*(Dk{d_update}-Gra_old{d_update});
    end
    % update the factor
    tA{d_update} = A{d_update} - alpha* Dk{d_update};
    A_new{d_update} = proxr(tA{d_update}, ops, alpha, d_update); 

    A_old = A;
    A = A_new;

    % compute MSE after each MTTKRP
    if mod(it,out_iter)==0
        TIME_A(mm+1)= TIME_A(mm)+toc(a);
        MSE_A(mm+1)=0;
        for dim_i=1:dim
            MSE_A(mm+1)=MSE_A(mm+1)+MSE_measure(A{dim_i},A_gt{dim_i});
        end
        MSE_A(mm+1)=(1/dim)*MSE_A(mm+1);
        P = ktensor(A);
        PP = tensor(P);
        NRE_A(mm+1) = NRE_obj(XX,PP,A,ops);

        if abs(NRE_A(mm+1))<=tol
            break;
        end

        disp(['DS-MVR at iteration ',num2str(mm+1),' and the MSE is ',num2str(MSE_A(mm+1))])
        disp(['DS-MVR at iteration ',num2str(mm+1),' and the NRE is ',num2str(NRE_A(mm+1))])
        disp('====')
        mm = mm + 1;
        a=tic;
    end
end
end

