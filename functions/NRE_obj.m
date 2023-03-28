function outs = NRE_obj(XX,PP,A,ops)

dim = length(size(XX));
loss_obj =0.5* norm(XX(:) - PP(:))^2;
reg_obj=zeros(1,dim);
lambda = ops.lambda;
for i=1:dim   
    switch ops.constraint{i}
        case 'nn' %'nn' for nonegative;
            reg_obj(i)=0;
        case 'simplex_col' % for simplex constraints with each column
            reg_obj(i)=0;
        case 'simplex_row' % for simplex constraints with each row
            reg_obj(i)=0;
        case 'l1' % for L1 norm; 
            reg_obj(i)=lambda*norm(A{i},1);
        case 'l1n' % for L1 norm + nonegative
            reg_obj(i)=lambda*norm(A{i},1);
        case 'l2' % for L2 norm 
            reg_obj(i)=lambda*norm(A{i},2)^2;
        case 'l2n' % for L2 norm + nonegative
            reg_obj(i)=lambda*norm(A{i},2)^2;
        case 'l2-bound' % for ball constraints
           return;
        case 'l2-boundn'  % for ball constraints + nonnegative
            reg_obj(i)=0;
        case 'l0' % for L0 norm
            reg_obj(i)=0;
         case 'nc' % for no constraint
            reg_obj(i)=0;
    end
end

outs = loss_obj+sum(reg_obj);