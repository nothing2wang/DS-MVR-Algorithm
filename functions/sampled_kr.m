function H = sampled_kr(A,factor_idx)
list = length(A):-1:1;

H = A{list(1)}(factor_idx(:,list(1)),:);
for i = list(2:end)
    H = H.* A{i}(factor_idx(:,i),:);
end
end