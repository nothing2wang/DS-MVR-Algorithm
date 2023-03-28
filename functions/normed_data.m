% map the dataset into [normed_low, normed_up]

function out_res =normed_data(in_res, normed_low, normed_up)

min_in = min(in_res(:));
max_in =max(in_res(:));

out_res = (normed_up-normed_low)*(in_res-min_in)/(max_in-min_in)+normed_low;