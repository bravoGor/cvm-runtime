# OPs Table

IOPs(Interger operators) = Op Weight * Output Shape Size

The OPs of deterministic operator is as below:

|Op Name|Op Weight|
|---|---|
|variable|1|
|conv2d, conv2d_transpose (without bias)|$w_2*w_3*w_4*3$|
|conv2d, conv2d_transpose (with bias)|$w_2*w_3*w_4*3+1$|
|dense (without bias)|$w_{1}*3$|
|dense (with bias)|$(w_{1}+1)*3$|
|non_max_suppression|$input_{0,2}*20$|
|max_pool2d|$poolsize^2$|
|\_\_add_scalar\_\_|1|
|\_\_add_symbol\_\_|1|
|\_\_div_scalar\_\_|1|
|\_\_div_symbol\_\_|1|
|\_\_equal_symbol\_\_|1|
|\_\_greater_equal_symbol\_\_|1|
|\_\_greater_symbol\_\_|1|
|\_\_left_shift_symbol\_\_|1|
|\_\_less_equal_symbol\_\_|1|
|\_\_less_symbol\_\_|1|
|\_\_lshift_scalar\_\_|1|
|\_\_max_symbol\_\_|1|
|\_\_min_symbol\_\_|1|
|\_\_mod_symbol\_\_|1|
|\_\_mul_scalar\_\_|1|
|\_\_mul_symbol\_\_|1|
|\_\_not_equal_symbol\_\_|1|
|\_\_pow_scalar\_\_|1|
|\_\_rdiv_scalar\_\_|1|
|\_\_right_shift_symbol\_\_|1|
|\_\_rshift_scalar\_\_|1|
|\_\_rsub_scalar\_\_|1|
|\_\_sub_scalar\_\_|1|
|\_\_sub_symbol\_\_|1|
|\_\_undef\_\_|1|
|abs|1|
|add|1|
|argmax|1|
|argmin|1|
|broadcast_add|1|
|broadcast_div|1|
|broadcast_equal|1|
|broadcast_greater|1|
|broadcast_greater_equal|1|
|broadcast_left_shift|1|
|broadcast_less|1|
|broadcast_less_equal|1|
|broadcast_max|1|
|broadcast_min|1|
|broadcast_mod|1|
|broadcast_mul|1|
|broadcast_not_equal|1|
|broadcast_right_shift|1|
|broadcast_sub|1|
|broadcast_to|1|
|cast|1|
|clip|--2|
|collapse_sum|1|
|concatenate|1|
|copy|1|
|cvm_clip|2|
|cvm_left_shift|1|
|cvm_lut|10|
|cvm_right_shift|1|
|div|1|
|elemwise_add|1|
|elemwise_div|1|
|elemwise_mod|1|
|elemwise_mul|1|
|elemwise_sub|1|
|elemwise_sum|1|
|expand_dims|1|
|expand_like|1|
|flatten|1|
|flip|1|
|full|1|
|full_like|1|
|gather_nd|1|
|get_valid_counts|1|
|global_max_pool2d|1|
|greater|1|
|less|1|
|log|1|
|log2|1 (constant time ~64)|
|logical_and|1|
|logical_not|1|
|logical_or|1|
|matmul|1|
|max|1|
|mean|1|
|min|1|
|multiply|1|
|negative|1|
|nn.relu|1|
|ones|1|
|ones_like|1|
|prod|1|
|relu|1|
|repeat|1|
|reshape|1|
|reshape_like|1|
|sigmoid|1|
|slice|1|
|slice_like|5|
|split|1|
|sqrt|1|
|squeeze|1|
|strided_slice|5|
|subtract|1|
|sum|$x_3*x_4$|
|take|10|
|tanh|1|
|tile|5|
|transpose|5|
|upsampling|1|
|where|1|
|zeros|1|
|zeros_like|1|

