extern "C"{
void log(const int *x, int *y, const int n){
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  //int tid = threadIdx.x + blockIdx.x * blockDim.x;
  //for(int j = tid; j < n; j += gridDim.x * blockDim.x){
  for(int j = 0; j < n; j++){
#pragma HLS PIPELINE
    const int x_val = x[j];
    y[j] = 64;
    for(int i = 1; i < 64; i++){
      int tmp = (int)1 << i;
      if(abs(x_val) < tmp){
        y[j] = i;
        break;
      }
    }
  }
}
}
