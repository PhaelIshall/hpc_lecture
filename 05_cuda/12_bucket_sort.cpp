#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void sort(int *bucket, int *key, int range) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(&bucket[key[j]], 1);
  
  __syncthreads(); //wait for buckets to be assigned 

  int sum = 0; //variable to contain bucket values
  for(int i=0;;i++){ //values in key array
    if (sum<=j){ //sum controls how much to let i increase and the offset of adding it to key array
      key[j]=i; //assign values one by one, starting from i=0. Stop adding if the bucket is empty
      sum+=bucket[i]; 
      bucket[i]--; 
    }
    else{
      return;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  //change key declaration
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  //same initialization code
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

 //change bucket declaration
 int *bucket;
 cudaMallocManaged(&bucket, range*sizeof(int));
 for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  //call sort on n threads
  sort<<<1,n>>>(bucket,key, range);
  cudaDeviceSynchronize(); //does not work without waiting here

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
