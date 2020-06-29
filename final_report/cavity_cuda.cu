#include<iostream>
#include <math.h> 
#include <fstream>
using namespace std;

//print functions 
void print_matrix(float *m, int x, int y){
  for (int i=0; i<x; i++){
     for (int j=0; j<y; j++){
        cout << m[i*x+j] << ", "; 
    }
  cout << "\n ";
  }
}

void print_vector(float *m, int size){
     for (int j=0; j<size; j++){
        cout << m[j] << ", "; 
    }
  cout << "\n ";
  
}

// Create a vector of evenly spaced numbers.
__global__ void range(float *r, float min, float max, int N) {
    float delta = (max-min)/float(N-1);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=N) return;
    r[i] = min + i*delta;
}


__device__ void copy_matrix(float* o, float* copy, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=size) return;
    copy[i]= o[i];
}

__global__ void pressure_poisson_conditions(float*  p,float* pn, float* b, float dx, float dy, int nx, int ny){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i>=nx*nx) return;
  if ((i+1) % nx == 0){
     p[i] = p[i-1];
  }
  if(i<nx){
     p[i] = p[i+nx];
  }
  if ((i%nx) == 0){
     p[i] = p[i+1];
  }
   __syncthreads();
  if(i>=(nx*(nx-1))){
      p[i]=0.0;
   }
}

__global__ void pressure_poisson(float*  p,float* pn, float* b, float dx, float dy, int nx, int ny){
     copy_matrix(p, pn, nx);
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     p[i] = (((pn[i+1] + pn[i-1]) * pow(dy,2) + (pn[i+nx] + pn[i-nx]) * pow(dx,2) )/ (2.0 * (pow(dx,2) + pow(dy,2))) - pow(dx,2) * pow(dy,2) / (2.0 * (pow(dx,2) + pow(dy,2))) * b[i]);
}

__global__ void build_up_b(float*b, float* u,  float*v, float* un, float* vn, float dx, float dy, int nx, int ny){
  int rho = 1;
  float nu = 0.1, dt = 0.001;
  copy_matrix(u, un, nx*nx); //problem can be here
  copy_matrix(v, vn, nx*nx);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i>=nx*nx || i<nx || i>nx*(nx-1)-1 || i%nx ==0 || i%nx ==nx-1) return; //loop bounds
  b[i] = (rho * ((1/dt) * ((u[i+1] - u[i-1]) / (2*dx) + (v[i+nx] - v[i-nx])/ (2 * dy)) - ((u[i+1] - u[i-1]) / (2*dx))*((u[i+1] - u[i-1]) / (2*dx)) - 2 * ((u[i+nx] - u[i-nx]) / (2*dy) * (v[i+1] - v[i-1])/(2 * dx)) - ((v[i+nx] - v[i-nx]) / (2*dy)) * ((v[i+nx] - v[i-nx]) / (2*dy))));
}

__global__ void initiate_matrix(float *a, float value, int x, int y){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=x*y) return;
    a[i] = i;
}

__global__ void cavity_conditions(int nt, float*u, float* v,float*un, float* vn,float dt, float dx, float dy, float* p, float rho, float nu, int  nx, int ny){
 int i = blockIdx.x * blockDim.x + threadIdx.x; 
   if (i>=nx*nx) return; 
     if (i % nx == 0){
          u[i] = 0.0;
          v[i] = 0.0;
      }
      if ((i+1) % nx == 0){
          u[i] = 0.0;
          v[i] = 0.0;
      }
       __syncthreads();
      if (i<nx){
          u[i] = 0.0;
          v[i] = 0.0;
      }
       __syncthreads();
      if(i>=(nx*(nx-1))){
          u[i]=1.0;
          v[i]=0.0;
      }
       __syncthreads();
  }

__global__ void cavity_flow_ops(int nt, float*u, float* v,float*un, float* vn,float dt, float dx, float dy, float* p, float rho, float nu, int  nx, int ny){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i/nx==0 || i%nx == nx-1 || i/nx == nx-1 || i%nx == 0) return; //loop boundaries
 if (i < nx * ny && i>=1) {
      u[i] = (un[i] - un[i] * dt/dx * (un[i] - un[i-1]) - vn[i] * dt/dy * (un[i] - un[i-nx]) - dt/ (2* rho * dx) * (p[i+1] - p[i-1]) + nu * (dt/ pow(dx,2)) * (un[i+1] - 2*un[i] + un[i-1] + dt/(pow(dy,2)) * (un[i+nx] - 2 *un[i] + un[i-nx])));
      __syncthreads();
      v[i]= (vn[i] - un[i] * dt/dx * (vn[i] - vn[i-1]) - vn[i] * dt/dy * (vn[i] - vn[i-nx]) - dt/ (2 * rho * dy) * (p[i+nx] - p[i-nx]) + nu * (dt /(pow(dx,2)) * (vn[i+1] - 2* vn[i] + vn[i-1]) + dt/(pow(dy,2)) * (vn[i+nx] - 2 * vn[i] + vn[i-nx])));
      __syncthreads();
 }
  __syncthreads();
}

ofstream output;
void print_matrix_to_file(float* m,int size){
  for (int i=0; i<size; i++){
     for (int j=0; j<size; j++){
        output << m[i*size+j] << ",";
    }
  output << "\n ";
  }
}


void print_to_file(float* X, float* Y, float* p, float* u, float* v, int nx){
  output.open("results_nt700.csv");
  output << "p\n";
  print_matrix_to_file(p, nx);
  output << "u\n";
  print_matrix_to_file(u,nx);
  output << "v\n";
  print_matrix_to_file(v, nx);
  
}

 void cavity_flow(int nt,float dt, float dx, float dy,  float rho, float nu, int nx, int  ny, float *X, float* Y){
    const int N = nx*nx;
    const int M = 1024;
    float *u, *v, *p;
    cudaMallocManaged(&u, nx*ny*sizeof(float));
    cudaMallocManaged(&v, nx*ny*sizeof(float));
    cudaMallocManaged(&p, nx*ny*sizeof(float));
    
    cudaMemset(u, 0.0, nx*nx*sizeof(float));
    cudaMemset(v, 0.0, nx*nx*sizeof(float));
    cudaMemset(p, 0.0, nx*nx*sizeof(float));
    //copy vectors for cavity_flow function
    float *un, *vn, *b;
    cudaMallocManaged(&un, nx*ny*sizeof(float));
    cudaMallocManaged(&vn, nx*ny*sizeof(float));
    cudaMallocManaged(&b, nx*ny*sizeof(float));
    //initiate to 0
    cudaMemset(un, 0.0, nx*nx*sizeof(float));
    cudaMemset(vn, 0.0, nx*nx*sizeof(float));
    cudaMemset(b, 0.0, nx*nx*sizeof(float));
   
    float *pn;
    cudaMallocManaged(&pn, nx*ny*sizeof(float));
    cudaDeviceSynchronize();
    for (int n=0; n<nt; n++){
      build_up_b<<<(N+M-1)/M,M>>>(b,u,v,un,vn,dx,dy,nx,ny);
      cudaDeviceSynchronize();
      for (int i=0; i<50; i++){
	      pressure_poisson<<<(N+M-1)/M,M>>>(p, pn,b, dx, dy,nx,ny);
  	    cudaDeviceSynchronize();
	      pressure_poisson_conditions<<<(N+M-1)/M,M>>>(p, pn,b, dx, dy,nx,ny);
      }
      cudaDeviceSynchronize();
      cavity_flow_ops<<<(N+M-1)/M,M>>>(nt, u, v, un, vn, dt, dx, dy, p, rho, nu, nx, ny);
      cudaDeviceSynchronize();
      cavity_conditions<<<(N+M-1)/M,M>>>(nt, u, v, un, vn, dt, dx, dy, p, rho, nu, nx, ny);
      cudaDeviceSynchronize();  
    } 
    print_to_file(X, Y, p, u, v,nx);
    cudaFree(u);
    cudaFree(un);
    cudaFree(v);
    cudaFree(vn);
    cudaFree(b);
    cudaFree(p);
    cudaFree(pn);
  }

int main(){
  float rho = 1.0;
  float nu = .1, dt = .001;  
  int nx = 41, ny = 41, nt = 700, c = 1;
  float dx = 2 / float((nx - 1)), dy = 2 / float((ny - 1));
  float* x, *y;
  const int N = nx*nx;
  const int M = 1024;
  cudaMallocManaged(&x, nx*sizeof(float));
  cudaMallocManaged(&y, ny*sizeof(float));
  range<<<1,nx>>>(x,0,2,nx);
  range<<<1,nx>>>(y,0,2,nx);
  cudaDeviceSynchronize();
  float *X, *Y;
  cudaMallocManaged(&X, nx*nx*sizeof(float));
  cudaMallocManaged(&Y, ny*ny*sizeof(float));
  for (int i=0; i<nx; i++){
    for (int j=0; j<ny; j++){
      X[i*nx+j] = x[j];
    }
  }
  for (int i=0; i<nx; i++){
    for (int j=0; j<ny; j++){
      Y[i*nx+j] = x[i];
    }
  } 
  cudaDeviceSynchronize();
  cavity_flow(nt, dt, dx, dy, rho, nu, nx, ny, X, Y);
  cudaDeviceSynchronize();
  cudaFree(Y);
  cudaFree(X);
  cudaFree(y);
  cudaFree(x);
}


