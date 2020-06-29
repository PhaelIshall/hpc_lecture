#include <ratio>
#include <vector>
#include<iostream>
#include <math.h> 
#include <openacc.h>
#import <fstream>

using namespace std;
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
void range(float *r, float min, float max, int N) {
    float delta = (max-min)/float(N-1);
    #pragma acc parallel loop
    for(int i=0; i<N; i++) {
        r[i] = min + i*delta;
    }
}

void copy_matrix(float* o, float* copy, int size){
    #pragma acc parallel loop
    for (int i=0; i<size*size; i++){
        copy[i]=o[i];
    }
}

void pressure_poisson(float*  p,float*  b, float dx, float dy, int nx, int ny){
  int nit = 50;
  float pn[nx*ny], p_copy[nx*ny];
  copy_matrix(p, pn, 41);
  for (int q=0; q<nit; q++){
      copy_matrix(p, pn, nx);
  
  #pragma acc parallel loop
  for (int i=1; i<ny-1; i++){
    for (int j=1; j<nx-1; j++){
      //already using pn instead of p, omp parallel for is sufficient
      p[i*41+j] = (((pn[i*nx+j+1] + pn[i*nx+j-1]) * pow(dy,2) + (pn[(i+1)*nx+j] + pn[(i-1)*nx+j]) * pow(dx,2) )/ (2.0 * (pow(dx,2) + pow(dy,2))) - pow(dx,2) * pow(dy,2) / (2.0 * (pow(dx,2) + pow(dy,2))) * b[i*nx+j]);
    }
  }

  #pragma acc parallel loop     			
  for (int z=0; z<nx; z++){
    p[z*nx+nx-1]=  p[z*nx+nx-2];
  }  
  
  #pragma acc parallel loop 
  for (int z=0; z<nx; z++){
    p[z] = p[1*nx+z]; //change first row with the second row
  }
  #pragma acc parallel loop
  for (int z=0; z<nx; z++){
    p[z*nx] = p[z*nx+1]; //change first col with the second col
  }
   
  #pragma acc parallel loop
  for (int z=0; z<nx; z++){
    p[nx*(nx-1)+z] = 0.0; //last row all zeros
  }
 }
}

void build_up_b(float*b, float* u,  float*v, float dx, float dy, int nx, int ny){
  int rho = 1;
  float nu = 0.1, dt = 0.001;
    //in python, last index is excluded
    #pragma acc parallel loop 
    for (int i=1; i<ny-1; i++){
      for (int j=1; j<nx-1; j++){  //below should be ny but since ny=nx it should work
        b[i*nx+j] = (rho * ((1/dt) * ((u[i*nx+j+1] - u[i*nx+j-1]) / (2*dx) + (v[(i+1)*nx+j] - v[(i-1)*nx+j])/ (2 * dy)) - 
        ((u[i*nx+j+1] - u[i*nx+j-1]) / (2*dx))*((u[i*nx+j+1] - u[i*nx+j-1]) / (2*dx)) - 
        2 * ((u[(i+1)*nx+j] - u[(i-1)*nx+j]) / (2*dy) * 
        (v[i*nx+j+1] - v[i*nx+j-1])/(2 * dx)) -
        ((v[(i+1)*nx+j] - v[(i-1)*nx+j]) / (2*dy)) * ((v[(i+1)*nx+j] - v[(i-1)*nx+j]) / (2*dy))));
      } 
    }
}

//initiates a matrix with any given value
void initiate_matrix(float *a, float value, int x, int y){
  #pragma acc parallel loop
  for (int i=0; i<x*y; i++){
      a[i] = value;
  } 
}

 void cavity_flow(int nt, float*u, float* v, float dt, float dx, float dy, float* p, float rho, float nu, int nx, int ny){

     float un[nx*ny], vn[nx*ny], b[nx*ny]; //set b to be a zero vector
     initiate_matrix(un, 0.0, ny, nx); //empty_like(u)
     initiate_matrix(vn, 0.0, ny, nx); //empty_like(u)
     initiate_matrix(b, 0.0, ny, nx);
 
     for (int n=0; n<nt; n++){
        copy_matrix(u, un, nx);
        copy_matrix(v,vn,  nx);
        build_up_b(b,u,v,dx,dy,nx,ny);
        pressure_poisson(p,b, dx, dy,nx,ny);
        //since we are using un and u copies there is no conflict 
       
	 #pragma acc parallel loop
	for (int i=1; i<ny-1; i++){
          for (int j=1; j<nx-1; j++){ 
            // u confirmed to work great :D
            u[i*nx+j]= (un[i*nx+j] - 
            un[i*nx+j] * dt/dx *
            (un[i*nx+j] - un[i*nx+j-1]) -
            vn[i*nx+j] * dt/dy * 
            (un[i*nx+j] - un[(i-1)*nx+j]) -
            dt/ (2 * rho * dx) * (p[i*nx+j+1] - p[i*nx+j-1]) + nu * (dt /(pow(dx,2)) * 
            (un[i*nx+j+1] - 2* un[i*nx+j] + un[i*nx+j-1]) + dt/(pow(dy,2)) * 
            (un[(i+1)*nx+j] - 2 * un[i*nx+j] + un[(i-1)*nx+j])));

            // vi is confirmed too
            v[i*nx+j]= (vn[i*nx+j] - 
            un[i*nx+j] * dt/dx *
            (vn[i*nx+j] - vn[i*nx+j-1]) -
            vn[i*nx+j] * dt/dy * 
            (vn[i*nx+j] - vn[(i-1)*nx+j]) -
            dt/ (2 * rho * dy) * (p[(i+1)*nx+j] - p[(i-1)*nx+j]) + nu * (dt /(pow(dx,2)) * 
            (vn[i*nx+j+1] - 2* vn[i*nx+j] + vn[i*nx+j-1]) + dt/(pow(dy,2)) * 
            (vn[(i+1)*nx+j] - 2 * vn[i*nx+j] + vn[(i-1)*nx+j])));
          }
        }

	#pragma acc parallel loop 
        for(int i=0;i<ny;i++){
            u[i*nx+0]=0.0;
            u[i*nx+nx-1]=0.0;
            v[i*nx+0]=0.0;
            v[i*nx+nx-1]=0.0;
        }

	#pragma acc parallel loop 
        for(int i=0;i<nx;i++){
            u[0*nx+i]=0.0;
            u[(ny-1)*nx+i]=1.0;
            v[0*nx+i]=0.0;
            v[(ny-1)*nx+i]=0.0;
        }
      }           
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


void print_to_file(float* p, float* u, float* v, int nx){
  output.open("results_cav_nt700.csv");
  output << "p\n";
  print_matrix_to_file(p, nx);
  output << "u\n";
  print_matrix_to_file(u,nx);
  output << "v\n";
  print_matrix_to_file(v, nx);

}

int main(){
  int nx = 41, ny = 41, nt = 100, c = 1;
  float dx = 2 / float((nx - 1)), dy = 2 / float((ny - 1));
  float x[nx], y[ny];
  float rho = 1.0;
  float nu = .1, dt = .001;
  
  range(x,0,2,nx); //linespace
  range(y,0,2,nx); //linespace

  float X[nx*ny], Y[nx*ny];
  float u[nx*ny], v[nx*ny], p[nx*ny], b[nx*ny]; 
  
  #pragma acc parallel loop
  for (int i=0; i<ny; i++){
    for (int j=0; j<nx; j++){
      X[i*nx+j] = x[j]; //initialize X
      Y[i*nx+j] = x[i]; //initialize Y
      u[i*nx+j] = 0.0; //set u to zero vector
      v[i*nx+j] = 0.0; //set v to zero vector 
      p[i*nx+j] = 0.0; //set v to zero vector
      b[i*nx+j] = 0.0; //set v to zero vector
    }
  }

  cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu, nx, ny);
  print_to_file(p, u,v,nx);
}
