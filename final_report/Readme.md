# 19M38202

Change the name of the output file and the value of nt to obtain the results in the notebook 

To run OpenMP code    

      g++ -fopenmp cavity_openmp.cpp

To run OpenAcc code

      pgc++ cavity_openacc.cpp -acc -std=c++11 -Minfo=accel

To run CUDA code

      nvcc cavity_cuda.cu 
