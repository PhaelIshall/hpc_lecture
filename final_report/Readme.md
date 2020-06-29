# 19M38202
### Files

#### Conversion from Python to C++
Look at file `cavity_flow.cpp`

#### OpenMP implementation
Look at file `cavity_openmp.cpp`

#### OpenACC implementation
Look at file `cavity_openacc.cpp`

#### CUDA implementation
Look at file `cavity_cuda.cu`

Please look at the jupyter file titled `Final_report_plots.ipynb`, it contains plots of running the C++ and CUDA code. The jupyter file parses the below result files which can be obtained by running the code.

The files `results_cav_nt100.csv` and `results_cav_nt700.csv` contains the results of the conversion of code from python to C++. They are the same results for OpenACC and OpenMP code for `nt=100` and `nt=700`.

Th files `results_nt100.csv` and `results_nt100.csv` are the results of running CUDA for `nt=100` and `nt=700`.

### Running the code

Change the name of the output file and the value of nt to obtain the results in the notebook 

To run OpenMP code    

      g++ -fopenmp cavity_openmp.cpp

To run OpenAcc code

      pgc++ cavity_openacc.cpp -acc -std=c++11 -Minfo=accel

To run CUDA code

      nvcc cavity_cuda.cu 
