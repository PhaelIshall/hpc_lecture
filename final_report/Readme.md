To run OpenMP code    

      g++ -fopenmp cavity_openmp.cpp

To run OpenAcc code

      pgc++ cavity_openacc.cpp -acc -std=c++11 -Minfo=accel
