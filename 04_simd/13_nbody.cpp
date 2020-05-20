#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <cmath>
int main() {
        const int N = 8;
        float x[N], y[N], m[N], fx[N], fy[N];
        for(int i=0; i<N; i++) {
                x[i] = drand48();
                y[i] = drand48();
                m[i] = drand48();
                fx[i] = fy[i] = 0;
        }

        __m256 xvec = _mm256_load_ps(x);
        __m256 yvec = _mm256_load_ps(y);
        __m256 fxvec = _mm256_load_ps(fx);
        __m256 fyvec = _mm256_load_ps(fy);
        __m256 mvec = _mm256_load_ps(m);

        //float fxivec[8];
        //float fyivec[8];
        for(int i=0; i<N; i++) {
                __m256 xi = _mm256_set1_ps(x[i]);
                __m256 yi = _mm256_set1_ps(y[i]);

                //obtain rx, ry and r
                __m256 rx = _mm256_sub_ps(xi,xvec);
                __m256 ry = _mm256_sub_ps(yi,yvec);
                __m256 rx2 = _mm256_mul_ps(rx, rx);
                __m256 ry2 = _mm256_mul_ps(ry, ry);
                __m256 rsum = _mm256_add_ps(rx2,ry2);
                __m256 rvec = _mm256_rsqrt_ps(rsum);
          
                __m256 fxi = _mm256_set1_ps(fx[i]);
                __m256 fyi = _mm256_set1_ps(fy[i]);

                __m256 r2 = _mm256_mul_ps(rvec,rvec);
                __m256 nominator = _mm256_mul_ps(rx,mvec);
                __m256 denominator = _mm256_mul_ps(rvec,r2);
                __m256 subtracted = _mm256_div_ps(nominator,denominator);

                //apply the conditional
                __m256 no_distance = _mm256_setzero_ps();
                __m256 mask = _mm256_cmp_ps(rvec,no_distance, _CMP_GT_OQ); //the mask will assign value 0 to index i since distance is 0
                
                fxi = _mm256_sub_ps(fxi,subtracted);
                fyi = _mm256_sub_ps(fyi,subtracted);
                fxi = _mm256_blendv_ps(no_distance, fxi, mask);
                fyi = _mm256_blendv_ps(no_distance, fyi, mask);

                //we can use eduction or some temp array to store the values

                //_mm256_store_ps(fyivec, fyi);
                //_mm256_store_ps(fxivec, fxi);  
                //fx[i] = fxivec;
                //fy[i] = fyivec;
                printf("%d %g %g\n", i, fx[i], fy[i]);

        }
}
