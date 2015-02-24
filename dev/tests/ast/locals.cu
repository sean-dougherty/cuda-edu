#include <wb.h>

void foo() {
    int s0;
    const int s1 = 4;
    unsigned int s2;

    int *p0;
    const int *p1 = &s1;
    int *p2 = (int *)&s2;
    

    //int a0[];
    int a1[5];
    int a2[10][20];

    extern __shared__ int es0[];
    extern __shared__ int *es1;

    __shared__ int ss0[5];

    int3 v0;

    int3 *pv0 = (int3 *)&v0;
}
