/*
 * maxdet, version alpha
 * C source for MAXDET-programs
 *
 * Shao-Po Wu, Lieven Vandenberghe and Stephen Boyd 
 * Mar. 24, 1996, last major update
 */

#include <stdio.h>
#include <string.h>
#include "maxdet.h"
#include "mex.h"

#ifdef __STDC__
void mexFunction(
  int nlhs, Matrix *plhs[],
  int nrhs, Matrix *prhs[]
)
#else
mexFunction(nlhs, plhs, nrhs, prhs)
  int nlhs; 
  Matrix *plhs[];
  int nrhs;
  Matrix *prhs[];
#endif


/*
 * [x,Z,W,ul,hist,infostr]=maxdet(F,F_blkszs,G,G_blkszs,c,x0,Z0,W0,abstol,
 *                                reltol,gamma,NTiters)
 */


{
    static int firstcall=YES;
    register int i, j, k;
    int    L, K, n, l, max_n, max_l, F_sz, G_sz, F_upsz, G_upsz, m;
    int    *F_blkszs, *G_blkszs, *iwork;
    int    NTiters, lwork, iters;
    int    p_pos, up_pos, pos2, lngth;
    int    info, info2;
    double ul[2], abstol, reltol, gamma, *work, *dps;

    int    int1=1;

    /* initial setup */
    if (firstcall) {
        fprintf(stdout,"\nThis is maxdet, version %s.\n",VERSION);
        firstcall = NO;
    }
    if (nrhs != 12) 
        mexErrMsgTxt("12 input arguments required.\n");
    if (nlhs != 6)
        mexErrMsgTxt("6 output arguments required.\n");
    if (MIN(mxGetM(prhs[1]),mxGetN(prhs[1])) != 1)
        mexErrMsgTxt("F_blkszs must be a vector.\n");
    L = MAX(mxGetM(prhs[1]),mxGetN(prhs[1]) );
    if (MIN(mxGetM(prhs[3]),mxGetN(prhs[3])) != 1)
        mexErrMsgTxt("G_blkszs must be a vector.\n");
    K = MAX(mxGetM(prhs[3]),mxGetN(prhs[3]) );
    F_blkszs = (int *) mxCalloc(L, sizeof(int));
    G_blkszs = (int *) mxCalloc(K, sizeof(int));
    for (i=0; i<L; i++) {
        F_blkszs[i] = (int) mxGetPr(prhs[1])[i];
        if (F_blkszs[i] <= 0) 
            mexErrMsgTxt("Elements of F_blkszs must be positive.\n");
    }
    for (i=0; i<K; i++) {
        G_blkszs[i] = (int) mxGetPr(prhs[3])[i];
        if (G_blkszs[i] <= 0) 
            mexErrMsgTxt("Elements of G_blkszs must be positive.\n");
    }

    /* various dimensions
     * n, l: dimensions
     * F_sz, G_sz: length in packed storage
     * F_upsz, G_upsz: length in unpacked storage
     * max_n, max_l: max block size */
    for (i=0, n=0, F_sz=0, F_upsz=0, max_n=0; i<L; i++) {
        n += F_blkszs[i];
        F_sz += (F_blkszs[i]*(F_blkszs[i]+1))/2;
        F_upsz += SQR(F_blkszs[i]);
        max_n = MAX(max_n, F_blkszs[i]);
    }
    for (i=0, l=0, G_sz=0, G_upsz=0, max_l=0; i<K; i++) {
        l += G_blkszs[i];
        G_sz += (G_blkszs[i]*(G_blkszs[i]+1))/2;
        G_upsz += SQR(G_blkszs[i]);
        max_l = MAX(max_l, G_blkszs[i]);
    }

    /* read F and G (unpacked storage); m = number of columns minus one */
    m = MIN(mxGetN(prhs[0]),mxGetN(prhs[2])) - 1;
    if (mxGetM(prhs[0]) != F_upsz) 
        mexErrMsgTxt("Dimension of F does not match F_blkszs given.\n");
    if (mxGetM(prhs[2]) != G_upsz) 
        mexErrMsgTxt("Dimension of G does not match G_blkszs given.\n");
    /* pack F; loop over matrices F_j */
    for (j=0, up_pos=0, p_pos=0; j<=m; j++)
        /* loop over blocks i=0,...,L-1
         * up_pos: position of (0,0)-element of block i in unpacked storage */
        for (i=0; i<L; up_pos += SQR(F_blkszs[i]), i++)
            /* loop over columns, k=0,...,F_blkszs[i]-1
             * p_pos: position of (k,k)-element in packed storage   */
            for (k=0, lngth=F_blkszs[i]; k<F_blkszs[i]; p_pos+=lngth,
                 lngth-=1, k++)
                memcpy(mxGetPr(prhs[0])+p_pos,
                       mxGetPr(prhs[0])+up_pos+k*(F_blkszs[i]+1),
                       lngth*sizeof(double));
    /* pack G; loop over matrices G_j */
    for (j=0, up_pos=0, p_pos=0; j<=m; j++)
        /* loop over blocks i=0,...,K-1
         * up_pos: position of (0,0)-element of block i in unpacked storage */
        for (i=0; i<K; up_pos += SQR(G_blkszs[i]), i++)
            /* loop over columns, k=0,...,G_blkszs[i]-1
             * p_pos: position of (k,k)-element in packed storage   */
            for (k=0, lngth=G_blkszs[i]; k<G_blkszs[i]; p_pos+=lngth,
                 lngth-=1, k++)
                memcpy(mxGetPr(prhs[2])+p_pos,
                       mxGetPr(prhs[2])+up_pos+k*(G_blkszs[i]+1),
                       lngth*sizeof(double));

    /* check dimension of c (5th input argument) */
    if (MIN(mxGetM(prhs[4]), mxGetN(prhs[4])) != 1) 
        mexErrMsgTxt("c must be a vector.\n");
    if (MAX(mxGetM(prhs[4]), mxGetN(prhs[4])) != m) 
        mexErrMsgTxt("Dimensions of c and F (G) do not match.\n");

    /* read x0 (6th input arg), copy in x (1st output arg) */
    if (MIN(mxGetM(prhs[5]), mxGetN(prhs[5])) != 1) 
        mexErrMsgTxt("x0 must be a vector.\n");
    if (MAX(mxGetM(prhs[5]), mxGetN(prhs[5])) != m) 
        mexErrMsgTxt("Dimensions of x0 and c do not match.\n");
    plhs[0] = mxCreateFull(m, 1, REAL);  
    memcpy(mxGetPr(plhs[0]), mxGetPr(prhs[5]), m*sizeof(double));

    /* read Z0 (7th input arg), copy in Z (2nd output arg), packed storage */
    if (MIN(mxGetM(prhs[6]), mxGetN(prhs[6])) != 1)
        mexErrMsgTxt("Z0 must be a vector.\n");
    if (MAX(mxGetM(prhs[6]), mxGetN(prhs[6])) != F_upsz) 
        mexErrMsgTxt("Dimension of Z0 does not match F_blkszs.\n");
    plhs[1] = mxCreateFull(F_upsz, 1, REAL);
    /* loop over blocks, i=0,...,L-1
     * up_pos: position of (0,0) element of block i in unpacked storage */
    for (i=0, up_pos=0, p_pos=0; i<L; up_pos+=SQR(F_blkszs[i]), i++)
        /* loop over columns, k=0,...,F_blkszs[i]-1
         * p_pos: position of (k,k) element of block i in packed storage */
        for (k=0, lngth=F_blkszs[i]; k<F_blkszs[i]; p_pos+=lngth,
             lngth-=1, k++)
            memcpy(mxGetPr(plhs[1]) + p_pos,
                   mxGetPr(prhs[6]) + up_pos + k*(F_blkszs[i]+1),
                   lngth*sizeof(double));

    /* read W0 (8th input arg), copy in W (3rd output arg), packed storage */
    if (MIN(mxGetM(prhs[7]), mxGetN(prhs[7])) != 1)
        mexErrMsgTxt("W0 must be a vector.\n");
    if (MAX(mxGetM(prhs[7]), mxGetN(prhs[7])) != G_upsz) 
        mexErrMsgTxt("Dimension of W0 does not match G_blkszs.\n");
    plhs[2] = mxCreateFull(G_upsz, 1, REAL);
    /* loop over blocks, i=0,...,K-1
     * up_pos: position of (0,0) element of block i in unpacked storage */
    for (i=0, up_pos=0, p_pos=0; i<K; up_pos+=SQR(G_blkszs[i]), i++)
        /* loop over columns, k=0,...,G_blkszs[i]-1
         * p_pos: position of (k,k) element of block i in packed storage */
        for (k=0, lngth=G_blkszs[i]; k<G_blkszs[i]; p_pos+=lngth,
             lngth-=1, k++)
            memcpy(mxGetPr(plhs[2]) + p_pos,
                   mxGetPr(prhs[7]) + up_pos + k*(G_blkszs[i]+1),
                   lngth*sizeof(double));

    /* read abstol, gamma, maxiters, NTiters */
    abstol = mxGetScalar(prhs[8]);
    reltol = mxGetScalar(prhs[9]);
    gamma = mxGetScalar(prhs[10]);
    NTiters = (int) mxGetScalar(prhs[11]);

    /* ul (4th output argument) */
    plhs[3] = mxCreateFull(2,1,REAL);

    /* hist (5th output argument) */
    plhs[4] = mxCreateFull(3,NTiters,REAL);

    /* allocate work space */
    lwork = (2*m+5)*(F_sz+G_sz) + 2*(n+l) +
        MAX(m+(F_sz+G_sz)*NB,MAX(3*(m+SQR(m)+MAX(G_sz,F_sz)),
                                 MAX(3*(MAX(max_l,max_n)+MAX(G_sz,F_sz)),
                                     MAX(G_sz+3*max_l,F_sz+3*max_n))));
    work = (double *) mxCalloc(lwork, sizeof(double));
    iwork = (int *) mxCalloc(10*m, sizeof(int));


    /* call maxdet */
    fprintf(stdout,"\ninvoking maxdet...\n");
    info2 = maxdet(m,L,mxGetPr(prhs[0]),F_blkszs,K,mxGetPr(prhs[2]),G_blkszs,
                   mxGetPr(prhs[4]),mxGetPr(plhs[0]),mxGetPr(plhs[1]),
                   mxGetPr(plhs[2]),mxGetPr(plhs[3]),mxGetPr(plhs[4]),gamma,
                   abstol,reltol,&NTiters,work,lwork,iwork,&info);

    /* unpack Z */
    /* loop over blocks i=L-1,...,0
     * up_pos: position of last element of block i in unpacked storage
     * p_pos: position of last element of block i in packed storage */
    for (i=L-1, up_pos=F_upsz-1, p_pos=F_sz-1; i>=0; 
         up_pos -= SQR(F_blkszs[i]),
         p_pos -= F_blkszs[i]*(F_blkszs[i]+1)/2, i--) 
        /* loop over columns of blok i;  pos2 is position of 
         * (F_blkszs[i]-k) x (F_blkszs[i]-k) element of block i in packed 
         * storage */
        for (k=0, lngth=1, pos2=p_pos;  k<F_blkszs[i]; 
             lngth+=1, pos2-=lngth, k++)
            /* move subdiagonal part of column F_blkszs[i]-k */
            memmove(mxGetPr(plhs[1]) + up_pos - k*(F_blkszs[i]+1),
                    mxGetPr(plhs[1]) + pos2, lngth*sizeof(double) );
    /* loop over blocks i=0,...,L-1
     * up_pos: position of (0,0) element of block i */
    for (i=0, up_pos=0;  i<L;  up_pos+=SQR(F_blkszs[i]), i++)
        /* loop over columns k=0,...,F_blkszs[i]-1 */
        for (k=0, lngth=F_blkszs[i]-1;  k<F_blkszs[i]-1;  lngth-=1, k++)
            /* copy part of column k under diagonal to part of row k 
             * above the diagonal */ 
            dcopy_(&lngth, mxGetPr(plhs[1]) + up_pos + k*(F_blkszs[i]+1) + 1,
                   &int1, mxGetPr(plhs[1]) + up_pos + (k+1)*F_blkszs[i] + k, 
                   F_blkszs+i); 

    /* unpack W */
    /* loop over blocks i=K-1,...,0
     * up_pos: position of last element of block i in unpacked storage
     * p_pos: position of last element of block i in packed storage */
    for (i=K-1, up_pos=G_upsz-1, p_pos=G_sz-1; i>=0; 
         up_pos -= SQR(G_blkszs[i]),
         p_pos -= G_blkszs[i]*(G_blkszs[i]+1)/2, i--) 
        /* loop over columns of blok i;  pos2 is position of 
         * (G_blkszs[i]-k) x (G_blkszs[i]-k) element of block i in packed 
         * storage */
        for (k=0, lngth=1, pos2=p_pos;  k<G_blkszs[i]; 
             lngth+=1, pos2-=lngth, k++)
            /* move subdiagonal part of column G_blkszs[i]-k */
            memmove(mxGetPr(plhs[2]) + up_pos - k*(G_blkszs[i]+1),
                    mxGetPr(plhs[2]) + pos2, lngth*sizeof(double) );
    /* loop over blocks i=0,...,K-1
     * up_pos: position of (0,0) element of block i */
    for (i=0, up_pos=0;  i<K;  up_pos+=SQR(G_blkszs[i]), i++)
        /* loop over columns k=0,...,G_blkszs[i]-1 */
        for (k=0, lngth=G_blkszs[i]-1;  k<G_blkszs[i]-1;  lngth-=1, k++)
            /* copy part of column k under diagonal to part of row k 
             * above the diagonal */ 
            dcopy_(&lngth, mxGetPr(plhs[2]) + up_pos + k*(G_blkszs[i]+1) + 1,
                   &int1, mxGetPr(plhs[2]) + up_pos + (k+1)*G_blkszs[i] + k, 
                   G_blkszs+i); 

    /* unpack F again */
    /* loop over columns j=m,...,0  */
    for (j=m;  j>=0;  j--){ 

        /* loop over blocks i=L-1, ..., 0 
         * up_pos = position of last element of block i in unpacked storage
         * p_pos = position of last element of block i in packed storage */
        for (i=L-1, up_pos=(j+1)*F_upsz-1, p_pos=(j+1)*F_sz-1;  i>=0; 
             up_pos -= SQR(F_blkszs[i]), 
             p_pos -= F_blkszs[i]*(F_blkszs[i]+1)/2, i--)

            /* loop over columns k=F_blkszs[i]-1, ..., 0;  pos2 is position
             * of elt (F_blkszs[i]-k) x (F_blkszs[i]-k) of block i in 
             * packed storage */
            for (k=0, lngth=1, pos2=p_pos;  k<F_blkszs[i];  
                 lngth+=1, pos2-=lngth, k++)

                /* move subdiagonal part of column F_blkszs[i]-k */ 
                memmove( mxGetPr(prhs[0]) + up_pos - k*(F_blkszs[i]+1),
                         mxGetPr(prhs[0]) + pos2, lngth*sizeof(double));

        /* loop over blocks i=0,..,L-1 
         * up_pos: position of (0,0) element of block i */
        for (i=0, up_pos=j*F_upsz;  i<L;  up_pos+=SQR(F_blkszs[i]), i++) 
      
            /* loop over columns k=0,...,F_blkszs[i]-1 */
            for (k=0, lngth=F_blkszs[i]-1;  k<F_blkszs[i]-1;  lngth-=1, k++)

                /* copy part of column k under diagonal to part of row k
                 * above the diagonal */
                dcopy_(&lngth, 
                       mxGetPr(prhs[0]) + up_pos + k*(F_blkszs[i]+1) + 1,
                       &int1, 
                       mxGetPr(prhs[0]) + up_pos + (k+1)*F_blkszs[i] + k, 
                       F_blkszs+i);
    }


    /* unpack G again */
    /* loop over columns j=m,...,0  */
    for (j=m;  j>=0;  j--){ 

        /* loop over blocks i=K-1, ..., 0 
         * up_pos = position of last element of block i in unpacked storage
         * p_pos = position of last element of block i in packed storage */
        for (i=K-1, up_pos=(j+1)*G_upsz-1, p_pos=(j+1)*G_sz-1;  i>=0; 
             up_pos -= SQR(G_blkszs[i]), 
             p_pos -= G_blkszs[i]*(G_blkszs[i]+1)/2, i--)

            /* loop over columns k=G_blkszs[i]-1, ..., 0;  pos2 is position
             * of elt (G_blkszs[i]-k) x (G_blkszs[i]-k) of block i in 
             * packed storage */
            for (k=0, lngth=1, pos2=p_pos;  k<G_blkszs[i];  
                 lngth+=1, pos2-=lngth, k++)

                /* move subdiagonal part of column G_blkszs[i]-k */ 
                memmove( mxGetPr(prhs[2]) + up_pos - k*(G_blkszs[i]+1),
                         mxGetPr(prhs[2]) + pos2, lngth*sizeof(double));

        /* loop over blocks i=0,..,K-1 
         * up_pos: position of (0,0) element of block i */
        for (i=0, up_pos=j*G_upsz;  i<K;  up_pos+=SQR(G_blkszs[i]), i++) 

            /* loop over columns k=0,...,G_blkszs[i]-1 */
            for (k=0, lngth=G_blkszs[i]-1;  k<G_blkszs[i]-1;  lngth-=1, k++)

                /* copy part of column k under diagonal to part of row k
                 * above the diagonal */
                dcopy_(&lngth, 
                       mxGetPr(prhs[2]) + up_pos + k*(G_blkszs[i]+1) + 1,
                       &int1, 
                       mxGetPr(prhs[2]) + up_pos + (k+1)*G_blkszs[i] + k, 
                       G_blkszs+i);
    }

    /* truncate hist */
    dps = mxGetPr(plhs[4])+2;
    iters = 0;
    while (*dps != 0) {
        iters++;
        dps += 3;
    }
    mxSetN(plhs[4],iters);

    /* infostr */
    switch (info) {
      case 1:
        plhs[5] = mxCreateString("maximum Newton iteration exceeded");
        break;
      case 2:
        plhs[5] = mxCreateString("absolute tolerance reached");
        break;
      case 3:
        plhs[5] = mxCreateString("relative tolerance reached");
        break;
      default:
        plhs[5] = mxCreateString("error occurred in maxdet"); 
    }

    /* free matrices allocated */
    mxFree(F_blkszs);
    mxFree(G_blkszs);
    mxFree(work);
    mxFree(iwork);

    if (info2) mexErrMsgTxt("Error in maxdet.\n");
}

