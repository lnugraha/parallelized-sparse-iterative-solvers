#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double ddot(double *x, double *y, int size){
   double sum = 0.0; int i = 0;
//   #pragma omp parallel for reduction(+:sum) private(i) shared(x,y,size)
   for(i=0;i<size;i++) sum += x[i]*y[i];
}

double distance(double *x_new, double *x_old, int size){
   int i; double sum = 0.0;
   for(i=0;i<size;i++) sum += (x_new[i]-x_old[i])*(x_new[i]-x_old[i]);
   return sum;
}

void spdgemv(double *val,int *colidx,int *rowptr,double *x,double *y,int size,int nnz){
   int i,j,k; double sum;
//   #pragma omp for private(i,j) schedule(guided)
   for(i=0;i<size;i++){
   sum = 0.0;
      for(j=rowptr[i];j<rowptr[i+1];j++){
         sum += val[j]*x[colidx[j]];
         y[i] = sum;
      }
   }
   return;
}

double define_max(double *input,int size,double max){
double new_max=max;
for(int i=0;i<size;i++){
   if(fabs(input[i]) > fabs(max)){
     new_max = input[i];
   }
}
return new_max;
}

int define_idx(double *input,int size,double max,int idx){
int new_idx=idx; double new_max=max;
for(int i=0;i<size;i++){
   if(fabs(input[i]) > fabs(new_max)){
     new_idx = i;
     new_max = input[i];
   }
}
return new_idx;
}

void csr_sw(double *val,int *colidx,int *rowptr,double *x,double *b,int size, double epsilon,int nnz,int maxit,int *numit){
int i,j,k,index; double error,diag,max_res,dxi;
double *dx = (double*)malloc(size*sizeof(double)); double *x_old = (double*)malloc(size*sizeof(double));
double *r = (double*)malloc(size*sizeof(double)); double *r_new = (double*)malloc(size*sizeof(double));
double *Ax = (double*)malloc(size*sizeof(double));

for(i=0;i<size;i++) dx[i] = 0.0;
double resid_norm;
double solut_norm = ddot(b,b,size);

for(k=0;k<maxit;k++){
   spdgemv(val,colidx,rowptr,x,Ax,size,nnz);
   for(i=0;i<size;i++){
//      printf("Ax[%d]: %.5f \n",i,Ax[i]);
      r[i] = b[i] - Ax[i];
//      printf("residue: %.4f \n",r[i]);
   }

   resid_norm = ddot(r,r,size);
   error = sqrt(resid_norm/solut_norm);

   double max_val = r[0]; int max_idx = 0;
   for(i=0;i<size;i++) x_old[i] = x[i];

   max_res = define_max(r,size,max_val);
   index = define_idx(r,size,max_val,max_idx);
//   printf("max res %.3f and loc %d \n", max_res,index); // Monitor the max residue & its index

      // Find the Diagonal Value of the Max Residue
      for(j=rowptr[index];j<rowptr[index+1];j++){
         if(index == colidx[j]) diag = val[j];
      } // FOR j
//      printf("Diagonal Value: %.3f \n", diag);

      dxi = b[index];
      for(j=rowptr[index];j<rowptr[index+1];j++){
//         if(i != index){
           dxi -= (val[j]*x[colidx[j]]) ;  // update one residue
//         }
      } // FOR j // update r[i] values
      x[index] += dxi/diag;
//      printf("Updated x[%d]: %.3f \n", index,x[index]);

   if(error <= epsilon) break;
} // FOR k

printf("Difference: %.4e \n", error);
if(k>=maxit){
  printf("Fail To Converge! \n");
  exit(1);
} //END-IF
*numit = k+1;
free(r);free(x_old);free(r_new);free(Ax);
return;
}

//////////// PARALLEL SOLVERS /////////////
void csr_omp_jacobi(double *val, int *colidx, int *rowptr, double *x, double *b, int size, double epsilon, int nnz, int maxit, int *numit)
{
int i, j, k; double error, diag, dxi;
double *dx = (double*)malloc(size*sizeof(double));
double *y  = (double*)malloc(size*sizeof(double));
double *x_old = (double*)malloc(size*sizeof(double));
double *Ax = (double*)malloc(size*sizeof(double));
double *resid = (double*)malloc(size*sizeof(double));

for(i=0;i<size;i++) y[i] = 0.0;

spdgemv(val,colidx,rowptr,x,Ax,size,nnz); 	// Obtain initial Ax_0
for(i=0;i<size;i++) resid[i]=b[i]-Ax[i];	// Obtain initial r_0
double resid_norm = ddot(resid,resid,size); 	// will always be updated
double solut_norm = ddot(b,b,size); 		// remains constant
error = sqrt(resid_norm/solut_norm); 		// should be equals to 1
// printf("Initial Error = %.3f\n", error);

// START ITERATION
for(k=0;k<maxit;k++){

   for(i=0;i<size;i++){
      dx[i] = b[i]; x_old[i] = x[i];
   }

   #pragma omp for private(i,j) schedule(static)
   for(i=0;i<size;i++){
      dxi = 0.0;
//      #pragma omp parallel for shared(val,colidx,rowptr) reduction(+:dxi)
      for(j=rowptr[i];j<rowptr[i+1];j++){
         if(i == colidx[j]){
//           printf("Diagonal Element: %.2f \n", val[j]);
           diag = val[j];
         }
         else if(i != colidx[j]){
//         printf("Non-Diagonal Element: %.2f \n", val[j]);
//	   printf("val[%d] and x[colidx[%d]] is %.2f and %.2f \n", j,j,val[j],x[colidx[j]]);
//           #pragma omp critical
	   dxi += val[j]*x[colidx[j]];
         }
      } // END-FOR j
//      #pragma omp single
      dx[i] = (dx[i]-dxi)/diag;
//      printf("dx[%d] is %.4f \n", i, dx[i]);
//      y[i] += dx[i];
   } // FOR i

   for(i=0;i<size;i++){ // printf("%.2f \n",x[i]);
      x_old[i] = x[i];
      x[i] = dx[i];
   }

   spdgemv(val,colidx,rowptr,x,Ax,size,nnz); // updated Ax
   for(i=0;i<size;i++) resid[i] = b[i] - Ax[i]; // updated resid

   resid_norm = ddot(resid,resid,size);
   error = sqrt(resid_norm/solut_norm);
//   if((k%10)==0) printf("%.6e \n", error);
   if(error <= epsilon) break;
} // FOR k

printf("Final Error: %.4e \n", error);
if(k>=maxit){
  printf("Fail To Converge! \n");
  exit(1);
}
*numit = k+1;
free(dx);free(y);free(x_old);free(Ax);free(resid);
return;
}

void csr_omp_gs(double *val,int *colidx,int *rowptr,double *x,double *b,int size,double epsilon,int nnz,int maxit,int *numit){
int i,j,k; double error, diag, dxi;
double *dx = (double*)malloc(size*sizeof(double));
double *x_old = (double*)malloc(size*sizeof(double));
double *Ax = (double*)malloc(size*sizeof(double));
double *resid = (double*)malloc(size*sizeof(double));

spdgemv(val,colidx,rowptr,x,Ax,size,nnz); 	// Obtain initial Ax_0
for(i=0;i<size;i++) resid[i]=b[i]-Ax[i];	// Obtain initial r_0
double resid_norm = ddot(resid,resid,size); 	// will always be updated
double solut_norm = ddot(b,b,size); 		// remains constant
error = sqrt(resid_norm/solut_norm); 		// should be equals to 1
// printf("Initial Error = %.3f\n", error);

for(k=0;k<maxit;k++){

   for(i=0;i<size;i++){
      dx[i] = b[i];
      x_old[i] = x[i];
   }

   #pragma omp parallel for private(i,j) reduction(+:dxi)
   for(i=0;i<size;i++){
//   #pragma omp parallel for private(j) reduction(+:dxi)
      for(j=rowptr[i];j<rowptr[i+1];j++){
         dxi = 0.0;
         if(i == colidx[j]){
//           printf("Diagonal Element: %.3f \n",val[j]);
           diag = val[j];
         }
         else if(i != colidx[j]){
//           printf("Non-Diagonal Elements: %.2f \n",val[j]);
//           printf("val[%d] and x[colidx[%d]] is %.4f and %.4f \n",j,j,val[j],x[colidx[j]]);
//           dx[i] -= val[j]*x[colidx[j]]; // Non-Activated in OpenMP Ver.
           dxi = val[j]*x[colidx[j]];
           #pragma omp critical
           dx[i] -= dxi;
         } // END IF CLAUSE
      } // FOR j
      dx[i] /= diag;
      x[i] = dx[i];
//      printf("dx[%d] and x[%d] is %.4f and %.4f \n",i,i,dx[i],x[i]);
   } // FOR i

   spdgemv(val,colidx,rowptr,x,Ax,size,nnz); // updated Ax
   for(i=0;i<size;i++) resid[i] = b[i] - Ax[i]; // updated resid
   resid_norm = ddot(resid,resid,size);
   error = sqrt(resid_norm/solut_norm);

//   if((k%10)==0) printf("%.6e \n", error);
   if(error <= epsilon) break;
} // FOR k

// for(i=0;i<size;i++) printf("x[%d] is %.3f \n",i,x[i]);
printf("Difference: %.4e \n", error);
if(k>=maxit){
  printf("Fail To Converge! \n");
  exit(1);
} // END-IF
*numit = k+1;
free(dx);free(x_old);free(Ax);free(resid);
return;
}

void csr_sor(double *val,int *colidx, int *rowptr,double *x, double *b,int size,double epsilon,double omg,int nnz,int maxit,int *numit){
int i,j,k; double error, diag;
double *dx = (double*)malloc(size*sizeof(double));
double *x_old = (double*)malloc(size*sizeof(double));
double *Ax = (double*)malloc(size*sizeof(double));
double *resid = (double*)malloc(size*sizeof(double));

spdgemv(val,colidx,rowptr,x,Ax,size,nnz); 	// Obtain initial Ax_0
for(i=0;i<size;i++) resid[i]=b[i]-Ax[i];	// Obtain initial r_0
double resid_norm = ddot(resid,resid,size); 	// will always be updated
double solut_norm = ddot(b,b,size); 		// remains constant
error = sqrt(resid_norm/solut_norm); 		// should be equals to 1
// printf("Initial Error = %.3f\n", error);

for(k=0;k<maxit;k++){
   for(i=0;i<size;i++){
      dx[i] = b[i];
      x_old[i] = x[i];
      for(j=rowptr[i];j<rowptr[i+1];j++){
         if(i == colidx[j]){
//           printf("Diagonal Element: %.3f \n",val[j]);
           diag = val[j];
         }
         else if(i != colidx[j]){
//           printf("Non-Diagonal Elements: %.2f \n",val[j]);
//           printf("val[%d] and x[colidx[%d]] is %.4f and %.4f \n",j,j,val[j],x[colidx[j]]);
           dx[i] -= omg*(val[j]*x[colidx[j]]);
//           printf("dx[%d] is %.4f \n",i,dx[i]);
         }
      } // FOR j
      dx[i] /= diag;
      x[i] = (1.0-omg)*x[i] + dx[i];
//      printf("dx[%d] and x[%d] is %.4f and %.4f \n",i,i,dx[i],x[i]);
   } // FOR i

   spdgemv(val,colidx,rowptr,x,Ax,size,nnz); // updated Ax
   for(i=0;i<size;i++) resid[i] = b[i] - Ax[i]; // updated resid
   resid_norm = ddot(resid,resid,size);
   error = sqrt(resid_norm/solut_norm);

   if((k%10)==0) printf("%.6e \n", error);
   if(error <= epsilon) break;
} // FOR k

// for(i=0;i<size;i++) printf("x[%d] is %.4f \n", i,x[i]); // Print and Check your solution

//printf("Difference: %.4e \n", error);
if(k>=maxit){
  printf("Fail To Converge! \n");
  exit(1);
} // END-IF
*numit = k+1;
free(dx);free(x_old);free(Ax);free(resid);
return;
}
