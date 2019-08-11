


#include <stdio.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <cassert>
#include <time.h>
#include <string.h>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <stdint.h>
#include <windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define limit_x 63
#define limit_y 63
#define limit_z 19
#define THREADS_PER_BLOCK 512

using namespace std;


vector<int> voxelIDList;

typedef struct {
	float x;
	float y;
	float z;
} coordinate3D;


static inline bool
is_half_integer(const float a)
{
	return fabs(floor(a) + .5F - a)<.0001F;
}

/*
void swap(int *a, int *b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}

void randomize(int arr[], int n)
{
	// Use a different seed value so that we don't get same 
	// result each time we run this program 
	srand(time(NULL));

	// Start from the last element and swap one by one. We don't 
	// need to run for the first element that's why i > 0 
	for (int i = n - 1; i > 0; i--)
	{
		// Pick a random index from 0 to i 
		int j = rand() % (i + 1);

		// Swap arr[i] with the element at random index 
		swap(&arr[i], &arr[j]);
	}
}
*/

/*static inline float
norm(const coordinate3D d)
{
return sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
}*/
/*static inline float
Max(int x, int y) {
return (x > y) ? x : y;
}

static inline float
Min(int x, int y) {
return (x < y) ? x : y;
}
*/

/*int lookup_voxelID(const coordinate3D current_voxel, const int Nx, const int Ny, const int Nz) {
return (int)(current_voxel.z + (Nz - 1) / 2)*Nx*Ny + (int)(current_voxel.y + (Ny - 1) / 2)*Nx + (int)(current_voxel.x + (Nx - 1) / 2);
}
*/
/*
__global__ void
AddArr(int *array, int N) {
int tid = blockIdx.x*blockDim.x+threadIdx.x;
if(tid<N)
atomicAdd(&array[tid], 1);
}
*/


__global__ void
RayTraceVoxelsOnCartesianGrid(
	const coordinate3D *lors,
	const float *x_offset,
	const float *y_offset,
	const float *z_offset,
	const int *index_array,
	const float *eff,
	const float *geo_eff,
	float *hist_fp,
	float *hist_attn,
	float *voxel_list,
	float *img,
	const float voxel_size,
	const int Nx,
	const int Ny,
	const int Nz,
	const int N_LOR,
	bool compute_sensitivity,
	bool compute_forward,
	bool compute_backward)
{
	/* core code deleted */
	//return;
	//return lor_probability;
}




__global__ void compute_geometry(const coordinate3D *crystalPos, const coordinate3D *crystalPos2, const int *index_array, float *x_offset, float *y_offset, float *z_offset, float *eff, const int N_LOR, int sign_kk, int sign_jj, int sign_aa, int sign_bb) {

	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < N_LOR) {

		float x1_sc = crystalPos2[index_array[3 * tid]].x;
		float y1_sc = crystalPos2[index_array[3 * tid]].y;
		float z1_sc = crystalPos2[index_array[3 * tid]].z;
		float x2_sc = crystalPos2[index_array[3 * tid + 1]].x;
		float y2_sc = crystalPos2[index_array[3 * tid + 1]].y;
		float z2_sc = crystalPos2[index_array[3 * tid + 1]].z;
		float x1_c = crystalPos[index_array[3 * tid]].x;
		float y1_c = crystalPos[index_array[3 * tid]].y;
		float z1_c = crystalPos[index_array[3 * tid]].z;
		float x2_c = crystalPos[index_array[3 * tid + 1]].x;
		float y2_c = crystalPos[index_array[3 * tid + 1]].y;
		float z2_c = crystalPos[index_array[3 * tid + 1]].z;

		float delta_x1 = x1_c - x1_sc;
		float delta_y1 = y1_c - y1_sc;
		float delta_z1 = z1_c - z1_sc;
		float delta_x2 = x2_c - x2_sc;
		float delta_y2 = y2_c - y2_sc;
		float delta_z2 = z2_c - z2_sc;
		float norm_x1 = -(delta_y1) / sqrt(delta_x1*delta_x1 + delta_y1*delta_y1);
		float norm_y1 = delta_x1 / sqrt(delta_x1*delta_x1 + delta_y1*delta_y1);
		float norm_x2 = -(delta_y2) / sqrt(delta_x2*delta_x2 + delta_y2*delta_y2);
		float norm_y2 = delta_x2 / sqrt(delta_x2*delta_x2 + delta_y2*delta_y2);
		float delta_x1x2 = x1_c - x2_c + sign_kk*0.6*norm_x1 - sign_jj*0.6*norm_x2;
		float delta_y1y2 = y1_c - y2_c + sign_kk*0.6*norm_y1 - sign_jj*0.6*norm_y2;
		float delta_z1z2 = z1_c - z2_c + sign_aa*0.6 - sign_bb*0.6;
		float cos_theta_1 = fabs((delta_x1*delta_x1x2 + delta_y1*delta_y1y2 + delta_z1*delta_z1z2) / (sqrt(delta_x1*delta_x1 + delta_y1*delta_y1 + delta_z1*delta_z1)*sqrt(delta_x1x2*delta_x1x2 + delta_y1y2*delta_y1y2 + delta_z1z2*delta_z1z2)));;
		float cos_theta_2 = fabs((delta_x2*delta_x1x2 + delta_y2*delta_y1y2 + delta_z2*delta_z1z2) / (sqrt(delta_x2*delta_x2 + delta_y2*delta_y2 + delta_z2*delta_z2)*sqrt(delta_x1x2*delta_x1x2 + delta_y1y2*delta_y1y2 + delta_z1z2*delta_z1z2)));;
		float distance = delta_x1x2*delta_x1x2 + delta_y1y2*delta_y1y2 + delta_z1z2*delta_z1z2;

		x_offset[2 * tid] = sign_kk*0.6*norm_x1;
		x_offset[2 * tid + 1] = sign_jj*0.6*norm_x2;
		y_offset[2 * tid] = sign_kk*0.6*norm_y1;
		y_offset[2 * tid + 1] = sign_jj*0.6*norm_y2;
		z_offset[2 * tid] = sign_aa*0.6;
		z_offset[2 * tid + 1] = sign_bb*0.6;
		eff[tid] = distance / (cos_theta_1*cos_theta_2);
	}

}

int calculate_imagenorm(float *imgnorm, float *img, float * hist_norm, float * hist_attn, const coordinate3D *crystalPos, const coordinate3D *crystalPos2, const int * index_array, const int n_crystal, const int N_LOR_persubset, const int Nx, const int Ny, const int Nz)
{

	int count = 0;
	int N = 512;
	int N_LOR = n_crystal*(n_crystal - 1) / 2;
	int N_prime = N_LOR_persubset / N;
	float *x_offset = new float[N_LOR_persubset * 2];
	float *y_offset = new float[N_LOR_persubset * 2];
	float *z_offset = new float[N_LOR_persubset * 2];
	float *eff = new float[N_LOR_persubset];
	int *d_index_array = 0;
	float *d_hist_norm = 0;
	float *d_imgnorm = 0;
	float *d_hist_attn = 0;
	float *d_img = 0;
	float *d_hist_fp = 0;
	float *d_eff = 0;
	float *d_x_offset = 0;
	float *d_y_offset = 0;
	float *d_z_offset = 0;
	coordinate3D *d_crystalPos2 = 0;
	coordinate3D *d_crystalPos = 0;
	cudaError_t cudaStatus;
	int sign_jj, sign_kk, sign_aa, sign_bb;
	for (int jj = 0; jj < 2; jj++) {
		for (int kk = 0; kk < 2; kk++) {
			for (int aa = 0; aa < 2; aa++) {
				for (int bb = 0; bb < 2; bb++) {
					/* core code deleted */
					cudaMalloc((void **)&d_crystalPos, n_crystal * sizeof(coordinate3D));
					cudaMalloc((void **)&d_crystalPos2, n_crystal * sizeof(coordinate3D));
					cudaMalloc((void **)&d_index_array, N_LOR_persubset * 3 * sizeof(int));  //copy the index array from host to device;
					cudaMalloc((void **)&d_eff, N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_x_offset, 2 * N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_y_offset, 2 * N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_z_offset, 2 * N_LOR_persubset * sizeof(float));			
					cudaMemcpy(d_crystalPos, crystalPos, n_crystal * sizeof(coordinate3D), cudaMemcpyHostToDevice);
					cudaMemcpy(d_crystalPos2, crystalPos2, n_crystal * sizeof(coordinate3D), cudaMemcpyHostToDevice);
					cudaMemcpy(d_index_array, index_array, N_LOR_persubset * 3 * sizeof(int), cudaMemcpyHostToDevice);
					cudaMemcpy(d_eff, eff, N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_x_offset, x_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_y_offset, y_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_z_offset, z_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
			
					compute_geometry << <N_prime, N >> > (d_crystalPos, d_crystalPos2, d_index_array, d_x_offset, d_y_offset, d_z_offset, d_eff, N_LOR_persubset, sign_jj, sign_kk, sign_aa, sign_bb);
					cudaMemcpy(x_offset, d_x_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(y_offset, d_y_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(z_offset, d_z_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(eff, d_eff, N_LOR_persubset * sizeof(float), cudaMemcpyDeviceToHost);
					cudaFree(d_crystalPos);
					cudaFree(d_crystalPos2);
					cudaFree(d_index_array);
					cudaFree(d_x_offset);
					cudaFree(d_y_offset);
					cudaFree(d_z_offset);
					cudaFree(d_eff);
					cudaStatus = cudaGetLastError();
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "computing geometry failed: %s\n", cudaGetErrorString(cudaStatus));
						//goto Error;
					}
					
					cudaMalloc((void **)&d_crystalPos, n_crystal * sizeof(coordinate3D));
					cudaMalloc((void **)&d_index_array, N_LOR_persubset * 3 * sizeof(int));  //copy the index array from host to device;
					cudaMalloc((void **)&d_hist_norm, N_LOR * sizeof(float));
					cudaMalloc((void **)&d_hist_attn, N_LOR * sizeof(float));
					cudaMalloc((void **)&d_hist_fp, N_LOR * sizeof(float));
					cudaMalloc((void **)&d_imgnorm, Nx*Ny*Nz * sizeof(float));
					cudaMalloc((void **)&d_img, Nx*Ny*Nz * sizeof(float));
					cudaMalloc((void **)&d_eff, N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_x_offset, 2 * N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_y_offset, 2 * N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_z_offset, 2 * N_LOR_persubset * sizeof(float));

					cudaMemcpy(d_crystalPos, crystalPos, n_crystal * sizeof(coordinate3D), cudaMemcpyHostToDevice);
					cudaMemcpy(d_index_array, index_array, N_LOR_persubset * 3 * sizeof(int), cudaMemcpyHostToDevice);
					cudaMemcpy(d_hist_norm, hist_norm, N_LOR * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_hist_attn, hist_attn, N_LOR * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_imgnorm, imgnorm, Nx*Ny*Nz * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_img, img, Nx*Ny*Nz * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_eff, eff, N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_x_offset, x_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_y_offset, y_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_z_offset, z_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);

					RayTraceVoxelsOnCartesianGrid << <N_prime, N >> > (d_crystalPos, d_x_offset, d_y_offset, d_z_offset, d_index_array, d_hist_norm, d_eff, d_hist_fp, d_hist_attn, d_imgnorm, d_img, 1.2, Nx, Ny, Nz, N_LOR_persubset, true, false, false);
					cudaStatus = cudaGetLastError();
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "Ray tracing launch failed: %s\n", cudaGetErrorString(cudaStatus));
						//goto Error;
					}
					cudaMemcpy(imgnorm, d_imgnorm, Nx*Ny*Nz * sizeof(float), cudaMemcpyDeviceToHost);
					cudaFree(d_index_array);
					cudaFree(d_crystalPos);
					cudaFree(d_hist_norm);
					cudaFree(d_hist_fp);
					cudaFree(d_imgnorm);
					cudaFree(d_img);
					cudaFree(d_x_offset);
					cudaFree(d_y_offset);
					cudaFree(d_z_offset);
					cudaFree(d_eff);
					cudaFree(d_hist_attn);
				}
			}
		}
	}

	delete[] x_offset;
	delete[] y_offset;
	delete[] z_offset;
	delete[] eff;
	return 0;

}




void forward_projection(float * img, float * hist_fp, float * hist_norm, float * hist_attn, float * voxelList, const coordinate3D *crystalPos, const coordinate3D *crystalPos2, const int * index_array, const int n_crystal, const int N_LOR_persubset, const int Nx, const int Ny, const int Nz) {

	int count = 0;
	int N = 512;
	int N_LOR = n_crystal*(n_crystal - 1) / 2;
	int N_prime = N_LOR_persubset / N;
	float *x_offset = new float[N_LOR_persubset * 2];
	float *y_offset = new float[N_LOR_persubset * 2];
	float *z_offset = new float[N_LOR_persubset * 2];
	float *eff = new float[N_LOR_persubset];
	int *d_index_array = 0;
	float *d_hist_norm = 0;
	float *d_hist_attn = 0;
	float *d_img = 0;
	float *d_voxelList = 0;
	float *d_hist_fp = 0;
	float *d_eff = 0;
	float *d_x_offset = 0;
	float *d_y_offset = 0;
	float *d_z_offset = 0;
	coordinate3D *d_crystalPos = 0;
	coordinate3D *d_crystalPos2 = 0;
	cudaError_t cudaStatus;
	int sign_jj, sign_kk, sign_aa, sign_bb;
	for (int jj = 0; jj < 2; jj++) {
		for (int kk = 0; kk < 2; kk++) {
			for (int aa = 0; aa < 2; aa++) {
				for (int bb = 0; bb < 2; bb++) {
					/*core code deleted for privacy */
               
					cudaMalloc((void **)&d_crystalPos, n_crystal * sizeof(coordinate3D));
					cudaMalloc((void **)&d_index_array, N_LOR_persubset * 3 * sizeof(int));  //copy the index array from host to device;
					cudaMalloc((void **)&d_hist_norm, N_LOR * sizeof(float));
					cudaMalloc((void **)&d_hist_fp, N_LOR * sizeof(float));
					cudaMalloc((void **)&d_hist_attn, N_LOR * sizeof(float));
					cudaMalloc((void **)&d_img, Nx*Ny*Nz * sizeof(float));
					cudaMalloc((void **)&d_voxelList, Nx*Ny*Nz * sizeof(float));
					cudaMalloc((void **)&d_eff, N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_x_offset, 2 * N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_y_offset, 2 * N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_z_offset, 2 * N_LOR_persubset * sizeof(float));

					cudaMemcpy(d_crystalPos, crystalPos, n_crystal * sizeof(coordinate3D), cudaMemcpyHostToDevice);
					cudaMemcpy(d_index_array, index_array, N_LOR_persubset * 3 * sizeof(int), cudaMemcpyHostToDevice);
					cudaMemcpy(d_hist_norm, hist_norm, N_LOR * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_hist_fp, hist_fp, N_LOR * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_hist_attn, hist_attn, N_LOR * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_img, img, Nx*Ny*Nz * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_voxelList, voxelList, Nx*Ny*Nz * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_eff, eff, N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_x_offset, x_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_y_offset, y_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_z_offset, z_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);

					RayTraceVoxelsOnCartesianGrid << <N_prime, N >> > (d_crystalPos, d_x_offset, d_y_offset, d_z_offset, d_index_array, d_hist_norm, d_eff, d_hist_fp, d_hist_attn, d_voxelList, d_img, 1.2, Nx, Ny, Nz, N_LOR_persubset, false, true, false);
					cudaMemcpy(hist_fp, d_hist_fp, N_LOR * sizeof(float), cudaMemcpyDeviceToHost);
					cudaFree(d_index_array);
					cudaFree(d_crystalPos);
					cudaFree(d_hist_norm);
					cudaFree(d_hist_fp);
					cudaFree(d_voxelList);
					cudaFree(d_img);
					cudaFree(d_x_offset);
					cudaFree(d_y_offset);
					cudaFree(d_z_offset);
					cudaFree(d_eff);
					cudaFree(d_hist_attn);
				}
			}
		}
	}
	/*
	FILE *fp_hist_fp_temp;
	string f_fp_temp;
	f_fp_temp = "hist_fp_forward.s";
	if ((fp_hist_fp_temp = fopen(f_fp_temp.c_str(), "wb")) == NULL)
	{
	cout << "\nMLEM ERROR: can't open " << f_fp_temp << " for write access" << endl;
	return;
	}

	fwrite(hist_fp, sizeof(float), N_LOR, fp_hist_fp_temp);
	fclose(fp_hist_fp_temp);
	*/

	delete[] x_offset;
	delete[] y_offset;
	delete[] z_offset;
	delete[] eff;
}

void backward_projection(float * voxelList, float * imgtemp, float * hist_correct, float * hist_norm, float * hist_attn, const coordinate3D *crystalPos, const coordinate3D *crystalPos2, const int * index_array, const int n_crystal, const int N_LOR_persubset, const int Nx, const int Ny, const int Nz) {

	/* Core code simply deleted for privacy */

					cudaMalloc((void **)&d_crystalPos, n_crystal * sizeof(coordinate3D));
					cudaMalloc((void **)&d_crystalPos2, n_crystal * sizeof(coordinate3D));
					cudaMalloc((void **)&d_index_array, N_LOR_persubset * 3 * sizeof(int));  //copy the index array from host to device;
					cudaMalloc((void **)&d_eff, N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_x_offset, 2 * N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_y_offset, 2 * N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_z_offset, 2 * N_LOR_persubset * sizeof(float));
					cudaMemcpy(d_crystalPos, crystalPos, n_crystal * sizeof(coordinate3D), cudaMemcpyHostToDevice);
					cudaMemcpy(d_crystalPos2, crystalPos2, n_crystal * sizeof(coordinate3D), cudaMemcpyHostToDevice);
					cudaMemcpy(d_index_array, index_array, N_LOR_persubset * 3 * sizeof(int), cudaMemcpyHostToDevice);
					cudaMemcpy(d_eff, eff, N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_x_offset, x_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_y_offset, y_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_z_offset, z_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					compute_geometry << <N_prime, N >> > (d_crystalPos, d_crystalPos2, d_index_array, d_x_offset, d_y_offset, d_z_offset, d_eff, N_LOR_persubset, sign_jj, sign_kk, sign_aa, sign_bb);
					cudaMemcpy(x_offset, d_x_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(y_offset, d_y_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(z_offset, d_z_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(eff, d_eff, N_LOR_persubset * sizeof(float), cudaMemcpyDeviceToHost);
					cudaFree(d_crystalPos);
					cudaFree(d_crystalPos2);
					cudaFree(d_index_array);
					cudaFree(d_x_offset);
					cudaFree(d_y_offset);
					cudaFree(d_z_offset);
					cudaFree(d_eff);

					

					cudaMalloc((void **)&d_crystalPos, n_crystal * sizeof(coordinate3D));
					cudaMalloc((void **)&d_index_array, N_LOR_persubset * 3 * sizeof(int));  //copy the index array from host to device;
					cudaMalloc((void **)&d_hist_norm, N_LOR * sizeof(float));
					cudaMalloc((void **)&d_hist_attn, N_LOR * sizeof(float));
					cudaMalloc((void **)&d_hist_correct, N_LOR * sizeof(float));
					cudaMalloc((void **)&d_imgtemp, Nx*Ny*Nz * sizeof(float));
					cudaMalloc((void **)&d_voxelList, Nx*Ny*Nz * sizeof(float));
					cudaMalloc((void **)&d_eff, N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_x_offset, 2 * N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_y_offset, 2 * N_LOR_persubset * sizeof(float));
					cudaMalloc((void **)&d_z_offset, 2 * N_LOR_persubset * sizeof(float));

					cudaMemcpy(d_crystalPos, crystalPos, n_crystal * sizeof(coordinate3D), cudaMemcpyHostToDevice);
					cudaMemcpy(d_index_array, index_array, N_LOR_persubset * 3 * sizeof(int), cudaMemcpyHostToDevice);
					cudaMemcpy(d_hist_norm, hist_norm, N_LOR * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_hist_attn, hist_attn, N_LOR * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_hist_correct, hist_correct, N_LOR * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_imgtemp, imgtemp, Nx*Ny*Nz * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_voxelList, voxelList, Nx*Ny*Nz * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_eff, eff, N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_x_offset, x_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_y_offset, y_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_z_offset, z_offset, 2 * N_LOR_persubset * sizeof(float), cudaMemcpyHostToDevice);

					RayTraceVoxelsOnCartesianGrid << <N_prime, N >> > (d_crystalPos, d_x_offset, d_y_offset, d_z_offset, d_index_array, d_hist_norm, d_eff, d_hist_correct, d_hist_attn, d_voxelList, d_imgtemp, 1.2, Nx, Ny, Nz, N_LOR_persubset, false, false, true);
					cudaMemcpy(voxelList, d_voxelList, Nx*Ny*Nz * sizeof(float), cudaMemcpyDeviceToHost);
					cudaFree(d_index_array);
					cudaFree(d_crystalPos);
					cudaFree(d_hist_norm);
					cudaFree(d_hist_correct);
					cudaFree(d_voxelList);
					cudaFree(d_imgtemp);
					cudaFree(d_x_offset);
					cudaFree(d_y_offset);
					cudaFree(d_z_offset);
					cudaFree(d_eff);
					cudaFree(d_hist_attn);
				}
			}
		}
	}


	delete[] x_offset;
	delete[] y_offset;
	delete[] z_offset;
	delete[] eff;
}


int main(int argc, char * argv[])
{
	FILE *fp_crystaltable, *fp_crystaltable2, *fp_imgnorm, *fp_hist, *fp_img_iter, *fp_hist_delayed, *fp_hist_scatter, *fp_hist_norm, *fp_hist_attn, *fp_img_initial, *fp_random_seeds;
	int Nx = 127, Ny = 127, Nz = 39;
	int n_crystal = 3072;
	int N_LOR = n_crystal*(n_crystal - 1) / 2;
	int * index_array_original = (int*)malloc(N_LOR * 3 * sizeof(int));
	int N_subset = 16;
	int N_LOR_persubset = N_LOR / N_subset;
	int * LOR_order = (int*)malloc(N_LOR * sizeof(int));
	int **index_array = (int **)malloc(N_subset * sizeof(int *));
	for (int i = 0; i<N_subset; i++)
		index_array[i] = (int *)malloc(N_LOR_persubset * 3 * sizeof(int));
	float * voxelList = (float*)malloc(Nx*Ny*Nz * sizeof(float));
	float ** imgnorm = (float **)malloc(N_subset * sizeof(float *));
	for (int i = 0; i<N_subset; i++)
		imgnorm[i] = (float*)malloc(Nx*Ny*Nz * sizeof(float));
	float * img = (float*)malloc(Nx*Ny*Nz * sizeof(float));
	float * imgallones = (float*)malloc(Nx*Ny*Nz * sizeof(float));
	float * imgback = (float*)malloc(Nx*Ny*Nz * sizeof(float));
	float * imgtemp = (float*)malloc(Nx*Ny*Nz * sizeof(float));
	float * hist_fp = (float*)malloc(n_crystal*(n_crystal - 1) / 2 * sizeof(float));
	float * hist_norm = (float*)malloc(n_crystal*(n_crystal - 1) / 2 * sizeof(float));
	float * hist_attn = (float*)malloc(n_crystal*(n_crystal - 1) / 2 * sizeof(float));
	uint32_t * hist_data2 = (uint32_t *)malloc(n_crystal*(n_crystal - 1) / 2 * sizeof(uint32_t));    //this is float, which is for real data from the scanner, simulation wise, change to float
	uint32_t * hist_data_ran2 = (uint32_t *)malloc(n_crystal*(n_crystal - 1) / 2 * sizeof(uint32_t)); //ditto
	float * hist_data = (float*)malloc(n_crystal*(n_crystal - 1) / 2 * sizeof(float));
	float * hist_data_ran = (float*)malloc(n_crystal*(n_crystal - 1) / 2 * sizeof(float));
	float * hist_scatter = (float*)malloc(n_crystal*(n_crystal - 1) / 2 * sizeof(float));
	float * hist_correct = (float*)malloc(n_crystal*(n_crystal - 1) / 2 * sizeof(float));
	clock_t start_t, end_t;
	//vector<int> LOR_order;
	//float * img_test = new float[Nx*Ny*Nz];

	double total_t;
	int divzero = 0;
	for (int i = 0; i<Nx*Ny*Nz; i++) {

		imgtemp[i] = 1;
		imgallones[i] = 1;
		img[i] = 1;
		for (int j = 0; j<N_subset; j++)
		  imgnorm[j][i] = 0;
		voxelList[i] = 0;

	}
	for (int i = 0; i<n_crystal*(n_crystal - 1) / 2; i++) {
		hist_correct[i] = 1;
	}
	int iter = 1;
	int niter = 50;
	start_t = clock();
	//string f_crystaltable = "crystal_position_endpoints.txt"; // "crystal_position_smaller_radius.txt";
	//string f_crystaltable2 = "crystal_position_surface.txt";
	string f_crystaltable, f_crystaltable2, f_hist, f_hist_delayed, f_hist_scatter, f_hist_norm, f_hist_attn, f_random_seeds;
	string dest_directory;
	string f_output;
	/*cout << "Please type in the prompt file name:" << endl;
	cin >> f_hist;
	cout << "Please type in the delayed file name:" << endl;
	cin >> f_hist_delayed;
	cout << "Please type in the scatter file name:" << endl;
	cin >> f_hist_scatter;
	cout << "Please type in the normalization file name:" << endl;
	cin >> f_hist_norm;
	cout << "Please type in the attenuation file name:" << endl;
	cin >> f_hist_attn;
	cout << "Please type in the output file name:" << endl;
	cin >> f_output;
	cout << "Please name the directory that you would like to save your reconstructed images in:" << endl;
	cin >> dest_directory;
	cout << "Processing..." << endl;*/
	f_crystaltable = argv[1];
	f_crystaltable2 = argv[2];
	f_hist = argv[3];
	f_hist_delayed = argv[4];
	f_hist_scatter = argv[5];
	f_hist_norm = argv[6];
	f_hist_attn = argv[7];
	dest_directory = argv[8];
	f_output = argv[9];
	f_random_seeds = "C:/Users/wsy88/Documents/Visual Studio 2015/Projects/OSEM_GPU/OSEM_GPU/random_seeds.s";
	cout << dest_directory << endl;
	if (0 == CreateDirectory(dest_directory.c_str(), NULL))
	{
		// Error encountered; generate message and exit.
		cout << "Failed CreateDirectory!" << endl;;
		return 0;
	}

	/*
	string f_hist = "Hist_prompt_input_function_08042017_0003.s";
	string f_hist_delayed = "Hist_delayed_input_function_08042017_0003.s";
	string f_hist_norm = "Hist_attn_norm_uniform_08032017.s";
	*/
	if ((fp_crystaltable = fopen(f_crystaltable.c_str(), "rb")) == NULL)
	{
		cout << "\nMLEM ERROR: can't open " << f_crystaltable << " for read access" << endl;
		return 0;
	}
	//cout << "okay" << endl;
	if ((fp_crystaltable2 = fopen(f_crystaltable2.c_str(), "rb")) == NULL)
	{
		cout << "\nMLEM ERROR: can't open " << f_crystaltable2 << " for read access" << endl;
		return 0;
	}
	//cout << "okay" << endl;
	if ((fp_hist = fopen(f_hist.c_str(), "rb")) == NULL)
	{
		cout << "\nMLEM ERROR: can't open " << f_hist << " for read access" << endl;
		return 0;
	}
	//cout << "okay" << endl;
	if ((fp_hist_delayed = fopen(f_hist_delayed.c_str(), "rb")) == NULL)
	{
		cout << "\nMLEM ERROR: can't open " << f_hist << " for read access" << endl;
		return 0;
	}
	//cout << "okay" << endl;
	if ((fp_hist_scatter = fopen(f_hist_scatter.c_str(), "rb")) == NULL)
	{
		cout << "\nMLEM ERROR: can't open " << f_hist << " for read access" << endl;
		return 0;
	}
	//cout << "okay" << endl;
	if ((fp_hist_norm = fopen(f_hist_norm.c_str(), "rb")) == NULL)
	{
		cout << "\nMLEM ERROR: can't open " << f_hist_norm << " for read access" << endl;
		return 0;
	}
	//cout << "okay" << endl;
	if ((fp_hist_attn = fopen(f_hist_attn.c_str(), "rb")) == NULL)
	{
		cout << "\nMLEM ERROR: can't open " << f_hist_attn << " for read access" << endl;
		return 0;
	}
	if ((fp_random_seeds = fopen(f_random_seeds.c_str(), "rb")) == NULL)
	{
		cout << "\nMLEM ERROR: can't open " << f_random_seeds << " for read access" << endl;
		return 0;
	}

	cout << "okay" << endl;
	fread(hist_data2, sizeof(uint32_t), n_crystal*(n_crystal - 1) / 2, fp_hist);     //this is float, which is for real data from the scanner, simulation wise, change to float
	fread(hist_data_ran2, sizeof(uint32_t), n_crystal*(n_crystal - 1) / 2, fp_hist_delayed);  //ditto
	fread(hist_scatter, sizeof(float), n_crystal*(n_crystal - 1) / 2, fp_hist_scatter);
	fread(hist_norm, sizeof(float), n_crystal*(n_crystal - 1) / 2, fp_hist_norm);
	fread(hist_attn, sizeof(float), n_crystal*(n_crystal - 1) / 2, fp_hist_attn);
	fread(LOR_order, sizeof(int), N_LOR, fp_random_seeds);
	/*for (int i = 0; i < N_LOR; i++) {
	if (hist_norm[i] != 1000000) {
	//cout << "hist_norm[i] =" << hist_norm[i] << endl;
	hist_norm[i] = hist_norm[i]];
	//cout << "hist_norm[i] =" << hist_norm[i] << endl;
	}
	}*/

	coordinate3D *crystalPos = (coordinate3D*)malloc(n_crystal * sizeof(coordinate3D));
	coordinate3D *crystalPos2 = (coordinate3D*)malloc(n_crystal * sizeof(coordinate3D));

	float *xloc = (float*)malloc(n_crystal * sizeof(float));
	float *yloc = (float*)malloc(n_crystal * sizeof(float));
	float *zloc = (float*)malloc(n_crystal * sizeof(float));

	for (int i = 0; i<n_crystal*(n_crystal - 1) / 2; i++) {
		hist_data[i] = (float)hist_data2[i];
		hist_data_ran[i] = (float)hist_data_ran2[i];
	}

	for (int crystal = 0; crystal<n_crystal; crystal++)
	{
		fscanf(fp_crystaltable, "%f %f %f", &xloc[crystal], &yloc[crystal], &zloc[crystal]);
		crystalPos[crystal].x = xloc[crystal];
		crystalPos[crystal].y = yloc[crystal];
		crystalPos[crystal].z = zloc[crystal];
	}
	fclose(fp_crystaltable);


	for (int crystal = 0; crystal<n_crystal; crystal++)
	{
		fscanf(fp_crystaltable2, "%f %f %f", &xloc[crystal], &yloc[crystal], &zloc[crystal]);
		crystalPos2[crystal].x = xloc[crystal];
		crystalPos2[crystal].y = yloc[crystal];
		crystalPos2[crystal].z = zloc[crystal];
	}
	fclose(fp_crystaltable2);

	int count = 0;
	
	/*for (int i = 0; i < N_LOR; i++) {
		LOR_order[i] = i;
	}*/
	//randomize(LOR_order, N_LOR);
	
	//shuffle(LOR_order.begin(), LOR_order.end());
	//shuffle(LOR_order, N_LOR);
	count = 0;
	for (int i = 0; i < n_crystal; i++) {
		for (int j = i + 1; j < n_crystal; j++) {
			int id = 0;
			for (int x = 0; x < i; x++) {
				id += n_crystal - x - 1;
			}
			id += j - i - 1;
			//cout << "LOR_order[count] = " << LOR_order[count] << endl;
			index_array_original[LOR_order[count] * 3] = i;
			index_array_original[LOR_order[count] * 3 + 1] = j;
			index_array_original[LOR_order[count] * 3 + 2] = id;
			//if (hist_norm[id] == 1)
			//	hist_norm[id] = 1000000;
			count++;
		}
	}

	for (int i = 0; i < N_subset; i++) {
		for (int j = 0; j < N_LOR_persubset; j++) {
			index_array[i][j * 3] = index_array_original[(i*N_LOR_persubset + j) * 3];
			index_array[i][j * 3 + 1] = index_array_original[(i*N_LOR_persubset + j) * 3 + 1];
			index_array[i][j * 3 + 2] = index_array_original[(i*N_LOR_persubset + j) * 3 + 2];
		}
	}
	//cout << "okay again!" << endl;
	/*string f_imgnorm = dest_directory + "/" + "imgnorm_withnorm.v";
	if ((fp_imgnorm = fopen(f_imgnorm.c_str(), "wb")) == NULL)
	{
		cout << "\nMLEM ERROR: can't open " << f_imgnorm << " for write access" << endl;
		return 0;
	}

	fwrite(imgnorm, sizeof(float), Nx*Ny*Nz, fp_imgnorm);
	fclose(fp_imgnorm);
	*/

	/*string f_img_initial = "img_smalldisk.v";

	if ((fp_img_initial = fopen(f_img_initial.c_str(), "rb")) == NULL)
	{
	cout << "\nMLEM ERROR: can't open " << f_img_initial << " for read access" << endl;
	return 0;
	}
	fread(img_test, sizeof(float), Nx*Ny*Nz, fp_img_initial);
	fclose(fp_img_initial);*/

	while (iter <= niter)
	{
		for (int subset = 0; subset < N_subset; subset++) {
			//cout << "okay begins!" << endl;

			if (iter == 1) {
				for (int i = 0; i < Nx*Ny*Nz; i++) {
					imgnorm[subset][i] = 0;
				}
				calculate_imagenorm(imgnorm[subset], img, hist_norm, hist_attn, crystalPos, crystalPos2, index_array[subset], n_crystal, N_LOR_persubset, Nx, Ny, Nz);
			}
			/*string f_imgnorm = dest_directory + "/" + "imgnorm_withnorm"+to_string(subset)+".v";
			if ((fp_imgnorm = fopen(f_imgnorm.c_str(), "wb")) == NULL)
			{
				cout << "\nMLEM ERROR: can't open " << f_imgnorm << " for write access" << endl;
				return 0;
			}

			fwrite(imgnorm[subset], sizeof(float), Nx*Ny*Nz, fp_imgnorm);
			fclose(fp_imgnorm);
			*/

			for (int i = 0; i < n_crystal*(n_crystal - 1) / 2; i++) {
				hist_fp[i] = 0;
			}

			forward_projection(img, hist_fp, hist_norm, hist_attn, voxelList, crystalPos, crystalPos2, index_array[subset], n_crystal, N_LOR_persubset, Nx, Ny, Nz);
			//cout << "okay!" << endl;
			//BACK PROJECTION	
			for (int i = 0; i < N_LOR_persubset; i++)
			{
				int LOR_index = index_array[subset][3 * i + 2];
				if (hist_fp[LOR_index] != 0) {
					//cout << "hist_scatter = " << hist_scatter[i] << endl;
					hist_correct[LOR_index] = hist_data[LOR_index] / (hist_fp[LOR_index] + hist_data_ran[LOR_index] + hist_scatter[LOR_index]);
				}
				else
				{
					if (hist_data_ran[LOR_index] + hist_scatter[LOR_index] != 0)
						hist_correct[LOR_index] = hist_data[LOR_index] / (hist_data_ran[LOR_index] + hist_scatter[LOR_index]);
					else
						hist_correct[LOR_index] = 0;

					if (hist_data[LOR_index] != 0) divzero++;
				}
			}

			for (int i = 0; i < Nx*Ny*Nz; i++) {
				imgback[i] = 0;
			}

			backward_projection(imgback, imgtemp, hist_correct, hist_norm, hist_attn, crystalPos, crystalPos2, index_array[subset], n_crystal, N_LOR_persubset, Nx, Ny, Nz);
			//cout << "okay again!" << endl;
			for (int i = 0; i < Nx*Ny*Nz; i++)
				if (imgnorm[subset][i] > 0)
					img[i] = img[i] * imgback[i] / imgnorm[subset][i];
				else
					img[i] = 0;
			//cout << "okay again again!" << endl;
			/*string f_img_iter = dest_directory + '/' + f_output + to_string(iter) + "_" + to_string(subset) + ".v"; //"inputfunction_08042017_0003_withnorm"

			if ((fp_img_iter = fopen(f_img_iter.c_str(), "wb")) == NULL)
			{
				cout << "\nMLEM ERROR: can't open " << f_img_iter << " for write access" << endl;
				return 0;
			}
			fwrite(img, sizeof(float), Nx*Ny*Nz, fp_img_iter);
			fclose(fp_img_iter);
			*/

		}
		//if (iter % 10 == 0 || iter == 1) {
		string f_img_iter = dest_directory + '/' + f_output + to_string(iter) + ".v"; //"inputfunction_08042017_0003_withnorm"

		if ((fp_img_iter = fopen(f_img_iter.c_str(), "wb")) == NULL)
		{
			cout << "\nMLEM ERROR: can't open " << f_img_iter << " for write access" << endl;
			return 0;
		}
		fwrite(img, sizeof(float), Nx*Ny*Nz, fp_img_iter);
		fclose(fp_img_iter);
		
		cout << "finished the No." << iter << "iteration!" << endl;
		iter++;
	}


	end_t = clock();
	total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	cout << "Total time taken by CPU for the reconstruction: " << total_t << endl;
	fclose(fp_hist_attn);
	fclose(fp_hist_norm);
	fclose(fp_hist_delayed);
	fclose(fp_hist);
	fclose(fp_hist_scatter);
	delete[] index_array;
	delete[] voxelList;
	delete[] imgnorm;
	delete[] img;
	delete[] imgback;
	delete[] imgtemp;
	delete[] hist_fp;
	delete[] hist_norm;
	delete[] hist_attn;
	delete[] hist_data2;
	delete[] hist_data_ran2;
	delete[] hist_data;
	delete[] hist_data_ran;
	delete[] hist_correct;
	fclose(fp_hist);
	fclose(fp_hist_delayed);
	fclose(fp_hist_norm);


	return 0;
}


