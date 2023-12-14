#include <algorithm>
#include <math.h>
#include <float.h>
#include <utility>
#include <stdio.h>
#include <cstring>
#include <vector>
#include <unistd.h>
#include "cudaRenderer.h"
#include "image.h"
#include "util.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define CELL_DIM 1
#define TIME_STEP 1
#define BLOCKSIDE 16
#define BLOCKSIZE BLOCKSIDE*BLOCKSIDE
///////////////////////////CUDA CODE BELOW////////////////////////////////
struct GlobalConstants {
    int cells_per_side;
    int width;
    int height;

    float* VX;
    float* VY;
    float* pressures;
    float* pressuresCopy;
    float* VXCopy;
    float* VYCopy;
    float* divergence;
    float* vorticity;
    float* color;
    float* colorCopy;
    float* imageData;

    int* mpls;
};

__constant__ GlobalConstants cuParams;


__global__ void kernelClearImage(float r, float g, float b, float a) {
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuParams.width;
    int height = cuParams.height;

    if (imageX >= width || imageY >= height) return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r,g,b,a);
    
    *(float4*)(&cuParams.imageData[offset]) = value;

}

// Parameters structure to avoid global variables
struct Params {
    int cells_per_side;
    float *VX, *VY, *VXCopy, *VYCopy;
    int *mpls; // Mouse point locations
    float *imageData;
    int width, height;
    // Add other necessary parameters
};

__device__ __inline__ int isBoundary(int i, int j, int cells_per_side) {
    if (j == 0) return 1; // left 
    if (i == 0) return 2; // top
    if (j == cells_per_side) return 3; // right
    if (i == cells_per_side) return 4; // bottom
    return 0;
}

__device__ __inline__ int isInBox(int row, int col, int blockDimX, int blockDimY, int blockIdxX, int blockIdxY) {
    int minRow = blockIdxY * blockDimY;
    int maxRow = minRow + blockDimY;
    int minCol = blockIdxX * blockDimX;
    int maxCol = minCol + blockDimX;
    return (row >= minRow && row < maxRow && col >= minCol && col < maxCol);
}

__device__ __inline__ double distanceToSegment(double ax, double ay, double bx, double by, double px, double py, double* fp) {
    double dx = px - ax;
    double dy = py - ay;
    double xx = bx - ax;
    double xy = by - ay;
    *fp = 0.0;
    double lx = sqrt(xx*xx + xy*xy);
    double ld = sqrt(dx*dx + dy*dy);
    if (lx <= 0.0001) return ld;
    double projection = dx*(xx/lx) + dy*(xy/lx);
    *fp = projection / lx;
    if (projection < 0.0) return ld;
    else if (projection > lx) return sqrt((px-bx) * (px-bx) + (py-by) * (py-by));
    return sqrt(abs(dx*dx + dy*dy - projection * projection));
}

__device__ __inline__ double distanceToNearestMouseSegment(double px, double py, double *fp, double* vx, double *vy, const Params& cuParams) {
    double minLen = DBL_MAX;
    double fpResult = 0.0;
    double vxResult = 0.0;
    double vyResult = 0.0;
    for (int i = 0; i < 400 - 2; i += 2) {
        int grid_col1 = cuParams.mpls[i];
        int grid_row1 = cuParams.mpls[i + 1];
        int grid_col2 = cuParams.mpls[i + 2];
        int grid_row2 = cuParams.mpls[i + 3];
        if (grid_col2 == 0 & grid_row2 == 0) break;
        double len = distanceToSegment(grid_col1, grid_row1, grid_col2, grid_row2, px, py, fp);
        if (len < minLen) {
            minLen = len;
            fpResult = *fp;
            vxResult = grid_col2 - grid_col1;
            vyResult = grid_row2 - grid_row1;
        }
    }
    *fp = fpResult;
    *vx = vxResult;
    *vy = vyResult;
    return minLen;
}
__global__ void kernelAdvectVelocityBackward(const Params cuParams) {
    const int colOnScreen = blockIdx.x * blockDim.x + threadIdx.x;
    const int rowOnScreen = blockIdx.y * blockDim.y + threadIdx.y; 

    if (colOnScreen >= cuParams.width || rowOnScreen >= cuParams.height) return;

    const int index = rowOnScreen * cuParams.width + colOnScreen;
    if (index >= cuParams.width * cuParams.height) return;

    __shared__ float sharedVX[BLOCKSIZE]; 
    __shared__ float sharedVY[BLOCKSIZE]; 

    sharedVX[threadIdx.y * blockDim.x + threadIdx.x] = cuParams.VXCopy[index];
    sharedVY[threadIdx.y * blockDim.x + threadIdx.x] = cuParams.VYCopy[index];
    __syncthreads();

    const int prevRowOnScreen = round(rowOnScreen - TIME_STEP * cuParams.VYCopy[index]);
    const int prevColOnScreen = round(colOnScreen - TIME_STEP * cuParams.VXCopy[index]);

    if (prevColOnScreen < cuParams.cells_per_side && prevRowOnScreen < cuParams.cells_per_side 
        && prevColOnScreen >= 0 && prevRowOnScreen >= 0) {
        const int prevIndex = prevRowOnScreen * cuParams.width + prevColOnScreen;
        if (isInBox(prevRowOnScreen, prevColOnScreen, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
            const int r = prevRowOnScreen % blockDim.y;
            const int c = prevColOnScreen % blockDim.x;
            cuParams.VX[index] = sharedVX[r * blockDim.x + c];
            cuParams.VY[index] = sharedVY[r * blockDim.x + c];
        } else {
            cuParams.VX[index] = cuParams.VXCopy[prevIndex];
            cuParams.VY[index] = cuParams.VYCopy[prevIndex];
        }
    } 

    if (prevColOnScreen == colOnScreen && prevRowOnScreen == rowOnScreen) {
        // Particle doesn't move, so set velocity to zero
        cuParams.VX[index] = 0;
        cuParams.VY[index] = 0;
    }
}

__global__ void kernelApplyVorticity(const Params cuParams) {
    const int colOnScreen = blockIdx.x * blockDim.x + threadIdx.x;
    const int rowOnScreen = blockIdx.y * blockDim.y + threadIdx.y; 

    if (colOnScreen >= cuParams.width || rowOnScreen >= cuParams.height) return;

    const int index = rowOnScreen * cuParams.width + colOnScreen;
    if (index >= cuParams.width * cuParams.height) return;

    __shared__ float sharedVX[BLOCKSIZE];
    __shared__ float sharedVY[BLOCKSIZE];

    sharedVX[threadIdx.y * blockDim.x + threadIdx.x] = cuParams.VX[index];
    sharedVY[threadIdx.y * blockDim.x + threadIdx.x] = cuParams.VY[index];

    __syncthreads();

    if (!isBoundary(rowOnScreen, colOnScreen, cuParams.cells_per_side)) {
        float L = 0.0, R = 0.0, B = 0.0, T = 0.0;
        getNeighborVelocities(rowOnScreen, colOnScreen, sharedVX, sharedVY, L, R, B, T, cuParams);
        cuParams.vorticity[index] = 0.5f * ((R - L) - (T - B));
    }
}

__global__ void kernelApplyVorticityForce(const Params cuParams) {
    const int colOnScreen = blockIdx.x * blockDim.x + threadIdx.x;
    const int rowOnScreen = blockIdx.y * blockDim.y + threadIdx.y; 

    if (colOnScreen >= cuParams.width || rowOnScreen >= cuParams.height) return;

    const int index = rowOnScreen * cuParams.width + colOnScreen;
    if (index >= cuParams.width * cuParams.height) return;

    __shared__ float sharedVort[BLOCKSIZE];
    sharedVort[threadIdx.y * blockDim.x + threadIdx.x] = cuParams.vorticity[index];

    __syncthreads();

    if (!isBoundary(rowOnScreen, colOnScreen, cuParams.cells_per_side)) {
        float vortConfinementFloat = 0.035f;
        float vortL = 0.0, vortR = 0.0, vortB = 0.0, vortT = 0.0, vortC = 0.0;
        getNeighborVorticities(rowOnScreen, colOnScreen, sharedVort, vortL, vortR, vortB, vortT, cuParams);
        vortC = cuParams.vorticity[index];

        float forceX = 0.5f * (fabsf(vortT) - fabsf(vortB));
        float forceY = 0.5f * (fabsf(vortR) - fabsf(vortL));
        float EPSILON = powf(2,-12);
        float magSqr = fmaxf(EPSILON, forceX * forceX + forceY * forceY);
        forceX *= vortConfinementFloat * vortC * (1/sqrtf(magSqr));
        forceY *= vortConfinementFloat * vortC * (-1/sqrtf(magSqr));
        cuParams.VX[index] += forceX;
        cuParams.VY[index] += forceY;
    }
}

//kernelApplyDivergence
__global__ void kernelApplyDivergence() {
    int cells_per_side = cuParams.cells_per_side;
    int rowInBox = threadIdx.y;
    int colInBox = threadIdx.x;
    int boxWidth = blockDim.x;
    int colOnScreen = blockIdx.x * blockDim.x + threadIdx.x;
    int rowOnScreen = blockIdx.y * blockDim.y + threadIdx.y; 
    int width = cuParams.width;
    int height = cuParams.height;

    if (rowOnScreen * width + colOnScreen >= width * height) return; 

    __shared__ float sharedVX[BLOCKSIZE];
    __shared__ float sharedVY[BLOCKSIZE];
    sharedVX[rowInBox * boxWidth + colInBox] = 
        cuParams.VX[rowOnScreen * width + colOnScreen];
    sharedVY[rowInBox * boxWidth + colInBox] = 
        cuParams.VY[rowOnScreen * width + colOnScreen];
     __syncthreads();

    if (!isBoundary(rowOnScreen,colOnScreen)) {
        float L = 0.0;
        float R = 0.0;
        float B = 0.0;
        float T = 0.0;
        int r = 0;
        int c = 0;

        if (rowOnScreen > 0) {
            if (isInBox(rowOnScreen-1, colOnScreen, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
                r = (rowOnScreen - 1) % blockDim.y;
                c = colOnScreen % blockDim.x;
                T = sharedVY[r * boxWidth + c];
            } else {
                T = cuParams.VY[(rowOnScreen-1) * width + colOnScreen];
            }
        }
        if (rowOnScreen < cells_per_side) {
            if (isInBox(rowOnScreen+1, colOnScreen, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
                r = (rowOnScreen + 1) % blockDim.y;
                c = colOnScreen % blockDim.x;
                B = sharedVY[r * boxWidth + c];
            } else {
                B = cuParams.VY[(rowOnScreen+1) * width + colOnScreen];
            }
        }
        if (colOnScreen < cells_per_side) {
            if (isInBox(rowOnScreen, colOnScreen+1, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
                r = rowOnScreen % blockDim.y;
                c = (colOnScreen+1) % blockDim.x;
                R = sharedVX[r * boxWidth + c];
            } else {
                R = cuParams.VX[rowOnScreen * width + (colOnScreen+1)];
            }
        }
        if (colOnScreen > 0) {
            if (isInBox(rowOnScreen, colOnScreen-1, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
                r = rowOnScreen % blockDim.y;
                c = (colOnScreen-1) % blockDim.x;
                L = sharedVX[r * boxWidth + c];
            } else {
                L = cuParams.VX[rowOnScreen * width + (colOnScreen-1)];
            }
        }
        cuParams.divergence[rowOnScreen * width + colOnScreen] = 0.5*((R-L) + (T-B));
    }
}

//kernelCopyPressures
__global__ void kernelCopyPressures() {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int width = cuParams.width;
    int height = cuParams.height;

    if (col >= width || row >= height) return;
    if (row * width + col >= width * height) return; 

    cuParams.pressuresCopy[row * width + col] = cuParams.pressures[row * width + col];
    cuParams.pressuresCopy[row * width + col] = cuParams.pressures[row * width + col];
}

//kernelPressureSolve
__global__ void kernelPressureSolve(){
    int cells_per_side = cuParams.cells_per_side;
    int rowInBox = threadIdx.y;
    int colInBox = threadIdx.x;
    int boxWidth = blockDim.x;
    int colOnScreen = blockIdx.x * blockDim.x + threadIdx.x;
    int rowOnScreen = blockIdx.y * blockDim.y + threadIdx.y; 
    int width = cuParams.width;
    int height = cuParams.height;

    if (rowOnScreen * width + colOnScreen >= width * height) return; 

    __shared__ float sharedPressuresCopy[BLOCKSIZE];
    sharedPressuresCopy[rowInBox * boxWidth + colInBox] =
       cuParams.pressuresCopy[rowOnScreen * width + colOnScreen];
    __syncthreads();
    
    if (!isBoundary(rowOnScreen,colOnScreen)) {
        float L = 0.0;
        float R = 0.0;
        float B = 0.0;
        float T = 0.0;
        int r = 0;
        int c = 0;

        if (rowOnScreen > 0) {
            if (isInBox(rowOnScreen-1, colOnScreen, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
                r = (rowOnScreen - 1) % blockDim.y;
                c = colOnScreen % blockDim.x;
                T = sharedPressuresCopy[r * boxWidth + c];
            } else {
                T = cuParams.pressuresCopy[(rowOnScreen-1) * width + colOnScreen];
            }
        }
        if (rowOnScreen < cells_per_side) {
            if (isInBox(rowOnScreen+1, colOnScreen, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
                r = (rowOnScreen + 1) % blockDim.y;
                c = colOnScreen % blockDim.x;
                B = sharedPressuresCopy[r * boxWidth + c];
            } else {
                B = cuParams.pressuresCopy[(rowOnScreen+1) * width + colOnScreen];
            }
        }
        if (colOnScreen < cells_per_side) {
            if (isInBox(rowOnScreen, colOnScreen+1, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
                r = rowOnScreen % blockDim.y;
                c = (colOnScreen+1) % blockDim.x;
                R = sharedPressuresCopy[r * boxWidth + c];
            } else { 
                R = cuParams.pressuresCopy[rowOnScreen * width + (colOnScreen+1)];
            }
        }
        if (colOnScreen > 0) {
            if (isInBox(rowOnScreen, colOnScreen-1, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
                r = rowOnScreen % blockDim.y;
                c = (colOnScreen-1) % blockDim.x;
                L = sharedPressuresCopy[r * boxWidth + c];
            } else {
                L = cuParams.pressuresCopy[rowOnScreen * width + (colOnScreen-1)];
            }
        }
        cuParams.pressures[rowOnScreen * width + colOnScreen] = 
            (L + R + B + T + -1 * cuParams.divergence[rowOnScreen * width + colOnScreen]) * .25;
    }
}

//kernelPressureGradient
__global__ void kernelPressureGradient(){
    int cells_per_side = cuParams.cells_per_side;
    int rowInBox = threadIdx.y;
    int colInBox = threadIdx.x;
    int boxWidth = blockDim.x;
    int colOnScreen = blockIdx.x * blockDim.x + threadIdx.x;
    int rowOnScreen = blockIdx.y * blockDim.y + threadIdx.y; 
    int width = cuParams.width;
    int height = cuParams.height;

    if (rowOnScreen * width + colOnScreen >= width * height) return; 

    __shared__ float sharedPressures[BLOCKSIZE];
    sharedPressures[rowInBox * boxWidth + colInBox] = 
        cuParams.pressures[rowOnScreen * width + colOnScreen];
    __syncthreads(); //now everything in the box should be loaded into shared mem.
    
    if (!isBoundary(rowOnScreen,colOnScreen)) {

        float L = 0.0;
        float R = 0.0;
        float B = 0.0;
        float T = 0.0;
        int r = 0;
        int c = 0;

        if (rowOnScreen > 0) {
            if (isInBox(rowOnScreen-1, colOnScreen, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
                r = (rowOnScreen - 1) % blockDim.y;
                c = colOnScreen % blockDim.x;
                T = sharedPressures[r * boxWidth + c];
            } else { 
                T = cuParams.pressures[(rowOnScreen-1) * width + colOnScreen];
            }
        }
        if (rowOnScreen < cells_per_side) {
            if (isInBox(rowOnScreen+1, colOnScreen, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
                r = (rowOnScreen + 1) % blockDim.y;
                c = colOnScreen % blockDim.x;
                B = sharedPressures[r * boxWidth + c];
            } else {
                B = cuParams.pressures[(rowOnScreen+1) * width + colOnScreen];
            }
        }
        if (colOnScreen < cells_per_side) {
            if (isInBox(rowOnScreen, colOnScreen+1, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
                r = rowOnScreen % blockDim.y;
                c = (colOnScreen+1) % blockDim.x;
                R = sharedPressures[r * boxWidth + c];
            } else {
                R = cuParams.pressures[rowOnScreen * width + (colOnScreen+1)];
            }
        }
        if (colOnScreen > 0) {
            if (isInBox(rowOnScreen, colOnScreen-1, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
                r = rowOnScreen % blockDim.y;
                c = (colOnScreen-1) % blockDim.x;
                L = sharedPressures[r * boxWidth + c];
            } else {
                L = cuParams.pressures[rowOnScreen * width + (colOnScreen-1)];
            }
        }
        cuParams.VX[rowOnScreen * width + colOnScreen] = cuParams.VX[rowOnScreen * width + colOnScreen] - 0.5*(R - L);
        cuParams.VY[rowOnScreen * width + colOnScreen] = cuParams.VY[rowOnScreen * width + colOnScreen] - 0.5*(T - B);
    }
}
__global__ void kernelCopyColor(const Params cuParams) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= cuParams.width || row >= cuParams.height) return;

    const int index = 4 * (row * cuParams.width + col);
    for (int i = 0; i < 4; ++i) {
        cuParams.colorCopy[index + i] = cuParams.color[index + i];
    }
}

__global__ void kernelAdvectColorForward(const Params cuParams) {
    const int colOnScreen = blockIdx.x * blockDim.x + threadIdx.x;
    const int rowOnScreen = blockIdx.y * blockDim.y + threadIdx.y;

    if (colOnScreen >= cuParams.width || rowOnScreen >= cuParams.height) return;

    __shared__ float sharedColorCopy[4 * BLOCKSIZE];
    const int sharedIndex = (threadIdx.y * blockDim.x + threadIdx.x) * 4;
    const int globalIndex = (rowOnScreen * cuParams.width + colOnScreen) * 4;

    for (int i = 0; i < 4; ++i) {
        sharedColorCopy[sharedIndex + i] = cuParams.colorCopy[globalIndex + i];
    }
    __syncthreads();

    const int nextRowOnScreen = round(rowOnScreen + TIME_STEP * cuParams.VY[globalIndex / 4]);
    const int nextColOnScreen = round(colOnScreen + TIME_STEP * cuParams.VX[globalIndex / 4]);

    if (nextColOnScreen < cuParams.cells_per_side && nextRowOnScreen < cuParams.cells_per_side &&
        nextColOnScreen >= 0 && nextRowOnScreen >= 0) {
        const int nextGlobalIndex = (nextRowOnScreen * cuParams.width + nextColOnScreen) * 4;
        if (isInBox(nextRowOnScreen, nextColOnScreen, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y)) {
            const int r = nextRowOnScreen % blockDim.y;
            const int c = nextColOnScreen % blockDim.x;
            const int nextSharedIndex = (r * blockDim.x + c) * 4;
            for (int i = 0; i < 4; ++i) {
                cuParams.color[nextGlobalIndex + i] = sharedColorCopy[nextSharedIndex + i];
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                cuParams.color[nextGlobalIndex + i] = cuParams.colorCopy[globalIndex + i];
            }
        }
    }
}



//////////////////////////////////////////////////////////////////////////
///////////////////////////HOST CODE BELOW////////////////////////////////
//////////////////////////////////////////////////////////////////////////

CudaRenderer::CudaRenderer() {
    image = NULL;

    VX = NULL;
    VY = NULL;
    color = NULL;
    colorCopy = NULL;
    pressures = NULL;
    pressuresCopy = NULL;
    VXCopy = NULL;
    VYCopy = NULL;
    divergence = NULL;
    vorticity = NULL;

    mpls = NULL;

    cdVX = NULL;
    cdVY = NULL;
    cdColor = NULL;
    cdColorCopy = NULL;
    cdPressures = NULL;
    cdPressuresCopy = NULL;
    cdVXCopy = NULL;
    cdVYCopy = NULL;
    cdDivergence = NULL;
    cdVorticity = NULL;
    cdImageData = NULL;

    cdMpls = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) delete image;

    if (VX) {
        delete VX;
        delete VY;
        delete pressures;
        delete pressuresCopy;
        delete VXCopy;
        delete VYCopy;
        delete divergence;
        delete vorticity;
        delete color;
        delete colorCopy;
        delete mpls;
    }

    if (cdVX) {
        cudaFree(cdVX);
        cudaFree(cdVY);
        cudaFree(cdPressures);
        cudaFree(cdPressuresCopy);
        cudaFree(cdVXCopy);
        cudaFree(cdVYCopy);
        cudaFree(cdDivergence);
        cudaFree(cdVorticity);
        cudaFree(cdColor);
        cudaFree(cdColorCopy);
        cudaFree(cdImageData);
        cudaFree(cdMpls);
    }
}

const Image*
CudaRenderer::getImage() {
    printf("Copying image data from device\n");

    cudaMemcpy(image->data, cdImageData, 
            4 * sizeof(float) * image->width * image->height,
            cudaMemcpyDeviceToHost);

    return image;
}

__global__ void kernelDrawColor(const Params cuParams, int mplsSize) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= cuParams.width || row >= cuParams.height) return;

    const int index = 4 * (row * cuParams.width + col);
    double vx = cuParams.VX[row * cuParams.width + col];
    double vy = cuParams.VY[row * cuParams.width + col];
    double v = sqrt(vx * vx + vy * vy);

    // Apply fading effect
    if (abs(v) < 0.00001) {
        for (int i = 0; i < 3; ++i) {
            cuParams.color[index + i] *= 0.9;
        }
    } else {
        for (int i = 0; i < 3; ++i) {
            cuParams.color[index + i] *= 0.9494;
        }
    }
    cuParams.color[index + 3] = 1.0; // Alpha channel

    // Apply color based on mouse segment distance
    if (mplsSize > 0) {
        double projection, vx, vy;
        double l = distanceToNearestMouseSegment(col, row, &projection, &vx, &vy);
        double taperFactor = 0.6;
        double projectedFraction = 1.0 - fminf(1.0, fmaxf(projection, 0.0)) * taperFactor;
        double R = 12;
        double m = exp(-l/R);
        double speed = sqrt(vx * vx + vy * vy);
        double x = fminf(1.0, fmaxf(fabs((speed * speed * 0.02 - projection * 5.0) * projectedFraction), 0.0));
        double r = (2.4 / 60.0) * x + (0.2 /30.0) * (1-x) + (1.0 * pow(x, 9.0));
        double g = (0.0 / 60.0) * x + (51.8 / 30.0) * (1-x) + (1.0 * pow(x, 9.0));
        double b = (5.9 / 60.0) * x + (100.0 / 30.0) * (1-x) + (1.0 * pow(x, 9.0));

        cuParams.color[index] += m * r;
        cuParams.color[index + 1] += m * g;
        cuParams.color[index + 2] += m * b;
    }

    // Copy color to image data
    for (int i = 0; i < 4; ++i) {
        cuParams.imageData[index + i] = cuParams.color[index + i];
    }
}



void
CudaRenderer::setup() {
   cells_per_side = image->width / CELL_DIM - 1;

   cudaMalloc(&cdVX, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMalloc(&cdVY, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMalloc(&cdPressures, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMalloc(&cdPressuresCopy, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMalloc(&cdVXCopy, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMalloc(&cdVYCopy, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMalloc(&cdDivergence, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMalloc(&cdVorticity, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMalloc(&cdColor, 4 * sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMalloc(&cdColorCopy, 4 * sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMalloc(&cdImageData, 4 * sizeof(float) * image->width * image->height);
   cudaMalloc(&cdMpls, 400 * sizeof(float) * image->width * image->height);

   cudaMemset(cdVX, 0, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMemset(cdVY, 0, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMemset(cdPressures, 0, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMemset(cdPressuresCopy, 0, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMemset(cdVXCopy, 0, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMemset(cdVYCopy, 0, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMemset(cdDivergence, 0, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMemset(cdVorticity, 0, sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMemset(cdColor, 0, 4 * sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));
   cudaMemset(cdColorCopy, 0, 4 * sizeof(float) * (cells_per_side + 1) * (cells_per_side + 1));

    GlobalConstants params;
    params.cells_per_side = cells_per_side;
    params.width = image->width;
    params.height = image->height;
    params.VX = cdVX;
    params.VY = cdVY;
    params.pressures = cdPressures;
    params.pressuresCopy = cdPressuresCopy;
    params.VXCopy = cdVXCopy;
    params.VYCopy = cdVYCopy;
    params.divergence = cdDivergence;
    params.vorticity = cdVorticity;
    params.color = cdColor;
    params.colorCopy = cdColorCopy;
    params.imageData = cdImageData;
    params.mpls = cdMpls;

    cudaMemcpyToSymbol(cuParams, &params, sizeof(GlobalConstants));
}

// Called after clear, before render
void CudaRenderer::setNewQuantities(std::vector<std::pair<int, int> > mpls) {

    mplsSize = mpls.size();
    if (mplsSize < 1) {
        // if mpls.size is 0, then call kernel that decreases VX,VY by 0.999
        dim3 blockDim(BLOCKSIDE,BLOCKSIDE,1);
        dim3 gridDim(
                (image->width + blockDim.x - 1) / blockDim.x,
                (image->height + blockDim.y - 1) / blockDim.y);
        kernelFadeVelocities<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();

    } else {
        int* mplsArray = new int[mplsSize * 2];
        int count = 0;
        for (std::vector<std::pair<int,int> >::iterator it = mpls.begin() 
                ; it != mpls.end(); ++it) {
            std::pair<int,int> c = *it;
            mplsArray[count] = c.first;
            mplsArray[count + 1] = c.second;
            count += 2;
        }
        cudaMemset(cdMpls, 0, 400 * sizeof(int));
        cudaMemcpy(cdMpls, mplsArray, (mplsSize * 2) * sizeof(int), 
                cudaMemcpyHostToDevice);

        dim3 blockDim(BLOCKSIDE,BLOCKSIDE,1);
        dim3 gridDim(
                (image->width + blockDim.x - 1) / blockDim.x,
                (image->height + blockDim.y - 1) / blockDim.y);
        kernelSetNewVelocities<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
    }
}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  
void
CudaRenderer::clearImage() {
    dim3 blockDim(BLOCKSIDE,BLOCKSIDE,1);
    dim3 gridDim(
            (image->width + blockDim.x - 1) / blockDim.x,
            (image->height + blockDim.y - 1) / blockDim.y);
    kernelClearImage<<<gridDim, blockDim>>>(1.f,1.f,1.f,1.f);
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {
    dim3 blockDim(BLOCKSIDE,BLOCKSIDE,1);
    dim3 gridDim(
            (image->width + blockDim.x - 1) / blockDim.x,
            (image->height + blockDim.y - 1) / blockDim.y);
    kernelCopyVelocities<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize(); 
    kernelAdvectVelocityForward<<<gridDim, blockDim>>>();
    //cudaDeviceSynchronize();
    kernelAdvectVelocityBackward<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    
    kernelApplyVorticity<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    kernelApplyVorticityForce<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();

    kernelApplyDivergence<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();

    kernelCopyPressures<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    kernelPressureSolve<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();

    kernelPressureGradient<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
   
  
    kernelDrawColor<<<gridDim, blockDim>>>(mplsSize);
 
    kernelCopyColor<<<gridDim,blockDim>>>();
    cudaDeviceSynchronize(); 
    kernelAdvectColorForward<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    kernelAdvectColorBackward<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}

