# FluidSim

Phase 1: 
Starting from a famous paper "Stable Fuilds" by Jos Stam, we used the theory given by the paper, we are able to generate a simple version of fluid simumation. 

![Screenshot 2023-11-29 at 14 09 57](https://github.com/Feng-Jiang28/FluidSim/assets/106386742/d5414692-a040-4982-a55d-c4777fc17dee)

Basics: 
Fluid simulation is based on Navier-Stokes equeations by treating the fluid as a grid of interacting cells or boxes.
Each cell has properties like velocity and density. For practicality, simulations use a limited number of cells and interactions due to computational limits.
<img width="384" alt="image" src="https://github.com/Feng-Jiang28/FluidSim/assets/106386742/43fe3ed4-6d7b-4792-9e80-c11bbf938158">

The focus is on incompressible fluids (like water) rather than compressible ones (like air), as they are easier to simulate with constant density and pressure. 
To visualize fluid movement, a dye is added, and its density variation helps in observing the motion.

****<img width="722" alt="image" src="https://github.com/Feng-Jiang28/FluidSim/assets/106386742/eca3b3ee-98ac-4cf1-a236-eaf25d896b7a">



Data Structures(C)
```
struct FluidCube {
    int size;
    float dt;
    float diff;
    float visc;
    
    float *s;
    float *density;
    
    float *Vx;
    float *Vy;
    float *Vz;

    float *Vx0;
    float *Vy0;
    float *Vz0;
};
typedef struct FluidCube FluidCube;
```
Since this just initializes a motionless cube with no dye, we need a way both to add some dye:
```
void FluidCubeAddDensity(FluidCube *cube, int x, int y, int z, float amount)
{
    int N = cube->size;
    cube->density[IX(x, y, z)] += amount;
}
```
And to add some velocity:
```
void FluidCubeAddVelocity(FluidCube *cube, int x, int y, int z, float amountX, float amountY, float amountZ)
{
    int N = cube->size;
    int index = IX(x, y, z);
    
    cube->Vx[index] += amountX;
    cube->Vy[index] += amountY;
    cube->Vz[index] += amountZ;
}
```

Simulation Outline
There are three main operations that we'll use to perform a simulation step.

Diffuse: This operation simulates the spreading out of substances in the fluid, like dye in water. It applies to both the dye and the fluid's velocities, representing how they naturally disperse over time.

Project: This step ensures the simulation adheres to the principle of incompressibility. It adjusts the fluid in each cell so that the volume remains constant, correcting any imbalances caused by other operations.

Advect: This process simulates the movement of the fluid and the dye within it, driven by the velocities of each cell. It affects both the dye and the fluid's motion.

Additionally, there are two subroutines:

set_bnd (Set Bounds): This subroutine acts as a boundary condition, preventing the fluid from "leaking" out of the simulated area. It does this by mirroring the velocities at the edges, creating a wall-like effect.

lin_solve (Linear Solver): A function used in both the diffusion and projection steps. It solves linear equations related to these processes, although the exact workings might be complex and not fully understood by the implementer.

![ezgif com-video-to-gif](https://github.com/Feng-Jiang28/FluidSim/assets/106386742/72acbe95-6bef-45e9-b595-e3c38e9e9add)

The implementation of fluid simulation as described, while effective for demonstrating the principles of fluid dynamics in a 3D environment, does have some computational limitations.
1. High Computational Load:
Fluid simulation involves complex calculations for each cell in the fluid grid for every simulation step. This includes solving differential equations and updating the properties (like velocity and density) of each cell based on its interactions with neighboring cells.
2. Numerical Methods:
The implementation uses numerical methods (like the linear solver) that require iterative calculations. These methods can be computationally expensive, especially for high-resolution simulations.
3. Real-Time Performance:
Achieving real-time performance in fluid simulations is challenging, particularly for simulations with a large number of cells or high complexity.
4. Serial Computation:
The basic implementation as described is serial in nature, meaning each calculation is done one after the other. This approach doesn't efficiently utilize modern multi-core processors or parallel processing capabilities of GPUs.

Phase 2:

In phase 2, we are trying to implement a fluid simulation a CPU with OpenMP and GPU with CUDA, and compared the performance of the two implementations. 
OpenMP (Open Multi-Processing):

It allows the fluid simulation to utilize multiple cores of a CPU efficiently.
By dividing the simulation grid and processing multiple parts simultaneously, OpenMP can significantly reduce the time required for each simulation step.

CUDA (Compute Unified Device Architecture):

GPUs are highly efficient at handling parallel tasks due to their large number of cores designed for handling graphics operations.
Fluid simulations can benefit greatly from CUDA as it allows distributing the computation across thousands of small, efficient cores in the GPU, drastically speeding up the simulation process.

Plan: 

To optimize the fluid simulation using both OpenMP and CUDA, we can follow a structured approach that leverages the strengths of both technologies. Here's a plan outlining how to integrate OpenMP and CUDA into the existing simulation code:

1. Identify Parallelizable Components
Fluid Dynamics Operations: Most operations in fluid dynamics (like advection, diffusion, projection) are inherently parallelizable as they involve calculations on grid cells that can often be done independently.

Mouse Interaction Handling: The processing of mouse events, especially when rendering every pixel in response to mouse movements, is another area where parallel processing can be beneficial.
3. CUDA Implementation

Memory Management: Transfer all necessary arrays (velocity, density, etc.) to the GPU's device memory.
Kernel Design:
Grid Cell Operations: Implement CUDA kernels for each operation (advection, diffusion, projection). Use a grid of 16x16 thread blocks for efficient processing.
Mouse Interaction: Implement a kernel to handle mouse press events for each pixel. Given the relatively small number of mouse segments compared to the number of pixels, a kernel per pixel with a loop for each segment is more efficient.

Shared Memory Utilization: Use shared memory within thread blocks to speed up operations that involve neighboring cells, reducing global memory access.
Synchronization: Ensure proper synchronization between kernel executions to maintain data consistency, especially where operations have dependencies.

4. OpenMP Integration
CPU Parallelism: Use OpenMP for parts of the simulation that remain on the CPU. This can include initialization routines, setting up data structures, or handling tasks not offloaded to the GPU.
Loop Parallelization: Apply OpenMP pragmas to parallelize loops, especially in pre-processing or post-processing stages of the simulation.
Thread Management: Control the number of threads and manage workload distribution among them using OpenMP directives.

5. Hybrid Approach
Load Balancing: Determine the workload balance between the CPU and GPU. Offload heavy computational tasks to the GPU while utilizing the CPU for tasks that are less parallelizable or require frequent synchronization.

Data Transfer Optimization: Minimize data transfer between the CPU and GPU. Only transfer essential data to reduce overhead.

6. Optimization and Tuning
Performance Profiling: Use tools like NVIDIA Nsight and OpenMP profiling tools to identify bottlenecks.
Kernel Optimization: Optimize CUDA kernels by tuning block sizes, reducing warp divergence, and optimizing memory access patterns.
Dynamic Adjustment: Implement dynamic adjustment of workload distribution based on the current performance metrics.

7. Testing and Validation
Correctness: Ensure that the parallelized version of the simulation produces correct results. This involves comparing outputs with the serial version.
Performance Testing: Measure the performance improvements in terms of simulation speed and responsiveness, especially for real-time interaction.

8. Documentation and Maintenance
Code Documentation: Document the changes and architecture clearly, explaining how OpenMP and CUDA are integrated.
Maintainability: Ensure the code remains maintainable and modular to facilitate future updates or optimizations.

