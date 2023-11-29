# FluidSim

Starting from a famous paper "Stable Fuilds" by Jos Stam, we used the theory given by the paper, we are able to generate a simple version of fluid simumation. 


![Screenshot 2023-11-29 at 14 09 57](https://github.com/Feng-Jiang28/FluidSim/assets/106386742/d5414692-a040-4982-a55d-c4777fc17dee)

Basics: 
Fluid simulation is based on Navier-Stokes equeations by treating the fluid as a grid of interacting cells or boxes.
Each cell has properties like velocity and density. For practicality, simulations use a limited number of cells and interactions due to computational limits.
<img width="384" alt="image" src="https://github.com/Feng-Jiang28/FluidSim/assets/106386742/43fe3ed4-6d7b-4792-9e80-c11bbf938158">

The focus is on incompressible fluids (like water) rather than compressible ones (like air), as they are easier to simulate with constant density and pressure. 
To visualize fluid movement, a dye is added, and its density variation helps in observing the motion.

Data Structures(C)


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

<img width="400" alt="image" src="https://github.com/Feng-Jiang28/FluidSim/assets/106386742/a6d2df98-ddaf-4c5d-8e70-75be99497eb2">

