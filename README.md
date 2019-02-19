# Normal_Mode_Analysis
An normal mode analysis plugin based on basics objects of the OpenMM Simulation Module

This module uses classical way to analyze normal modes of a minimized protein structure. 
After minimization, the protein is assumed harmonically aprroximated. 
To obtain the hessian, structure corrdinates are tweak each steps and then the first and second derivatives of potential energy can be calculated. 
The algorithms are designed so that the sizes of tweaked positions would produce a small enough, and at a similar scale for different lengths of protein, forces such that the harmonic approximation of potential energy will be automatically guaranteed.

More details would be commited later...

## Fearures
1. Two step (CPU, followed by GPU accelerated) energy minimization of protein structure before calculating normal modes
2. Calculating demass-weighted hessian
3. Obtaining normal modes and their corresponding vibrational frequencies
4. Plotting vibrational spectrum to be compared with IR spectrum data
5. Calculation of classical or semi-classical (QM formula) vibrational entropy

## Inputs
### topology (simtk.openmm.openmm.Topology)
The OpenMM topology object
### system (simtk.openmm.openmm.System)
The OpenMM system object
### integrator (simtk.openmm.openmm.Integrator)
The OpenMM integrator object
### initPositions (numpy.array)
The N*3 array of positions (N is the number of atoms)
### CPUProp (dictionary=None)
The CPU platform-specific properties used for generating simulation object in OpenMM
### CUDAProp (dictionary=None)
The CUDA platform-specific properties used for generating simulation object in OpenMM

## Algorithms
To be added later...


