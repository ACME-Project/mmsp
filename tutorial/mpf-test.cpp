#include "MMSP.hpp"
#include <cmath>
#include <string>
#include <random>

using namespace MMSP;

int main(int argc, char* argv[])
{
  Init(argc, argv);

  // Declare and initialize variables.
  double lengthx = 2.0;
  double lengthy = 2.0;
  int nx = 200;
  double dx = lengthx/nx;
  int ny = 200;
  double dy = lengthy/ny;

  double m = 1.0;

  int grains = 3;

  // These need to be normalized so that the unit grid is dealt with
  // correctly.
  double width = 1.0;
  double gamma = 0.1;
  double w = 3.*(gamma/width);
  double epsilon2 = sqrt(3.*(width * gamma));

  // This is a guess for now.  Can be computed from m*epsilon2/dx/dx.
  double dt = 0.1;

  // In a future version I will fix this to read the starting file
  // number and set current_step and steps_to_take.
  int output_files_to_make = 40;
  int plot_freq = 200;
  int steps = output_files_to_make*plot_freq;

  // Here we set up two strings to be used in the construction of the
  // file name.  They are used in the main iteration loop.
  std::string base;
  base = "phi_fields";
	
  std::string suffix;
  suffix = "dat";
	
  // These are the grid declarations for the 2D PF calculation.
  grid<2,vector<double> > phi_fields_old(grains,0,nx,0,ny);
  grid<2,vector<double> > phi_fields_new(grains,0,nx,0,ny);
  grid<2,vector<double> > delFdelPhi(grains,0,nx,0,ny);
  grid<2,scalar<double> > delFdelPhiSum(1,0,nx,0,ny);
  grid<2,scalar<double> > source(1,0,nx,0,ny);
  grid<2,scalar<double> > phiSquaredSum(1,0,nx,0,ny);
	
  // Here we set the initial conditions for the simulation.
  //
  // For the random number generator to work we need -std=gnu++11 to
  // be supplied during compilation.
  //
  // Start by setting up the random number generator.  The mean of the
  // distribution should be centered at 1/N where N is the number of
  // grains.  We choose an arbitrary noise level.
  double noise_level = 0.05;
  std::default_random_engine generator (1);
  std::normal_distribution<double> distribution (1.0/3.0,noise_level);

  // Here we loop through each field and node and set a random value
  // for phi.
	
  for (int grain=0; grain<grains; grain++) {
    for (int i=0; i<nodes(phi_fields_old); i++) {
      double number = distribution(generator);
      phi_fields_old(i)[grain] = number;
    }
  }

  // This ensures that all the ghost cells are updated prior to the
  // first calculation.
  ghostswap(phi_fields_old);
	
  // Main iteration loop.
  for (int step=0; step<steps; step++) {
		
    int rank=0;
#ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
#endif

    // Feedback to the user.
    if (rank==0)
      print_progress(step, steps);
				
    if (step%plot_freq == 0) {
      std::string sstep = std::to_string(step);
      std::string filename = base + "." + sstep + "." + suffix;
      char* fname = &filename[0];
      output(phi_fields_old,fname);
    }
		
    // Compute all the phi squared values.
    for (int i=0; i<nodes(phi_fields_old); i++) {
      phiSquaredSum(i)= 0.0;
      for (int grain=0; grain<grains; grain++) {
	phiSquaredSum(i) += phi_fields_old(i)[grain]*phi_fields_old(i)[grain];
      }
    }
	  
    // Compute all the delFdelPhi terms for each field.
    //
    // I think these two loops can be switched.
    //
    for (int grain=0; grain<grains; grain++) {
      for (int i=0; i<nodes(phi_fields_old); i++) {
	double source = w*phi_fields_old(i)[grain]*(phiSquaredSum(i) - phi_fields_old(i)[grain]);
	vector<double> lap = laplacian(phi_fields_old,i);
	delFdelPhi(i)[grain] = source - epsilon2*lap[grain];
      }
    }

    // Compute the sum of delFdelPhi
    for (int i=0; i<nodes(phi_fields_old); i++) {
      delFdelPhiSum(i) = 0.0;
      for (int grain=0; grain<grains; grain++) {
	delFdelPhiSum(i) += delFdelPhi(i)[grain];
      }
    }

    // Compute the new values of phi.
    for (int grain=0; grain<grains; grain++) {
      for (int i=0; i<nodes(phi_fields_old); i++) {
	phi_fields_new(i)[grain] = -m*dt*(delFdelPhi(i)[grain] + (1.0/grains)*(delFdelPhiSum(i)))
	  + phi_fields_old(i)[grain];
      }
    }
	  
    // Swap and repeat.
    swap(phi_fields_old, phi_fields_new);

    // Update ghost cells.
    ghostswap(phi_fields_old);
	  
  }

  Finalize();
  return 0;	
}
