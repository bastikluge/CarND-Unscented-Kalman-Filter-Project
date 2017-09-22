#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse;
  if ( estimations.size() == ground_truth.size() )
  {
    int size =  estimations.size();
    if ( size != 0 )
    {
      if ( estimations[0].size() == ground_truth[0].size() )
      {
        int dim = ground_truth[0].size();
        rmse = VectorXd(dim);

        // sum the error squares
        for ( int d=0; d<dim; d++ )
        {
          rmse(d) = 0;
        }
        for ( int s=0; s<size; s++ )
        {
          if ( (estimations[s].size() >= dim) && (ground_truth[s].size() >= dim) )
          {
		        VectorXd residual = estimations[s] - ground_truth[s];
		        residual = residual.array()*residual.array();
		        rmse += residual;
          }
          else
          {
            std::cout << "ERROR: Tools::CalculateRMSE called with estimation and/or ground_truth of unexpected dimension!\n";
          }
        }
        
        //calculate the mean
	      rmse = rmse/size;

	      //calculate the squared root
	      rmse = rmse.array().sqrt();
      }
      else
      {
        std::cout << "ERROR: Tools::CalculateRMSE called with estimations and ground_truth of different dimension!\n";
      }
    }
    else
    {
      std::cout << "ERROR: Tools::CalculateRMSE called with estimations and ground_truth of size 0!\n";
    }
  }
  else
  {
    std::cout << "ERROR: Tools::CalculateRMSE called with estimations and ground_truth of different size!\n";
  }
  return rmse;
}