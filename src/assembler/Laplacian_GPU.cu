#include <polyfem/Laplacian.hpp>
#include <iostream>

namespace polyfem
{

	__global__ double Laplacian::assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
	{
		const Eigen::MatrixXd &gradj = vals.basis_values[j].grad_t_m;
		// return ((gradi.array() * gradj.array()).rowwise().sum().array() * da.array()).colwise().sum();
		double res = 0;
		for (int k = 0; k < gradi.rows(); ++k) {
			res += gradi.row(k).dot(gradj.row(k)) * da(k);
		}
		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}


}
