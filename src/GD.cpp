#include <iostream>
#include <Eigen/Dense>
#include <typeinfo>
#include <unistd.h>

using namespace Eigen;
//Given a function and its graident, perform gradient descent

/*Test function 1 for Task 1 => Used to demonstate convergence to a global min
double func(const VectorXd& x){
    //Create matrix A and vector b for testing
    Matrix3d A;
    A << 5.0, 2.0, 1.0,
         2.0, 4.0, 3.0,
         1.0, 3.0, 6.0; 
    
    Vector3d b(1.0, 2.0, 3.0);

    return ((0.5 * (x.transpose() * (A * x))) - (x.transpose() * b))(0,0); 
}

//Function's gradient
VectorXd grad(const VectorXd& x){
    //Create matrix A and vector b for testing
    Matrix3d A;
    A << 5.0, 2.0, 1.0,
         2.0, 4.0, 3.0,
         1.0, 3.0, 6.0; 
    
    Vector3d b(1.0, 2.0, 3.0);

    return (A * x) - b;
}
*/

//Task 2 => this function has two local minimums that seem to be semetric around the origin
// x*^T ~ +or-(6.5e-5, 0.000105)  aka your minimum depends on your initial guess
/*
double func(const VectorXd& x){
    Matrix2d A;
    A << 2.0, -1.0,
        -1.0, 1;
    return (x.transpose() * A * x)(0,0);
}

VectorXd grad(const VectorXd& x){
    Matrix2d A;
    A << 2.0, -1.0,
        -1.0, 1;
    return ((A + A.transpose()) * x);
}
*/

double func(const VectorXd& x, const MatrixXd& A,const VectorXd& b) {
	return ((A*x) + b).squaredNorm();
}


VectorXd grad(const VectorXd& x, const MatrixXd& A,const VectorXd& b) {
	return (A.transpose() * (2*((A*x)+b)));
}


double armijo(const VectorXd&x, const VectorXd& direction, double a, double c, const MatrixXd& A,const VectorXd& b) {
	while(func(x + a * direction, A, b) > func(x, A, b) + c * a * grad(x, A, b).dot(direction)){
		//std::cout << a << std::endl;
		a *= 0.5;
	}
	return a;
}

int main() {
	//Gradient descent
	//stepSize for part 1&2 should be 1.0, don't do Armijo for gradient decent in general
	double stepSize = 0.003;
	const double epsilon = 0.0001;

	Matrix2d R12, R23, R34, R41;
	R12 << 0.17, -0.98,
		0.98, 0.17;
	R23 << -0.17, -0.98,
		0.98, -0.17;
	R34 << 0.71, -0.71,
		0.71, 0.71;
	R41 << -0.71, -0.71,
		0.71, -0.71;

	Matrix<double, 16, 12> A;
	A.setZero();

	A.block(8, 0, 2, 2) = -R23.transpose();
	A.block(10, 2, 2, 2) = -R23.transpose();
	A.block(12, 4, 2, 2) = -R34.transpose();
	A.block(14, 6, 2, 2) = -R34.transpose();
	A.block(0, 8, 2, 2) = -R41.transpose();
	A.block(2, 10, 2, 2) = -R41.transpose();

	for(int i = 0; i < 12; i++){
		A(4+i,i) = 1;
	}


	Matrix<double, 16, 4> B;
	B.setZero();
	B.block(4,0,2,2) = -R12.transpose();
	B.block(6,2,2,2) = -R12.transpose();

	for(int i = 0; i < 4; i++){
		B(i, i) = 1;
	}
	Vector<double, 4> Rw1(1.0, 0, 0, 1.0);
	VectorXd b = B * Rw1;

	//Initial Guess
	VectorXd x0 = 100.0 * (VectorXd::Random(12));
	VectorXd gradient = grad(x0,A,b);
	double norm2 = gradient.norm();

	VectorXd xk;
	xk = x0 - (stepSize * gradient);

	//Check for convergence
	while(norm2 > epsilon){
		//std::cout << "x0: " << x0.transpose() << " xk: " << xk.transpose() << " norm2: " << norm2 << " a: " << stepSize <<std::endl;

		while(func(xk,A,b) >= func(x0,A,b)){
			stepSize = armijo(x0, -gradient, stepSize, 0.5,A,b);
			xk = x0 - (stepSize * gradient);
		}
		x0 = xk;
		gradient = grad(x0,A,b);
		norm2 = gradient.norm();
		xk = x0 - (stepSize * gradient);
	}
	//std::cout << "x* is :" << std::endl << x0 << std::endl;

	Matrix2d Rw2, Rw3, Rw4;

	Rw2 << x0.row(0), x0.row(1), x0.row(2), x0.row(3);
	Rw3 << x0.row(4), x0.row(5), x0.row(6), x0.row(7);
	Rw4 << x0.row(8), x0.row(9), x0.row(10), x0.row(11);

	std::cout << "Rw2: " << std::endl << Rw2 << std::endl << "Rw3:" << std::endl << Rw3 << std::endl << "Rw4:" << std::endl << Rw4 << std::endl;
	/*Parts 1 & 2
	//Create initial guess; you must change dimentions according to your functions
	VectorXd x0 = 100.0 * (VectorXd::Random(2));
	x0 = 100.0 * (VectorXd::Random(2));
	x0 = 100.0 * (VectorXd::Random(2));
	//x0 = 100.0 * (VectorXd::Random(2));
	//x0 = 100000.0 * (VectorXd::Random(2));
	std::cout << x0 << std::endl;
	//Initial norm2 value
	VectorXd gradient = grad(x0);
	double norm2 = gradient.norm();

	VectorXd xk;
	xk = x0 - (stepSize * gradient);
	//Check for convergence
	while(norm2 > epsilon){
		//std::cout << "x0: " << x0.transpose() << " xk: " << xk.transpose() << " norm2: " << norm2 << " a: " << stepSize <<std::endl;

		while(func(xk) >= func(x0)){
			stepSize = armijo(x0, -gradient, stepSize, 0.5);
			xk = x0 - (stepSize * gradient);
		}
		x0 = xk;
		gradient = grad(x0);
		norm2 = gradient.norm();
		xk = x0 - (stepSize * gradient);
	}
	std::cout << "x* is :" << std::endl << x0 << std::endl;
	*/


	return 0;
}
