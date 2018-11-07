#!/usr/bin/env python

"""numerical_integration.py"""

__author__  = "LJ Brown"

import math

# pip install autograd
# https://github.com/HIPS/autograd
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad

from scipy.integrate import quad


def integrate(f, a, b, tol):

	# F(a + h) = F(a) + hf(a) + h^2f'(a)/2 + ...
	# F(a + h) - F(a) = hf(a) + h^2f'(a)/2 + ...

	# Note: the derivative is lagging the power and the factorial by 1 -- h^i, i!, f^(i-1)(a)
	# Note: make sure f(a) != 0 and b-a < 1 if this is the only piece

	# number of terms in the orginal expansion -- error is O(h^(n+1))
	n = 100
	h = b-a

	cur_func = f
	I = 0
	for i in range( 1, n ):

		coeff = np.divide( np.power(h, i), math.factorial(i) )

		I += np.multiply( coeff, f(a) )

		# take the derivative
		cur_func = grad(cur_func)


	return I


# break appart integral
def integrate_pieces(f, a, b, tol):

	# number of peices to break [a,b] into
	n = 1000

	# interval start and end nodes [a, a+h, a+2h, ..., b]
	ts = np.linspace(a, b, num=n, endpoint=True)
	# h = (b-a)/n

	I = 0
	for i in range(len(ts)-1):
		ti = ts[i]
		tip1 = ts[i+1]

		# [TODO]: don't recompute the derivatives each time calculate this up front and do a dot product
		I += integrate(f, ti, tip1, tol)

	return I


#
# Testing
#


# test function
def tanh(x):
	y = np.exp(-2.0 * x)
	return (1.0 - y) / (1.0 + y)


if __name__ == "__main__":

	# test function
	test_func = lambda t: tanh(t)

	# test integrate function
	a = 1.0
	b = 5.0
	tol = None # [TODO]

	I = integrate_pieces(test_func, a, b, tol)


	# check against scipy
	check_I = quad(test_func, a, b)

	print("I = %s" % I)
	print("check I = %s +- %s" % check_I)




