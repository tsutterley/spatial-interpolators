# Calculate generalized Legendre functions of arbitrary degree v
# Based on recipe in "An Atlas of Functions" by Spanier and Oldham, 1987 (589)
# Pv is the Legendre function of the first kind
# Qv is the Legendre function of the second kind
from libc.math cimport sin, cos, pow, floor, sqrt, INFINITY, M_PI
cdef extern from "complex.h":
	double cabs(double complex)
	double creal(double complex)
	double complex cpow(double complex, double complex)
	double complex csqrt(double complex)
	double complex csin(double complex)
	double complex ccos(double complex)

def PvQv_C(double x, double complex v):
	cdef unsigned iter
	cdef double a, k
	cdef double complex P, Q, R, K, s, c, w, X, g, u, f, t
	iter = 0
	if (x == -1):
		P = -INFINITY
		Q = -INFINITY
	elif (x == +1):
		P = 1.0
		Q = INFINITY
	else:
		# set a and R to 1
		a = 1.0
		R = 1.0
		K = 4.0*csqrt(cabs(v - cpow(v,2)))
		if ((cabs(1.0 + v) + floor(1.0 + creal(v))) == 0):
			a = 1.0e99
			v = -1.0 - v
		# s and c = sin and cos of (pi*v/2.0)
		s = csin(0.5*M_PI*v)
		c = ccos(0.5*M_PI*v)
		w = cpow(0.5 + v, 2)
		# if v is less than or equal to six (repeat until greater than six)
		while (creal(v) <= 6.0):
			v += 2.0
			R = R*(v - 1.0)/v
		# calculate X and g and update R
		X = 1.0 / (4.0 + 4.0*v)
		g = 1.0 + 5*X*(1.0 - 3.0*X*(0.35 + 6.1*X))
		R = R*(1.0 - X*(1.0 - g*X/2))/csqrt(8.0*X)
		# set g and u to 2.0*x
		g = 2.0*x
		u = 2.0*x
		# set f and t to 1
		f = 1.0
		t = 1.0
		# set k to 1/2
		k = 0.5
		# calculate new X
		X = 1.0 + (1e8/(1.0 - pow(x,2)))
		# update t
		t = t*pow(x,2) * (pow(k,2) - w)/(pow(k + 1.0,2) - 0.25)
		# add 1 to k
		k += 1.0
		# add t to f
		f += t
		# update u
		u = u*pow(x,2) * (pow(k,2) - w)/(pow(k + 1.0,2) - 0.25)
		# add 1 to k
		k += 1.0
		# add u to g
		g += u
		# if k is less than K and |Xt| is greater than |f|
		# repeat previous set of operations until valid
		while ((k < creal(K)) | (cabs(X*t) > cabs(f))):
			iter += 1
			t = t*pow(x,2) * (pow(k,2) - w)/(pow(k + 1.0,2) - 0.25)
			k += 1.0
			f += t
			u = u*pow(x,2) * (pow(k,2) - w)/(pow(k + 1.0,2) - 0.25)
			k += 1.0
			g += u
		# update f and g
		f = f + (pow(x,2)*t/(1.0 - pow(x,2)))
		g = g + (pow(x,2)*u/(1.0 - pow(x,2)))
		# calculate generalized Legendre functions
		P = ((s*g*R) + (c*f/R))/sqrt(M_PI)
		Q = a*sqrt(M_PI)*((c*g*R) - (s*f/R))/2.0
	# return P, Q and number of iterations
	return (P, Q, iter)
