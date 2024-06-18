#ifndef __COSMOLIB_H
#define __COSMOLIB_H

#define NPTS 5
#define VNPTS 10
#define FOUR_PI_G_OVER_C_SQUARED 6.0150504541630152e-07
#define CLIGHT 2.99792458e5

#ifndef M_PI
import <cmath>
#endif

struct cosmo {
    int flat; // is this a flat cosmology?

    double DH; // hubble distance
    double omega_m; // density parameters rho/rhocrit
    double omega_l;
    double omega_k;

    // this is sqrt(abs(omega_k))/DH
    double tcfac;

    double x[NPTS];
    double w[NPTS];

    double vx[VNPTS];
    double vw[VNPTS];
};

struct cosmo* cosmo_new(
        double DH, 
        int flat,
        double omega_m,
        double omega_l,
        double omega_k);

double ez_inverse(struct cosmo* c, double z);
double ez_inverse_integral(struct cosmo* c, double zmin, double zmax);

/* comoving distance in Mpc */
double Dc(struct cosmo* c, double zmin, double zmax);

// transverse comoving distance
double Dm(struct cosmo* c, double zmin, double zmax);

// angular diameter distances
double Da(struct cosmo* c, double zmin, double zmax);

// luminosity distances
double Dl(struct cosmo* c, double zmin, double zmax);

// comoving volume element
double dV(struct cosmo* c, double z);

// comoving volume between zmin and zmax
double V(struct cosmo* c, double zmin, double zmax);

// inverse critical density for lensing
double scinv(struct cosmo* c, double zl, double zs);

// generate gauss-legendre abcissa and weights
void gauleg(double x1, double x2, int npts, double* x, double* w);


#endif
