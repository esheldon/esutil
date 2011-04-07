#ifndef __COSMOLIB_H
#define __COSMOLIB_H

#define NPTS 5
#define VNPTS 10
#define FOUR_PI_G_OVER_C_SQUARED 6.0150504541630152e-07
#define CLIGHT 2.99792458e5

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


double ez_inverse(struct cosmo* c, double z);
void ez_inverse_vec(struct cosmo* c, int nz, double* z, int nezi, double* ezi);
double ez_inverse_integral(struct cosmo* c, double zmin, double zmax);

/* comoving distance in Mpc */
double cdist(struct cosmo* c, double zmin, double zmax);
void cdist_vec1(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        double zmax, 
        int ndc, double* dc);
void cdist_vec2(
        struct cosmo* c, 
        double zmin, 
        int nzmax, double* zmax, 
        int ndc, double* dc);
void cdist_2vec(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        int nzmax, double* zmax, 
        int ndc, double* dc);


// transverse comoving distance
double tcdist(struct cosmo* c, double zmin, double zmax);

void tcdist_vec1(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        double zmax, 
        int ndm, double* dm);
void tcdist_vec2(
        struct cosmo* c, 
        double zmin, 
        int nzmax, double* zmax, 
        int ndm, double* dm);
void tcdist_2vec(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        int nzmax, double* zmax, 
        int ndm, double* dm);


// angular diameter distances
double angdist(struct cosmo* c, double zmin, double zmax);

void angdist_vec1(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        double zmax, 
        int ndm, double* dm);
void angdist_vec2(
        struct cosmo* c, 
        double zmin, 
        int nzmax, double* zmax, 
        int ndm, double* dm);
void angdist_2vec(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        int nzmax, double* zmax, 
        int ndm, double* dm);

// luminosity distances
double lumdist(struct cosmo* c, double zmin, double zmax);

void lumdist_vec1(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        double zmax, 
        int nd, double* d);

void lumdist_vec2(
        struct cosmo* c, 
        double zmin, 
        int nzmax, double* zmax, 
        int nd, double* d);

void lumdist_2vec(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        int nzmax, double* zmax, 
        int nd, double* d);


// comoving volume element
double dV(struct cosmo* c, double z);
void dV_vec(
        struct cosmo* c, 
        int nz, double* z, 
        int ndv, double* dv);


// comoving volume between zmin and zmax
double volume(struct cosmo* c, double zmin, double zmax);

// inverse critical density for lensing
double scinv(struct cosmo* c, double zl, double zs);


void scinv_vec1(
        struct cosmo* c, 
        int nzl, double* zl, 
        double zs, 
        int nsci, double* sci);
void scinv_vec2(
        struct cosmo* c, 
        double zl, 
        int nzs, double* zs, 
        int nsci, double* sci);
void scinv_2vec(
        struct cosmo* c, 
        int nzl, double* zl, 
        int nzs, double* zs, 
        int nsci, double* sci);





#endif
