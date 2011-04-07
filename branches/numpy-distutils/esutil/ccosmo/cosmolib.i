// vim: set filetype=c
%module cosmolib
%{
#define SWIG_FILE_WITH_INIT
#include "cosmolib.h"
#include "gauleg.h"
%}

%include "numpy.i"

%init %{
import_array();
%}


#define NPTS 5
#define VNPTS 10
#define FOUR_PI_G_OVER_C_SQUARED 6.0150504541630152e-07
#define CLIGHT 2.99792458e5


void gauleg(double x1, double x2, int npts, double* x, double* w);

double ez_inverse(struct cosmo* c, double z);

%apply (int DIM1, double* IN_ARRAY1) {(int nz, double* z),
                                      (int nezi, double* ezi)}
void ez_inverse_vec(struct cosmo* c, int nz, double* z, int nezi, double* ezi);
double ez_inverse_integral(struct cosmo* c, double zmin, double zmax);

// these will apply to all our vectorized distances
%apply (int DIM1, double* IN_ARRAY1) {(int nzmin, double* zmin),
                                      (int nd, double* d)}
%apply (int DIM1, double* IN_ARRAY1) {(int nzmax, double* zmax),
                                      (int nd, double* d)}
%apply (int DIM1, double* IN_ARRAY1) {(int nzmin, double* zmin),
                                      (int nzmax, double* zmax),
                                      (int nd, double* d)}

double cdist(struct cosmo* c, double zmin, double zmax);
void cdist_vec1(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        double zmax, 
        int nd, double* d);

void cdist_vec2(
        struct cosmo* c, 
        double zmin, 
        int nzmax, double* zmax, 
        int nd, double* d);

void cdist_2vec(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        int nzmax, double* zmax, 
        int nd, double* d);


// transverse comoving distance
double tcdist(struct cosmo* c, double zmin, double zmax);

void tcdist_vec1(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        double zmax, 
        int nd, double* d);

void tcdist_vec2(
        struct cosmo* c, 
        double zmin, 
        int nzmax, double* zmax, 
        int nd, double* d);

void tcdist_2vec(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        int nzmax, double* zmax, 
        int nd, double* d);


// angular diameter distances
double angdist(struct cosmo* c, double zmin, double zmax);

void angdist_vec1(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        double zmax, 
        int nd, double* d);

void angdist_vec2(
        struct cosmo* c, 
        double zmin, 
        int nzmax, double* zmax, 
        int nd, double* d);

void angdist_2vec(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        int nzmax, double* zmax, 
        int nd, double* d);

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

%apply (int DIM1, double* IN_ARRAY1) {(int nz, double* z),
                                      (int ndv, double* dv)}
void dV_vec(
        struct cosmo* c, 
        int nz, double* z, 
        int ndv, double* dv);


// comoving volume between zmin and zmax
double volume(struct cosmo* c, double zmin, double zmax);


// inverse critical density for lensing
double scinv(struct cosmo* c, double zl, double zs);

%apply (int DIM1, double* IN_ARRAY1) {(int nzl, double* zl),
                                      (int nsci, double* sci)}
%apply (int DIM1, double* IN_ARRAY1) {(int nzs, double* zs),
                                      (int nsci, double* sci)}
%apply (int DIM1, double* IN_ARRAY1) {(int nzl, double* zl),
                                      (int nzs, double* zs),
                                      (int nsci, double* sci)}


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




// add a constructor and destructor
struct cosmo {

    %extend {
        cosmo(double DH, int flat, double omega_m, double omega_l, double omega_k) {
            struct cosmo* c;
            c = (struct cosmo*) malloc(sizeof(struct cosmo));

            c->flat    = flat;
            c->DH      = DH;
            c->omega_m = omega_m;
            c->omega_l = omega_l;
            c->omega_k = omega_k;

            c->tcfac = 0;
            if (c->flat != 1) {
                if (c->omega_k > 0) {
                    c->tcfac = sqrt(c->omega_k)/c->DH;
                } else {
                    c->tcfac = sqrt(-c->omega_k)/c->DH;
                }
            }

            gauleg(-1.0,1.0, NPTS,  c->x,  c->w);
            gauleg(-1.0,1.0, VNPTS, c->vx, c->vw);

            return c;
        }
        ~cosmo() {
            if ($self != NULL) {
                free($self);
            }
        }
    }
};


