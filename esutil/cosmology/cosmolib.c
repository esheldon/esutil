#include "cosmolib.h"
#include <math.h>

/* comoving distance in Mpc */
double cdist(struct cosmo* c, double zmin, double zmax) {
    return c->DH*ez_inverse_integral(c, zmin, zmax);
}

void cdist_vec1(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        double zmax, 
        int nd, double* d) {
    int i;
    for (i=0; i<nzmin; i++) {
        d[i] = c->DH*ez_inverse_integral(c, zmin[i], zmax);
    }
}
void cdist_vec2(
        struct cosmo* c, 
        double zmin, 
        int nzmax, double* zmax, 
        int nd, double* d) {
    int i;
    for (i=0; i<nzmax; i++) {
        d[i] = c->DH*ez_inverse_integral(c, zmin, zmax[i]);
    }
}
void cdist_2vec(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        int nzmax, double* zmax, 
        int nd, double* d) {
    int i;
    for (i=0; i<nzmax; i++) {
        d[i] = c->DH*ez_inverse_integral(c, zmin[i], zmax[i]);
    }
}

// transverse comoving distance
double tcdist(struct cosmo* c, double zmin, double zmax) {

    double d;

    d = cdist(c, zmin, zmax);

    if (!c->flat) {
        if (c->omega_k > 0) {
            d= sinh(d*c->tcfac)/c->tcfac;
        } else {
            d= sin(d*c->tcfac)/c->tcfac;
        }
    }
    return d;
}

void tcdist_vec1(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        double zmax, 
        int nd, double* d) {
    int i;
    for (i=0; i<nzmin; i++) {
        d[i] = tcdist(c, zmin[i], zmax);
    }
}
void tcdist_vec2(
        struct cosmo* c, 
        double zmin, 
        int nzmax, double* zmax, 
        int nd, double* d) {
    int i;
    for (i=0; i<nzmax; i++) {
        d[i] = tcdist(c, zmin, zmax[i]);
    }
}
void tcdist_2vec(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        int nzmax, double* zmax, 
        int nd, double* d) {
    int i;
    for (i=0; i<nzmax; i++) {
        d[i] = tcdist(c, zmin[i], zmax[i]);
    }
}


// angular diameter distances
double angdist(struct cosmo* c, double zmin, double zmax) {
    double d;
    d = tcdist(c, zmin, zmax);
    d /= (1.+zmax);
    return d;
}


void angdist_vec1(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        double zmax, 
        int nd, double* d) {
    int i;
    for (i=0; i<nzmin; i++) {
        d[i] = angdist(c, zmin[i], zmax);
    }
}
void angdist_vec2(
        struct cosmo* c, 
        double zmin, 
        int nzmax, double* zmax, 
        int nd, double* d) {
    int i;
    for (i=0; i<nzmax; i++) {
        d[i] = angdist(c, zmin, zmax[i]);
    }
}
void angdist_2vec(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        int nzmax, double* zmax, 
        int nd, double* d) {
    int i;
    for (i=0; i<nzmax; i++) {
        d[i] = angdist(c, zmin[i], zmax[i]);
    }
}


// luminosity distances
double lumdist(struct cosmo* c, double zmin, double zmax) {
    double d;
    d = tcdist(c, zmin, zmax);
    d *= (1.+zmax);
    return d;
}

void lumdist_vec1(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        double zmax, 
        int nd, double* d) {
    int i;
    for (i=0; i<nzmin; i++) {
        d[i] = lumdist(c, zmin[i], zmax);
    }
}
void lumdist_vec2(
        struct cosmo* c, 
        double zmin, 
        int nzmax, double* zmax, 
        int nd, double* d) {
    int i;
    for (i=0; i<nzmax; i++) {
        d[i] = lumdist(c, zmin, zmax[i]);
    }
}
void lumdist_2vec(
        struct cosmo* c, 
        int nzmin, double* zmin, 
        int nzmax, double* zmax, 
        int nd, double* d) {
    int i;
    for (i=0; i<nzmax; i++) {
        d[i] = lumdist(c, zmin[i], zmax[i]);
    }
}


// comoving volume element
double dV(struct cosmo* c, double z) {
    double da, ezinv, oneplusz;
    double dv;

    oneplusz = 1.+z;

    da = angdist(c, 0.0, z);
    ezinv = ez_inverse(c, z);
    dv = c->DH*da*da*ezinv*oneplusz*oneplusz;

    return dv;
}
void dV_vec(
        struct cosmo* c, 
        int nz, double* z, 
        int ndv, double* dv) {
    int i;
    for (i=0; i<nz; i++) {
        dv[i] = dV(c, z[i]);
    }
}

// comoving volume between zmin and zmax
double volume(struct cosmo* c, double zmin, double zmax) {
    int i;
    double f1,f2,z;
    double dv;
    double v=0;

    f1 = (zmax-zmin)/2.;
    f2 = (zmax+zmin)/2.;

    for (i=0; i<VNPTS; i++) {
        z = c->vx[i]*f1 + f2;
        dv = dV(c, z);
        v += f1*dv*c->vw[i];
    }

    return v;

}


// inverse critical density for lensing
double scinv(struct cosmo* c, double zl, double zs) {
    double dl, ds, dls;

    if (zs <= zl) {
        return 0.0;
    }

    dl = angdist(c, 0.0, zl);
    ds = angdist(c, 0.0, zs);
    dls = angdist(c, zl, zs);
    return dls*dl/ds*FOUR_PI_G_OVER_C_SQUARED;
}

void scinv_vec1(
        struct cosmo* c, 
        int nzl, double* zl, 
        double zs, 
        int nsci, double* sci) {
    int i;
    for (i=0; i<nzl; i++) {
        sci[i] = scinv(c, zl[i], zs);
    }
}
void scinv_vec2(
        struct cosmo* c, 
        double zl, 
        int nzs, double* zs, 
        int nsci, double* sci) {
    int i;
    for (i=0; i<nzs; i++) {
        sci[i] = scinv(c, zl, zs[i]);
    }
}
void scinv_2vec(
        struct cosmo* c, 
        int nzl, double* zl, 
        int nzs, double* zs, 
        int nsci, double* sci) {
    int i;
    for (i=0; i<nzs; i++) {
        sci[i] = scinv(c, zl[i], zs[i]);
    }
}





double ez_inverse(struct cosmo* c, double z) {
    double oneplusz, oneplusz2;
    double ezi;

    oneplusz = 1.+z;
    if (c->flat) {
        ezi = c->omega_m*oneplusz*oneplusz*oneplusz + c->omega_l;
    } else {
        oneplusz2 = oneplusz*oneplusz;
        ezi = c->omega_m*oneplusz2*oneplusz + c->omega_k*oneplusz2 + c->omega_l;
    }
    ezi = sqrt(1.0/ezi);
    return ezi;
}

void ez_inverse_vec(struct cosmo* c, int nz, double* z, int nezi, double* ezi) {
    int i;
    for (i=0; i<nz; i++) {
        ezi[i] = ez_inverse(c, z[i]);
    }
}

double ez_inverse_integral(struct cosmo* c, double zmin, double zmax) {
    int i;
    double f1, f2, z, ezinv_int=0, ezinv;

    f1 = (zmax-zmin)/2.;
    f2 = (zmax+zmin)/2.;

    ezinv_int = 0.0;

    for (i=0;i<NPTS;i++) {
        z = c->x[i]*f1 + f2;

        ezinv = ez_inverse(c, z);
        ezinv_int += f1*ezinv*c->w[i];
    }

    return ezinv_int;

}

