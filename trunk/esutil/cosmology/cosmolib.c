#include <math.h>
#include <stdlib.h>
#include "cosmolib.h"


struct cosmo* cosmo_new(
        double DH, 
        int flat,
        double omega_m,
        double omega_l,
        double omega_k) {

    struct cosmo* c;
    c=(struct cosmo* ) calloc(1,sizeof(struct cosmo));

    if (c == NULL) {
        return NULL;
    }

    c->DH=DH;
    c->flat=flat;
    c->omega_m=omega_m;
    c->omega_l=omega_l;
    c->omega_k=omega_k;

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


/* comoving distance in Mpc */
double Dc(struct cosmo* c, double zmin, double zmax) {
    return c->DH*ez_inverse_integral(c, zmin, zmax);
}


// transverse comoving distance
double Dm(struct cosmo* c, double zmin, double zmax) {

    double d;

    d = Dc(c, zmin, zmax);

    if (!c->flat) {
        if (c->omega_k > 0) {
            d= sinh(d*c->tcfac)/c->tcfac;
        } else {
            d= sin(d*c->tcfac)/c->tcfac;
        }
    }
    return d;
}



// angular diameter distances
double Da(struct cosmo* c, double zmin, double zmax) {
    double d;
    d = Dm(c, zmin, zmax);
    d /= (1.+zmax);
    return d;
}




// luminosity distances
double Dl(struct cosmo* c, double zmin, double zmax) {
    double d;
    d = Dm(c, zmin, zmax);
    d *= (1.+zmax);
    return d;
}

// comoving volume element
double dV(struct cosmo* c, double z) {
    double da, ezinv, oneplusz;
    double dv;

    oneplusz = 1.+z;

    da = Da(c, 0.0, z);
    ezinv = ez_inverse(c, z);
    dv = c->DH*da*da*ezinv*oneplusz*oneplusz;

    return dv;
}

// comoving volume between zmin and zmax
double V(struct cosmo* c, double zmin, double zmax) {
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

    dl = Da(c, 0.0, zl);
    ds = Da(c, 0.0, zs);
    dls = Da(c, zl, zs);
    return dls*dl/ds*FOUR_PI_G_OVER_C_SQUARED;
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

void gauleg(double x1, double x2, int npts, double* x, double* w) {
	int i, j, m;
	double xm, xl, z1, z, p1, p2, p3, pp=0, EPS, abszdiff;

	EPS = 4.e-11;

	m = (npts + 1)/2;

	xm = (x1 + x2)/2.0;
	xl = (x2 - x1)/2.0;
	z1 = 0.0;

	for (i=1; i<= m; ++i) 
	{

		z=cos( M_PI*(i-0.25)/(npts+.5) );

		abszdiff = fabs(z-z1);

		while (abszdiff > EPS) 
		{
			p1 = 1.0;
			p2 = 0.0;
			for (j=1; j <= npts;++j)
			{
				p3 = p2;
				p2 = p1;
				p1 = ( (2.0*j - 1.0)*z*p2 - (j-1.0)*p3 )/j;
			}
			pp = npts*(z*p1 - p2)/(z*z -1.);
			z1=z;
			z=z1 - p1/pp;

			abszdiff = fabs(z-z1);

		}

		x[i-1] = xm - xl*z;
		x[npts+1-i-1] = xm + xl*z;
		w[i-1] = 2.0*xl/( (1.-z*z)*pp*pp );
		w[npts+1-i-1] = w[i-1];


	}

}
