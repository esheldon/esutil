! vim: set filetype=fortran
module cosmolib
    ! class to calculate distances.
    !
    ! f2py doesn't yet support derived types (structures), forcing me to send
    ! all the cosmological parameters each to the functions rather than a
    ! structure.  This is pretty ugly and harder to maintain, but still nicer
    ! than writing it in C.
    ! 
    ! f2py will support structures after this summer....
    !
    ! Uses gauss-legendre integration extremely fast and accurate
    !
    ! For 1/E(z) integration, 5 points is good to 1.e-8
    ! vnpts=10 for volumes good to E(z) precision.

    implicit none


    ! class variables
    integer*8, save, private :: has_been_init = 0
    integer*8, parameter :: npts=5
    integer*8, parameter :: vnpts=10
    real*8, private, save, dimension(npts) :: xxi, wwi
    real*8, private, save, dimension(vnpts) :: vxxi, vwwi

    ! use in scinv for dlens in Mpc
    real*8, parameter :: four_pi_G_over_c_squared = 6.0150504541630152e-07_8
    real*8, parameter :: c = 2.99792458e5_8

    ! for integral calculations
    real*8, private :: f1,f2,z,ezinv

    ! for integration
    real*8, parameter, public :: M_PI    = 3.141592653589793238462643383279502884197_8


contains

    ! you must initialize
    subroutine cosmo_init()

        if (has_been_init == 0) then
            call set_cosmo_weights()
        endif
        has_been_init = 1

    end subroutine cosmo_init

    ! comoving distance
    ! variants for a vector of zmax and zmin,zmax both vectors
    real*8 function cdist(zmin, zmax, &
                          DH, flat, omega_m, omega_l, omega_k )
        ! comoving distance
        real*8, intent(in) :: zmin, zmax

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        cdist = DH*ez_inverse_integral(zmin, zmax, &
                                       flat, omega_m, omega_l, omega_k )
    end function cdist


    subroutine cdist_vec1(zmin, zmax, n, dc, &
                          DH, flat, omega_m, omega_l, omega_k )
        ! first arg, zmin, is the vector
        integer*8, intent(in) :: n
        real*8, intent(in), dimension(n) :: zmin
        real*8, intent(in) :: zmax
        real*8, intent(inout), dimension(n) :: dc

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        integer*8 i

        do i=1,n
            dc(i) = DH*ez_inverse_integral(zmin(i), zmax, &
                                           flat, omega_m, omega_l, omega_k )
        end do
    end subroutine cdist_vec1

    subroutine cdist_vec2(zmin, zmax, n, dc, &
                          DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in) :: zmin
        real*8, intent(in), dimension(n) :: zmax
        real*8, intent(inout), dimension(n) :: dc

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        integer*8 i

        do i=1,n
            dc(i) = DH*ez_inverse_integral(zmin, zmax(i), &
                                           flat, omega_m, omega_l, omega_k )
        end do
    end subroutine cdist_vec2


    subroutine cdist_2vec(zmin, zmax, n, dc, &
                          DH, flat, omega_m, omega_l, omega_k )
        ! both zmin and zmax are vectors
        integer*8, intent(in) :: n
        real*8, intent(in), dimension(n) :: zmin
        real*8, intent(in), dimension(n) :: zmax
        real*8, intent(inout), dimension(n) :: dc

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        integer*8 i

        do i=1,n
            dc(i) = DH*ez_inverse_integral(zmin(i), zmax(i), &
                                           flat, omega_m, omega_l, omega_k )
        end do
    end subroutine cdist_2vec



    real*8 function tcdist(zmin, zmax, &
                           DH, flat, omega_m, omega_l, omega_k )
        ! useful for calculating transverse comoving distance at zmax.  When
        ! zmin is not zero, useful in angular diameter distances

        real*8, intent(in) :: zmin, zmax

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        real*8 t

        tcdist = cdist(zmin, zmax, &
                       DH, flat, omega_m, omega_l, omega_k )

        if (flat) then
             !just use comoving distance already calculated
        else if (omega_k > 0) then
            t = sqrt(omega_k)/DH
            tcdist = sinh( tcdist*t)/t
        else
            t = sqrt(-omega_k)/DH
            tcdist =  sin( tcdist*t)/t
        endif

    end function tcdist

    subroutine tcdist_vec1(zmin, zmax, n, dm, &
                           DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in), dimension(n) :: zmin
        real*8, intent(in) :: zmax
        real*8, intent(inout), dimension(n) :: dm

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        real*8 d, t

        integer*8 i

        if (flat) then
            ! just use comoving distance
            do i=1,n
                dm(i) = DH*ez_inverse_integral(zmin(i), zmax, &
                                               flat, omega_m, omega_l, omega_k )
            enddo
        else if (omega_k > 0) then
            t = sqrt(omega_k)/DH
            do i=1,n
                d= DH*ez_inverse_integral(zmin(i), zmax, &
                                          flat, omega_m, omega_l, omega_k )
                dm(i) = sinh(d*t)/t
            enddo
        else
            t = sqrt(-omega_k)/DH
            do i=1,n
                d= DH*ez_inverse_integral(zmin(i), zmax, &
                                          flat, omega_m, omega_l, omega_k )
                dm(i) = sin(d*t)/t
            enddo

        endif

    end subroutine tcdist_vec1



    subroutine tcdist_vec2(zmin, zmax, n, dm, &
                           DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in) :: zmin
        real*8, intent(in), dimension(n) :: zmax
        real*8, intent(inout), dimension(n) :: dm

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        real*8 d, t

        integer*8 i

        if (flat) then
            ! just use comoving distance
            do i=1,n
                dm(i) = DH*ez_inverse_integral(zmin, zmax(i), &
                                               flat, omega_m, omega_l, omega_k )
            enddo
        else if (omega_k > 0) then
            t = sqrt(omega_k)/DH
            do i=1,n
                d= DH*ez_inverse_integral(zmin, zmax(i), &
                                          flat, omega_m, omega_l, omega_k )
                dm(i) = sinh(d*t)/t
            enddo
        else
            t = sqrt(-omega_k)/DH
            do i=1,n
                d= DH*ez_inverse_integral(zmin, zmax(i), &
                                          flat, omega_m, omega_l, omega_k )
                dm(i) = sin(d*t)/t
            enddo

        endif

    end subroutine tcdist_vec2

    subroutine tcdist_2vec(zmin, zmax, n, dm, &
                           DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in), dimension(n) :: zmin
        real*8, intent(in), dimension(n) :: zmax
        real*8, intent(inout), dimension(n) :: dm

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        real*8 d, t

        integer*8 i

        if (flat) then
            ! just use comoving distance
            do i=1,n
                dm(i) = DH*ez_inverse_integral(zmin(i), zmax(i), &
                                               flat, omega_m, omega_l, omega_k )
            enddo
        else if (omega_k > 0) then
            t = sqrt(omega_k)/DH
            do i=1,n
                d= DH*ez_inverse_integral(zmin(i), zmax(i), &
                                          flat, omega_m, omega_l, omega_k )
                dm(i) = sinh(d*t)/t
            enddo
        else
            t = sqrt(-omega_k)/DH
            do i=1,n
                d= DH*ez_inverse_integral(zmin(i), zmax(i), &
                                          flat, omega_m, omega_l, omega_k )
                dm(i) = sin(d*t)/t
            enddo
        endif

    end subroutine tcdist_2vec






    real*8 function angdist(zmin, zmax, &
                            DH, flat, omega_m, omega_l, omega_k )
        ! angular diameter distance
        real*8, intent(in) :: zmin, zmax

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        if (flat) then
            ! this is just the comoving distance over 1+zmax
            angdist = DH*ez_inverse_integral(zmin, zmax, &
                                             flat, omega_m, omega_l, omega_k )
        else
            angdist = tcdist(zmin, zmax, &
                             DH, flat, omega_m, omega_l, omega_k )
        endif
        angdist = angdist/(1.+zmax)
    end function angdist

    subroutine angdist_vec1(zmin, zmax, n, da, &
                           DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in), dimension(n) :: zmin
        real*8, intent(in) :: zmax
        real*8, intent(inout), dimension(n) :: da

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        integer*8 i

        do i=1,n
            da(i) = angdist(zmin(i), zmax, &
                            DH, flat, omega_m, omega_l, omega_k )
        enddo

    end subroutine angdist_vec1

    subroutine angdist_vec2(zmin, zmax, n, da, &
                           DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in) :: zmin
        real*8, intent(in), dimension(n) :: zmax
        real*8, intent(inout), dimension(n) :: da

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        integer*8 i

        do i=1,n
            da(i) = angdist(zmin, zmax(i), &
                            DH, flat, omega_m, omega_l, omega_k )
        enddo

    end subroutine angdist_vec2




    subroutine angdist_2vec(zmin, zmax, n, da, &
                           DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in), dimension(n) :: zmin
        real*8, intent(in), dimension(n) :: zmax
        real*8, intent(inout), dimension(n) :: da

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        integer*8 i

        do i=1,n
            da(i) = angdist(zmin(i), zmax(i), &
                            DH, flat, omega_m, omega_l, omega_k )
        enddo

    end subroutine angdist_2vec



    real*8 function lumdist(zmin, zmax, &
                            DH, flat, omega_m, omega_l, omega_k )
        ! angular diameter distance
        real*8, intent(in) :: zmin, zmax

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        lumdist = angdist(zmin, zmax, DH, flat, omega_m, omega_l, omega_k )
        lumdist = lumdist*(1.+zmax)**2

    end function lumdist

    subroutine lumdist_vec1(zmin, zmax, n, da, &
                           DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in), dimension(n) :: zmin
        real*8, intent(in) :: zmax
        real*8, intent(inout), dimension(n) :: da

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        real*8 d
        integer*8 i

        do i=1,n
            d = angdist(zmin(i), zmax, &
                        DH, flat, omega_m, omega_l, omega_k )
            da(i) = d*(1.+zmax)**2
        enddo

    end subroutine lumdist_vec1



    subroutine lumdist_vec2(zmin, zmax, n, da, &
                           DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in) :: zmin
        real*8, intent(in), dimension(n) :: zmax
        real*8, intent(inout), dimension(n) :: da

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        real*8 d
        integer*8 i

        do i=1,n
            d = angdist(zmin, zmax(i), &
                        DH, flat, omega_m, omega_l, omega_k )
            da(i) = d*(1.+zmax(i))**2
        enddo

    end subroutine lumdist_vec2

    subroutine lumdist_2vec(zmin, zmax, n, da, &
                           DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in), dimension(n) :: zmin
        real*8, intent(in), dimension(n) :: zmax
        real*8, intent(inout), dimension(n) :: da

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        real*8 d

        integer*8 i

        do i=1,n
            d = angdist(zmin(i), zmax(i), &
                        DH, flat, omega_m, omega_l, omega_k )
            da(i) = d*(1.+zmax(i))**2
        enddo

    end subroutine lumdist_2vec





    real*8 function dv(z, &
                       DH, flat, omega_m, omega_l, omega_k )
        ! comoving volume element at redshift z
        real*8, intent(in) :: z

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        real*8 ezinv, da

        da = angdist(0.0_8, z, DH, flat, omega_m, omega_l, omega_k)
        ezinv = ez_inverse(z, flat, omega_m, omega_l, omega_k )

        dv = DH*da**2*ezinv*(1.0+z)**2

    end function dv

    subroutine dv_vec(z, n, dvvec, &
                      DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in), dimension(n) :: z
        real*8, intent(inout), dimension(n) :: dvvec

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        integer*8 i

        do i=1,n
            dvvec(i) = dv( z(i), DH, flat, omega_m, omega_l, omega_k )
        enddo
    end subroutine dv_vec

    real*8 function volume(zmin, zmax, &
                           DH, flat, omega_m, omega_l, omega_k )
        real*8, intent(in) :: zmin, zmax
        real*8 f1, f2, z

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        real*8 tdv

        integer*8 i

        f1 = (zmax-zmin)/2.
        f2 = (zmax+zmin)/2.

        volume = 0
        do i=1,vnpts
            z = vxxi(i)*f1 + f2
            tdv = dv(z, DH, flat, omega_m, omega_l, omega_k)
            volume = volume + f1*tdv*vwwi(i)
        enddo
    end function volume






    real*8 function scinv(zl, zs, &
                          DH, flat, omega_m, omega_l, omega_k )
        ! inverse critical density
        real*8, intent(in) :: zl,zs

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        real*8 dl, ds, dls

        if (zs <= zl) then 
            scinv=0.0
            return
        end if


        if (flat) then
            ! we can save some computation in the flat case
            dl = cdist(0.0_8, zl, DH, flat, omega_m, omega_l, omega_k)
            ds = cdist(0.0_8, zs, DH, flat, omega_m, omega_l, omega_k)
            scinv = dl/(1.+zl)*(ds-dl)/ds * four_pi_G_over_c_squared
        else
            dl  = angdist(0.0_8, zl, DH, flat, omega_m, omega_l, omega_k)
            ds  = angdist(0.0_8, zs, DH, flat, omega_m, omega_l, omega_k)
            dls = angdist(zl, zs, DH, flat, omega_m, omega_l, omega_k)

            scinv = dls*dl/ds*four_pi_G_over_c_squared
        endif

    end function scinv

    subroutine scinv_vec1(zl, zs, n, sc_inv, &
                          DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in), dimension(n) :: zl
        real*8, intent(in) :: zs
        real*8, intent(inout), dimension(n) :: sc_inv

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        integer*8 i

        do i=1,n
            sc_inv(i) = scinv(zl(i), zs, &
                              DH, flat, omega_m, omega_l, omega_k )
        enddo

    end subroutine scinv_vec1

    subroutine scinv_vec2(zl, zs, n, sc_inv, &
                         DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in) :: zl
        real*8, intent(in), dimension(n) :: zs
        real*8, intent(inout), dimension(n) :: sc_inv

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        integer*8 i

        do i=1,n
            sc_inv(i) = scinv(zl, zs(i), &
                              DH, flat, omega_m, omega_l, omega_k )
        enddo

    end subroutine scinv_vec2

    subroutine scinv_2vec(zl, zs, n, sc_inv, &
                         DH, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, intent(in), dimension(n) :: zl
        real*8, intent(in), dimension(n) :: zs
        real*8, intent(inout), dimension(n) :: sc_inv

        logical, intent(in) :: flat
        real*8, intent(in) :: DH, omega_m,omega_l,omega_k

        integer*8 i

        do i=1,n
            sc_inv(i) = scinv(zl(i), zs(i), &
                              DH, flat, omega_m, omega_l, omega_k )
        enddo

    end subroutine scinv_2vec



    real*8 function ez_inverse(z, flat, omega_m, omega_l, omega_k )
        real*8, intent(in) :: z

        logical, intent(in) :: flat
        real*8, intent(in) :: omega_m,omega_l,omega_k

        if (flat) then
            ez_inverse = omega_m*(1.+z)**3 + omega_l
        else
            ez_inverse = omega_m*(1.+z)**3 + omega_k*(1.+z)**2 + omega_l
        endif
        ez_inverse = sqrt(1.0/ez_inverse)
    end function ez_inverse

    subroutine ez_inverse_vec(z, n, ez, flat, omega_m, omega_l, omega_k )
        integer*8, intent(in) :: n
        real*8, dimension(n), intent(in) :: z
        real*8, dimension(n), intent(inout) :: ez

        logical, intent(in) :: flat
        real*8, intent(in) :: omega_m,omega_l,omega_k

        integer*8 i

        do i=1,n
            ez(i) = ez_inverse( z(i), &
                               flat, omega_m, omega_l, omega_k)
        enddo
    end subroutine ez_inverse_vec


    real*8 function ez_inverse_integral(zmin, zmax, flat, omega_m, omega_l, omega_k) result(val)
        real*8, intent(in) :: zmin, zmax

        logical, intent(in) :: flat
        real*8, intent(in) :: omega_m,omega_l,omega_k

        integer*8 i


        f1 = (zmax-zmin)/2.
        f2 = (zmax+zmin)/2.

        val = 0.0

        do i=1,npts
            z = xxi(i)*f1 + f2
            ezinv = ez_inverse(z, &
                               flat, omega_m, omega_l, omega_k)

            val = val + f1*ezinv*wwi(i);
        end do

    end function ez_inverse_integral














    subroutine set_cosmo_weights()

        call gauleg(-1.0_8, 1.0_8, npts, xxi, wwi)
        call gauleg(-1.0_8, 1.0_8, vnpts, vxxi, vwwi)

    end subroutine set_cosmo_weights

    subroutine print_weights()
        integer*8 i

        do i=1,size(wwi)
            print '("xxi: ",F15.8,"  wwi: ",F15.8)',xxi(i),wwi(i)
        enddo
    end subroutine


    ! from numerical recipes
    subroutine gauleg(x1, x2, npts, x, w)

        integer*8, intent(in) :: npts
        real*8, intent(in) :: x1, x2
        real*8, intent(inout), dimension(npts) :: x, w
        

        integer*8 :: i, j, m
        real*8 :: xm, xl, z1, z, p1, p2, p3, pp, EPS, abszdiff


        pp = 0.0
        EPS = 4.e-11

        m = (npts + 1)/2

        xm = (x1 + x2)/2.0
        xl = (x2 - x1)/2.0
        z1 = 0.0

        do i=1,m

            z=cos( M_PI*(i-0.25)/(npts+.5) )

            abszdiff = abs(z-z1)

            do while (abszdiff > EPS) 

                p1 = 1.0
                p2 = 0.0
                do j=1,npts
                    p3 = p2
                    p2 = p1
                    p1 = ( (2.0*j - 1.0)*z*p2 - (j-1.0)*p3 )/j
                end do
                pp = npts*(z*p1 - p2)/(z*z -1.)
                z1=z
                z=z1 - p1/pp

                abszdiff = abs(z-z1)

            end do

            x(i) = xm - xl*z
            x(npts+1-i) = xm + xl*z
            w(i) = 2.0*xl/( (1.-z*z)*pp*pp )
            w(npts+1-i) = w(i)


        end do

    end subroutine gauleg


end module cosmolib
