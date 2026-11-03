! Conservative vertical remapping of tracer columns between layer distributions that
! differ (nesting: levels with different vertical grids). The compiled counterpart of
! pygetm.nesting.vremap.prepare + apply: for every column, the content of each target
! layer is the integral of the source reconstruction between its interfaces, found in one
! pass down the column (a merge of the two interface lists, O(ns + nd)).
!
! Reconstruction inside each source layer: constant (PCM, scheme 0) or a limited parabola
! (PPM, scheme 1; Colella and Woodward 1984, eqs. 1.6-1.10 for layers of unequal thickness;
! the outer two layers and layers next to empty ones constant), exactly as vremap.ppm_edges.
! Both keep the column content exactly and add no new extrema.
!
! Arrays as numpy passes them (C order): zs(n, ns+1), zd(n, nd+1), h(n, ns), c(nt, n, ns),
! out(nt, n, nd); in Fortran order below with the dimensions reversed. Interfaces are
! depths below the surface (zs(1) = zd(1) = 0); h is the thickness the content refers to
! (the source layer's content is c * h, spread over zs(k)..zs(k+1)).

module c_vertical_remap

   use iso_c_binding, only: c_int, c_double

   implicit none

contains

   subroutine ppm_edges(n, c, h, aL, aR)
      integer, intent(in) :: n
      real(c_double), intent(in) :: c(n), h(n)
      real(c_double), intent(out) :: aL(n), aR(n)

      real(c_double) :: d(n), e(n), da, dA_, a6, z1, z2, s
      logical :: inner(n)
      integer :: k, j

      aL = c
      aR = c
      if (n < 4) return

      ! slopes of the inner layers (1.7), monotonised (1.8)
      d = 0.0_c_double
      do k = 2, n - 1
         if (h(k-1) + h(k) + h(k+1) > 0.0_c_double .and. h(k+1) + h(k) > 0.0_c_double &
             .and. h(k-1) + h(k) > 0.0_c_double) then
            da = h(k) / (h(k-1) + h(k) + h(k+1)) * ((2 * h(k-1) + h(k)) / (h(k+1) + h(k)) * (c(k+1) - c(k)) &
                 + (h(k) + 2 * h(k+1)) / (h(k-1) + h(k)) * (c(k) - c(k-1)))
         else
            da = 0.0_c_double
         end if
         if ((c(k+1) - c(k)) * (c(k) - c(k-1)) > 0.0_c_double .and. da /= 0.0_c_double) then
            d(k) = sign(min(abs(da), 2 * min(abs(c(k) - c(k-1)), abs(c(k+1) - c(k)))), da)
         else
            d(k) = 0.0_c_double
         end if
      end do

      ! edge between layers j and j + 1 (1.6), j = 2 .. n - 2
      e = 0.0_c_double
      do j = 2, n - 2
         if (h(j-1) > 0.0_c_double .and. h(j) > 0.0_c_double .and. h(j+1) > 0.0_c_double &
             .and. h(j+2) > 0.0_c_double) then
            z1 = (h(j-1) + h(j)) / (2 * h(j) + h(j+1))
            z2 = (h(j+2) + h(j+1)) / (2 * h(j+1) + h(j))
            s = h(j-1) + h(j) + h(j+1) + h(j+2)
            e(j) = c(j) + h(j) / (h(j) + h(j+1)) * (c(j+1) - c(j)) &
                   + (2 * h(j+1) * h(j) / (h(j) + h(j+1)) * (z1 - z2) * (c(j+1) - c(j)) &
                      - h(j) * z1 * d(j+1) + h(j+1) * z2 * d(j)) / s
         else
            e(j) = 0.5_c_double * (c(j) + c(j+1))
         end if
      end do

      inner = .false.
      inner(3:n-2) = .true.
      do k = 3, n - 2
         aR(k) = e(k)
         aL(k) = e(k-1)
      end do

      ! limiter (1.10)
      do k = 1, n
         if ((aR(k) - c(k)) * (c(k) - aL(k)) <= 0.0_c_double .or. .not. inner(k)) then
            aL(k) = c(k)
            aR(k) = c(k)
         end if
         dA_ = aR(k) - aL(k)
         a6 = 6.0_c_double * (c(k) - 0.5_c_double * (aL(k) + aR(k)))
         if (dA_ * a6 > dA_ * dA_) then
            aL(k) = 3.0_c_double * c(k) - 2.0_c_double * aR(k)
         else if (-dA_ * dA_ > dA_ * a6) then
            aR(k) = 3.0_c_double * c(k) - 2.0_c_double * aL(k)
         end if
         if (.not. h(k) > 0.0_c_double) then
            aL(k) = c(k)
            aR(k) = c(k)
         end if
      end do
   end subroutine

   subroutine c_vertical_remap(n, ns, nd, nt, scheme, zs, zd, h, c, out) bind(c)
      integer(c_int), intent(in), value :: n, ns, nd, nt, scheme
      real(c_double), intent(in) :: zs(ns + 1, n), zd(nd + 1, n), h(ns, n), c(ns, n, nt)
      real(c_double), intent(inout) :: out(nd, n, nt)

      real(c_double) :: aL(ns), aR(ns), Qz(nd + 1), Qbase, Qtot, hk, f, z, dA_, a6
      integer :: i, t, m, j

      do t = 1, nt
         do i = 1, n
            if (scheme == 1) then
               call ppm_edges(ns, c(:, i, t), h(:, i), aL, aR)
            end if
            Qtot = sum(c(:, i, t) * h(:, i))
            j = 1
            Qbase = 0.0_c_double
            do m = 1, nd + 1
               z = zd(m, i)
               ! the source layer holding z: interfaces at z go to the layer below
               do while (j < ns)
                  if (zs(j + 1, i) > z) exit
                  Qbase = Qbase + c(j, i, t) * h(j, i)
                  j = j + 1
               end do
               hk = zs(j + 1, i) - zs(j, i)
               if (hk > 0.0_c_double) then
                  f = min(max((z - zs(j, i)) / hk, 0.0_c_double), 1.0_c_double)
               else
                  f = 0.0_c_double
               end if
               if (scheme == 1) then
                  dA_ = aR(j) - aL(j)
                  a6 = 6.0_c_double * (c(j, i, t) - 0.5_c_double * (aL(j) + aR(j)))
                  Qz(m) = Qbase + h(j, i) * (aL(j) * f + dA_ * f * f / 2 + a6 * (f * f / 2 - f * f * f / 3))
               else
                  Qz(m) = Qbase + f * c(j, i, t) * h(j, i)
               end if
            end do
            ! the outer interfaces exactly: no content lost to round-off
            Qz(1) = 0.0_c_double
            Qz(nd + 1) = Qtot
            do m = 1, nd
               out(m, i, t) = Qz(m + 1) - Qz(m)
            end do
         end do
      end do
   end subroutine

end module
