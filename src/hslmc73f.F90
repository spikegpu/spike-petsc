      subroutine HSLmc73(n, lirn, ia, ja, a, order, inprof, outprof, inbw, outbw, ierr)
      use hsl_mc73_double
      integer, parameter :: wp = kind(0.0d+0)
      integer :: job, n, lirn, inprof, outprof, inbw, outbw, ierr
      integer :: ia(*)
      integer :: ja(*)
      real (kind = wp) :: a(*)
      integer :: order(*)
      real (kind = wp), allocatable :: wgt(:)
      type (mc73_control) :: control
      integer :: info(1:10)
      real (kind = wp) :: rinfo(20)
      integer st

      call mc73_initialize(control)
      control%coarsest_size = 2
      job = 3

      if (a(1) > 0.0) then
        allocate(wgt(lirn),stat=st)
        if (st /= 0) then
          write (6,*) ' Allocation error'
          stop
        end if
        do 20 i=1,lirn
          wgt(i) = a(i)  
  20    end do
        call mc73_order(job,n,lirn,ja,ia,order,control,info,rinfo,wgt)
      else
        call mc73_order(job,n,lirn,ja,ia,order,control,info,rinfo)
      end if

      inprof  = int(rinfo(1))
      outprof = int(rinfo(5))
      inbw    = int(rinfo(3))
      outbw   = int(rinfo(7))
      if (a(1) > 0.0) then
        deallocate(wgt)
      end if 
      ierr = info(1)
      end subroutine
