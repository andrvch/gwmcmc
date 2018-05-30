      subroutine nsmax(ee,ne,param,ifl,photar,photer)

c-----------------------------------------------------------------------------
c     Model neutron star spectrum for magnetized, partially ionized atmosphere
c     (see Ho, WCG, Potekhin, AY, Chabrier, G 2008, ApJS, submitted)
c
c     ee(0:ne) = input energy bins
c     ne = input number of energy and spectral bins
c     param(3) = (fit) parameters
c       param(1) = log(unredshifted effective temperature, in K)
c       param(2) = redshift, 1+zg = 1/(1-2GM/Rc^2)^1/2
c       param(3) = model to use, see nsmax.dat
c       param(x) = normalization, (radius/distance)^2, with radius in km
c                  and distance in kpc (automatically added by XSPEC)
c     ifl = (not used)
c     photar(1:ne) = output count spectrum (in counts per bin)
c     photer(1:ne) = output count spectrum uncertainty (not used)
c-----------------------------------------------------------------------------

      implicit none
      INTEGER ne, ifl
      REAL ee(0:ne), param(3), photar(ne), photer(ne)

      INTEGER MAXFILES
      PARAMETER (MAXFILES=100)

      INTEGER ilun
      INTEGER ios
      CHARACTER(255) outstr, pstring, datfile, spfile(MAXFILES), infile
      INTEGER ispfile(MAXFILES), nspfiles
      INTEGER ntfile(MAXFILES), nefile(MAXFILES)
      INTEGER ntemp
      INTEGER nen
      REAL temp(20)
      REAL en(2000)
      REAL y(20,2000)

      INTEGER i, n
      REAL tt
      REAL redshift
      INTEGER ifile, ifilesv
      INTEGER it1, it2
      INTEGER ie1, ie2
      REAL ct1, ct2
      REAL ce1, ce2
      REAL energy
      REAL flux

      INTEGER lenact
      CHARACTER(255) fgmstr, fgmodf
      external lenact, fgmstr, fgmodf
      INTEGER lendir
      CHARACTER(255) pname, datdir

      LOGICAL qfirst, qnew

      SAVE qfirst, datdir, lendir
      SAVE nspfiles, ispfile, spfile, ntfile, nefile
      SAVE ntemp, nen, temp, en, y, ifilesv

      DATA qfirst /.TRUE./
      DATA ifilesv /0/
      DATA datdir /''/

c suppress a warning message from the compiler
      i = ifl

c this model does not calculate errors
      DO i = 1, ne
         photer(i) = 0.0
      ENDDO

c If the first time or NSMAX_DIR has changed read the available input files 
c from nsmax.dat

      pname = 'NSMAX_DIR'
      pstring = fgmstr(pname)
      IF ( lenact(pstring) .EQ. 0 ) pstring = datdir

      qnew = .FALSE.
      IF ( qfirst .OR. (pstring .NE. datdir) ) THEN

         datdir = pstring
         lendir = lenact(datdir)
         if (lendir.eq.0) then
            datdir = fgmodf()
            lendir = lenact(datdir)
         endif
         datfile = datdir(:lendir)//'nsmax.dat'

c Read nsmax.dat, which contains list of model spectra files;
c  format of nsmax.dat (2 columns): index number and spectrum file

         call getlun(ilun)
         call openwr(ilun,datfile,'old',' ',' ',0,0,ios)
         if (ios.ne.0) then
            outstr = 'Cannot open list file '//datfile(:lenact(datfile)) 
            call xwrite(outstr,10)
            return
         else
            outstr = 'Reading list of model spectra files from '//
     &           datfile(:lenact(datfile))
            call xwrite(outstr,25)
         endif
         
         nspfiles = 0
         read(ilun, *, end=20) ispfile(nspfiles+1), spfile(nspfiles+1),
     &        ntfile(nspfiles+1), nefile(nspfiles+1)
         do while (.TRUE.)
            nspfiles = nspfiles + 1
            read(ilun,*,end=20) ispfile(nspfiles+1), spfile(nspfiles+1),
     &        ntfile(nspfiles+1), nefile(nspfiles+1)
         enddo

 20      continue
         close(ilun)

         qnew = .TRUE.
         qfirst = .FALSE.
      ENDIF


      tt = param(1)
      redshift = param(2)
      ifile = NINT(param(3))

      IF ( qnew .OR. ifile .NE. ifilesv ) THEN

         infile = 'not found'
         DO i = 1, nspfiles
            IF ( ifile .EQ. ispfile(i) ) THEN
               infile = datdir(:lendir)//spfile(i)
               ntemp = ntfile(i)
               nen = nefile(i)
            ENDIF
         ENDDO
         IF ( infile .EQ. 'not found' ) THEN
            WRITE(outstr,'(a,i6)') 
     & 'No file found for input specfile parameter value of ', ifile
            call xwrite(outstr,10)
            WRITE(outstr,'(a,100(i6,1x))') 'Valid options are : ', 
     &           (ispfile(i),i=1,nspfiles)
            call xwrite(outstr,10)
            return
         ENDIF

c Read model spectra file, where ifile (parameter 3) = index number

         call openwr(ilun,infile,'old',' ',' ',0,0,ios)
         if (ios.ne.0) then
            outstr = 'Cannot open '//infile(:lenact(infile))
            call xwrite(outstr,10)
            return
         else
            outstr = 'Using '//infile(:lenact(infile))
            call xwrite(outstr, 25)
         endif
         do i = 1, ntemp
            if (i.eq.1) then
               read(ilun,*) (temp(n),n=1,ntemp)
               read(ilun,*) (en(n),n=1,nen)
               do n = 1, nen
                  en(n) = alog10(en(n))
               enddo
            endif
            read(ilun,*) (y(i,n),n=1,nen)
            do n = 1, nen
               y(i,n) = alog10(y(i,n))
            enddo
         enddo
         close(ilun)
         call frelun(ilun)

         ifilesv = ifile
      ENDIF

c Locate nearest two model temperatures and calculate weights
      if (tt.lt.temp(1)) then
        it1 = 1
        it2 = 2
        ct1 = 1.0
        ct2 = 0.0
      elseif (tt.gt.temp(ntemp)) then
        it1 = ntemp - 1
        it2 = ntemp
        ct1 = 0.0
        ct2 = 1.0
      else
        it1 = 1
        do i = 2, (ntemp-1)
          if (tt.lt.temp(i)) goto 40
          it1 = i
        enddo
 40     it2 = it1 + 1
        ct1 = (temp(it2) - tt) / (temp(it2) - temp(it1))
        ct2 = 1.0 - ct1
      endif
      do n = 1, ne
        energy = alog10(ee(n) * redshift)
c Locate nearest two model energies and calculate weights
        if (energy.lt.en(1)) then
          ie1 = 1
          ie2 = 2
          ce1 = 1.0
          ce2 = 0.0
        elseif (energy.gt.en(nen)) then
          ie1 = nen - 1
          ie2 = nen
          ce1 = 0.0
          ce2 = 1.0
        else
          ie1 = 1
          do i = 2, (nen-1)
            if (energy.lt.en(i)) goto 50
            ie1 = i
          enddo
 50       ie2 = ie1 + 1
          ce1 = (en(ie2) - energy) / (en(ie2) - en(ie1))
          ce2 = 1.0 - ce1
        endif
c Interpolated flux
        flux = ct1 * (ce1 * y(it1,ie1) + ce2 * y(it1,ie2))
     &    + ct2 * (ce1 * y(it2,ie1) + ce2 * y(it2,ie2))
c Scale by redshift
        flux = flux - alog10(redshift)
c Scale by (radius/distance)^2, where radius (in km) and distance (in kpc)
c PC=3.0856775807e18 cm
        flux = flux - 32.978701090
c Convert from ergs/(s cm^2 Hz) to counts/(s cm^2 keV)
c HH=6.6260693e-27 ergs s
        flux = flux + 26.178744 - alog10(ee(n))
        photar(n) = 10.0**flux
c Convert counts/(s cm^2 bin)
        photar(n) = photar(n) * (ee(n) - ee(n-1))
      enddo
      end
