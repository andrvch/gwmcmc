
      SUBROUTINE nsa(ear,ne,param,ifl,photar,photer)

      IMPLICIT NONE

      INTEGER ne, ifl
      REAL ear(0:ne),param(4),photar(ne),photer(ne)


c------------------------------------------------------------------------------
c
c      ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c      a SHORT DESCRIPTION:
c
c       Spectrum of X-ray radiation from neutron
c       star  atmosphere.
c       with account for the Comptonization effect
c       (see Zavlin et al. 1996, A&A, 315, 141 and
c       Pavlov et al. 1992 MNRAS, 253, 193)
c       Models are available for three magnetic field strengths 0, 1e12, 1e13.
c       ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
c      INPUT PARAMETERS:
c
c   param(1) - Log of the effective (UNREDSHIFTED) temperature of the
c              neutron star surface (in K);
c              Log T=5.5-7.0
c   param(2) - neutron star gravitational mass (in solar mass)
c   param(3) - neutron star radius (in km)
c   param(4) - magnetic field strength
c
c-----------------------------------------------------------------------------

      REAL temp(21), ene(1000)
      REAL t, rms, rs, gr, sa, t1, a, t2, dt, e
      REAL de, f1, f2, f, ff, magfld, magsve

      DOUBLE PRECISION flux(21,1000)

      INTEGER ilun, lfil, ios, nn, mm, ninp, minp
      INTEGER i, j, jt, kk, k
      CHARACTER(255) filenm, contxt
      CHARACTER(128) pname

      CHARACTER(255) trfnam, fgmstr, fgmodf
      INTEGER lenact
      EXTERNAL trfnam, fgmstr, lenact, fgmodf

      SAVE ninp, minp, temp, ene, flux, t1, t2, magsve

      DATA t1, magsve / 0., -1./

c suppress a warning message from the compiler
      i = ifl

c this model does not calculate errors
      DO i = 1, ne
         photer(i) = 0.0
      ENDDO

      t = param(1)
      rms = param(2)
      rs = param(3)
      magfld = param(4)


      gr = sqrt(1.-2.952*rms/rs)
      sa = (rs/3.086e13)**2


c find out whether we need to read the data files

      if ( t1 .gt. 0. .AND. magsve .EQ. magfld ) go to 1

      magsve = magfld

      CALL getlun(ilun)

c Open the model file. First check for an NSA_FILE model string and if it is present
c use that. If it is not then look for the appropriate file in the standard manager
c directory.

      pname = 'NSA_FILE'
      filenm = fgmstr(pname)
      lfil = lenact(filenm)

      IF ( lfil .EQ. 0 ) THEN
         filenm = fgmodf()
         lfil = lenact(filenm)
         IF ( ABS(magfld) .LT. 1e9 ) THEN
            filenm = filenm(1:lfil)//'nsa_spec.dat'
         ELSEIF ( ABS(magfld-1e12) .LT. 1e9 ) THEN
            filenm = filenm(1:lfil)//'nsa_spec_B_1e12G.dat'
         ELSEIF ( ABS(magfld-1e13) .LT. 1e9 ) THEN
            filenm = filenm(1:lfil)//'nsa_spec_B_1e13G.dat'
         ELSE
            CALL xwrite(
     &       'The magnetic field must be one of 0, 1e12, or 1e13 G', 5)
            RETURN
         ENDIF
      ENDIF

      contxt = 'Using '//filenm(:lenact(filenm))
      CALL xwrite(contxt, 25)

      CALL OPENWR(ilun,filenm,'old',' ',' ',0,0,ios)
      IF ( ios .NE. 0 ) THEN
         contxt = 'NSA: Failed to open '//filenm(:lenact(filenm))
         CALL xwrite(contxt, 5)
         WRITE(contxt, '(a,i4)') 'Status = ', ios
         CALL xwrite(contxt, 5)
         CALL frelun(ilun)
         RETURN
      ENDIF

      read(ilun,*,iostat=ios) nn,mm
      contxt = 'NSA: Failed to read first line of file'
      IF ( ios .NE. 0 ) GOTO 999

      ninp=nn
      minp=mm

      read(ilun,*,iostat=ios) a,(temp(j),j=1,minp)
      contxt = 'NSA: Failed to read second line of file'
      IF ( ios .NE. 0 ) GOTO 999

      do i=1,ninp
         read(ilun,*) ene(i),(flux(j,i),j=1,minp)
         WRITE(contxt,'(a,i4,a)') 'NSA: Failed to read line ', i+2,
     &                            'of the file'
         IF ( ios .NE. 0 ) GOTO 999
         ene(i)=alog10(ene(i))
         do j=1,minp
            if(flux(j,i).gt.0.0) then
               flux(j,i)=log10(flux(j,i))
            else
               flux(j,i)=flux(j,i-1)
            endif
         enddo
      enddo
      close(ilun)
      CALL frelun(ilun)

      t1=temp(1)
      t2=temp(minp)

c jump to here if we did not need to read a data file

1     continue

      do jt=2,minp
         if(temp(jt).ge.t) go to 2
      enddo
      jt=minp
2     dt=(t-temp(jt-1))/(temp(jt)-temp(jt-1))

      kk=2

      do i=0,ne
         e=alog10(ear(i)/gr)
         if(e.lt.ene(1)) e=ene(1)
         if(e.gt.ene(ninp)) go to 4

         do k=kk,ninp
            if(ene(k).ge.e) go to 3
         enddo

3        de=(e-ene(k-1))/(ene(k)-ene(k-1))
         f1=REAL(flux(jt-1,k-1))+de*(REAL(flux(jt-1,k)-flux(jt-1,k-1)))
         f2=REAL(flux(jt,k-1))+de*(REAL(flux(jt,k)-flux(jt,k-1)))
         f=f1+dt*(f2-f1)
         f=10**f*sa
         go to 5
4        photar(i)=photar(i-1)
         go to 6
5        if(i.eq.0) go to 7
         photar(i)=(f+ff)/2.*(ear(i)-ear(i-1))
7        ff=f
         kk=k
6     enddo

      RETURN
 999  IF ( ios .NE. 0 ) THEN
         CALL xwrite(contxt, 5)
         WRITE(contxt, '(a,i4)') 'Status = ', ios
         CALL xwrite(contxt, 5)
         CLOSE(ilun)
         CALL frelun(ilun)
      ENDIF

      end
