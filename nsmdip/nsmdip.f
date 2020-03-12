      subroutine nsmdip(ee,ne,param,ifl,photar,photer)
*                                                       Version 19.02.20
* This is a preliminary version of nsmdir woith fixed M,R,B
c-----------------------------------------------------------------------
c     Model neutron star spectrum for magnetized, partially ionized
c     atmosphere with dipole magnetic field
c     Contact: Alexander Potekhin <palex-spb@yandex.ru>
c
c     ee(0:ne) = input energy bins
c     ne = input number of energy and spectral bins
c     param(4) = (fit) parameters
c       param(1) = model to use, see nsmdipmod.txt
c       param(2) = log(REDSHIFTED effective temperature, in K)
c       param(3) = distance to neutron star, in kpc
c       param(4) = magn. axis to line-of-sight angle, in degrees
c       param(x) = 1, normalization (automatically added by XSPEC)
c     ifl = (not used)
c     photar(1:ne) = output count spectrum (in counts per bin)
c     photer(1:ne) = output count spectrum uncertainty (not used)
c-----------------------------------------------------------------------

      implicit double precision (A-H), double precision (O-Z)
      character*14 FNAME
      character*8 FNPREF
      save
      parameter(MAXTHM=6,MAXEN=300,MAXT=6)
      parameter(EPS=1.d-5)
      parameter(PI=3.14159265359d0)
      dimension ee(0:ne), param(4), photar(ne), photer(ne)
      dimension ATHM(0:MAXTHM) ! integer array of input \theta_m [deg]
      dimension ATEFFLG(MAXT),AENLG(MAXEN)
      dimension ASPFLG(MAXEN,MAXT,0:MAXTHM) ! array of input spectra
      dimension ASPFL1(0:MAXTHM) ! 1D (buffer) array of input spectra
      character*2 ACHTHM(0:MAXTHM) ! array of \theta_m-related suffixes
      data ATHM/0.d0,15.d0,30.d0,45.d0,60.d0,75.d0,90.d0/
      data ACHTHM/'00','15','30','45','60','75','90'/
      data modelsv/1235/
      i = ifl ! included to suppress a compiler warning
      model=nint(param(1))
      if (model.eq.1) then
         FNPREF='spdip01m'
      elseif (model.eq.2) then
         FNPREF='spdip02m'
      else
        stop'nsmdip: unknown model'
      endif
* If the first time or the model has changed read the input data
      if (model.ne.modelsv) then
         DTHM=90.d0/MAXTHM
        do ITHM=0,MAXTHM ! for each \theta_m
           FNAME=FNPREF//ACHTHM(ITHM)//'.dat'
           write(*,'(''Reading '',A14)') FNAME
           open(1,file=FNAME,status='OLD')
           read(1,*)
           read(1,*)
           read(1,*)
           read(1,*) KB,LAT1,KBB,KS,MT,KD,thetam, ! read input param.
     *       SMASS,Rkm,B12pole,DeltaM_M,RgR,ZGRAV,GS14
          if (dabs(B12pole/3.-1.).gt.EPS) stop'nsmdip: invalid B'
          if (DeltaM_M.ne.0) stop'nsmdip: invalid accreted mass'
           read(1,*)
           read(1,*) (ATEFFLG(NT),NT=1,MT) ! read lg Teff(redshifted)
           read(1,*)
           read(1,*) ! (ATEFFLG(NT),NT=1,MT) ! read lg Teff(loc)
           read(1,*)
           read(1,*)
           MEN=0
    1     continue
           read(1,*,end=2) ENLG,(ASPFL1(ITT),ITT=1,MT)
           MEN=MEN+1
          if (MEN.gt.MAXEN) stop'energy number out of range'
          if (ITHM.eq.0) then ! the same energy grid for each \theta_m
             AENLG(MEN)=ENLG
          else
            if (dabs(ENLG-AENLG(MEN)).gt.EPS) stop'inconsistent E'
          endif
          do ITT=1,MT
            if (ASPFL1(ITT).lt.EPS) stop'zero flux'
             ASPFLG(MEN,ITT,ITHM)=dlog10(ASPFL1(ITT))
          enddo
          goto 1
    2     continue
        enddo ! next input \theta_m
         modelsv=model
      endif
* end reading
      TeffLG = param(2)
      DIST=param(3)
      THMdeg=param(4)
      if (TeffLG.lt.4..or.TeffLG.gt.8.) stop'nsmdip: implausible Teff'
      R6=Rkm/10.
      GR_R=dsqrt(1.d0-.295325*SMASS/R6)
      redshift = 1.d0/GR_R
      REDLG=dlog10(redshift)
      G14=1.3271244d0*SMASS/R6**2/GR_R
      gg = dlog10(G14)+14.d0 ! lg(g)
c Locate nearest two model temperatures...
      if (TeffLG.lt.ATEFFLG(1)) then
         it1 = 1
         it2 = 2
         print*,'nsmdip warning: too low Teff'
      elseif (TeffLG.gt.ATEFFLG(MT)) then
        it1 = MT - 1
        it2 = MT
      else
         it1 = 1
        do i = 2, (MT-1)
          if (TeffLG.lt.ATEFFLG(i)) goto 40
           it1 = i
        enddo
   40   continue
         it2 = it1 + 1
      endif
*  ...and calculate weights (interpolation or extrapolation)
      ct1 = (ATEFFLG(it2) - TeffLG) / (ATEFFLG(it2) - ATEFFLG(it1))
      ct2 = 1.d0 - ct1
c Locate nearest two axis inclinations \theta_m and calculate weights
      MODTHM=THM/180.d0
      THMdeg=dabs(THMdeg-180.d0*MODTHM)
      if (THMdeg.gt.90.d0) THMdeg=180.d0-THMdeg
      im1=THMdeg/DTHM
      if (im1.ge.MAXTHM) im1=im1-1
      im2=im1+1
      cm1=(ATHM(im2)-THMdeg)/DTHM
      cm2=1.d0-cm1
* Interpolate/extrapolate the flux
      do n = 1, ne
         ELG = log10(ee(n) * redshift)
c Locate nearest two model energies and calculate weights
        if (ELG.lt.AENLG(1)) then
           ie1 = 1
           ie2 = 2
        elseif (ELG.gt.AENLG(MEN)) then
           ie1 = MEN - 1
           ie2 = MEN
        else
           ie1 = 1
          do i = 2, (MEN-1)
            if (ELG.lt.AENLG(i)) goto 60
             ie1 = i
          enddo
 60       continue
           ie2 = ie1 + 1
        endif
         ce1 = (AENLG(ie2) - ELG) / (AENLG(ie2) - AENLG(ie1))
         ce2 = 1.d0 - ce1
c Interpolated/extrapolated flux logarithm
        FLG=cm1*(ct1*(ce1*ASPFLG(ie1,it1,im1)+ce2*ASPFLG(ie2,it1,im1))+
     +          ct2*(ce1*ASPFLG(ie1,it2,im1)+ce2*ASPFLG(ie2,it2,im1)))+
     +      cm2*(ct1*(ce1*ASPFLG(ie1,it1,im2)+ce2*ASPFLG(ie2,it1,im1))+
     +          ct2*(ce1*ASPFLG(ie1,it2,im2)+ce2*ASPFLG(ie2,it2,im2)))
c Scale by redshift
        FLG = FLG - REDLG
c Scale by (radius[km]/distance[kpc])^2
c PC=3.0856776e18 cm
        fluxLG = FLG - 32.9787011 + 2.d0*dlog10(Rkm/DIST)
c Convert from ergs/(s cm^2 keV) to counts/(s cm^2 keV)
c keV = 1.602176565d-9 erg
        fluxLG = fluxLG + 8.79529d0 - dlog10(ee(n))
        photar(n) = 10.d0**fluxLG
c Convert counts/(s cm^2 bin)
        photar(n) = photar(n) * (ee(n) - ee(n-1))
        photer(n) = 0.
      enddo
      end
