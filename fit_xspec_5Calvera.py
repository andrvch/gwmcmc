#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
from xspec import *
import matplotlib.pyplot as plt
import math

####aa@guitar:~/Astro/j1741/Xrays$ python fit_xspec_5.py chi chi **-0.3 10.0-** bb 0.14 2.5 1 Yes Yes No 1230 No Yes > LOGbbTeffRlum_3.LOG &

######################################################constants
pi   = 3.141592654
GeV  = 1.6022E-3          # GeV in ergs
MeV  = 1.6022E-6          # MeV in ergs
eV   = 1.6022E-12         # eV in ergs
KeV  = 1.6022E-9          # KeV (in ergs)
eee  = 4.8032E-10         # elementary charge, CGS
hh   = 1.054572E-27       # erg*s
h    = 6.626068E-27       # erg*s, 2*pi*hh
c    = 2.997924E10        # cm/s
me   = 0.510998918        # MeV, electron mass
mp   = 938.27203          # MeV proton mass
mn   = 939.56536          # MeV neutron mass
ang  = 1.E-8              # cm in angstrom
kb   = 1.38E-16           # erg/K
sig  = 5.6704E-5          # erg*cm**{-2}*s**{-1}*K**{-4}
arad = 7.56E-15           # erg*cm**{-3}*K**{-4}
km   = 1.E5               # cm
Ms   = 1.989E33           # g
Rs   = 6.96E10            # cm, the Sun radius
Mn   = 1.4*Ms             # a neutron star mass
Rn   = 10*km              # a neutron star radius
pc   = 3.085E18           # cm, parsec
au   = 1.496E13           # cm
muJy = 1.E29              # \muJy
G    = 6.674E-8           # cm**{3}*s**{-1}*g**{-1}
yr   = 3.E7                # seconds in year
mpp  = (mp*MeV)*pow(c,-2) # proton mass in g
#####################################################end constants
Xset.parallel.leven = 4
Xset.parallel.error = 4
Mns = 1.4
Rns = 13.
D = 1000.
Dbb = D*1E-04
Dnsmax = D*1E-03
redshift = (1-2.952*(Mns/Rns))**(-1./2.)
#print redshiftR
################################sss
#FileName = sys.argv[1]
Stat        = sys.argv[1]
StatTest    = sys.argv[2]
ignore_less = sys.argv[3]
ignore_more = sys.argv[4]
therm       = sys.argv[5]
nh          = float(sys.argv[6])
gamma_i     = float(sys.argv[7])
specnumber  = int(sys.argv[8])
ifgroup     = sys.argv[9]
#annul       = sys.argv[11]
if therm == "nsmax" or therm == "2nsmax" or therm == "nsmax2pl":
  whatatm     = sys.argv[10]
elif  therm == "nsa" or therm == "2nsa":
  Bns = float(sys.argv[10])
ignore_lessXMM = sys.argv[11]
ignore_moreXMM = sys.argv[11]
#with2010 = sys.argv[15]
Xset.abund = "angr"
#Xset.abund = "aspl"
#Xset.abund = "feld"
#Xset.abund = "aneb"
#Xset.abund = "grsa"
#Xset.abund = "wilm"
Xset.xsect = "bcmc"
print Xset.abund
print Xset.xsect
if ifgroup == "Yes":
  #SPECNAME = "1:1 Calvera-M1-1.pi 1:2 Calvera-PN-1.pi 1:3 Calvera-M1-2.pi 1:4 Calvera-PN-2.pi "
  SPECNAME = "1:1 Calvera-M1-1.pi 1:2 Calvera-PN-1.pi 1:3 Calvera-M1-2.pi 1:4 Calvera-PN-2.pi  2:5 calvera-acis-cc1_grp.pi 2:6 calvera-acis-s_grp.pi 2:7 calvera-acis-cc2_grp.pi"
  #SPECNAME = "1:1 calvera-acis-cc1_grp.pi 1:2 calvera-acis-cc2_grp.pi"
  #SPECNAME = "calvera-acis-s_grp.pi"
  #SPECNAME = "1:1 psr1741-1_grp.pi 1:2 psr1741-2_grp.pi 1:3 psr1741-3_grp.pi 1:4 psr1741_15542_grp.pi 1:5 psr1741_15543_grp.pi 1:6 psr1741_15544_grp.pi 1:7 psr1741_15638_grp.pi"
elif  ifgroup == "No":
  #SPECNAME = "calvera-acis-s.pi"
  #SPECNAME = "1:1 calvera-acis-cc1.pi 1:2 calvera-acis-cc2.pi 1:3 calvera-acis-s.pi"
  #SPECNAME = "1:1 M1-1-calvera.fits 1:2 PN1-calvera.fits 1:3 M1-2-calvera.fits 1:4 PN2-calvera.fits 1:5 calvera-acis-cc1.pi 1:6 calvera-acis-cc2.pi 1:7 calvera-acis-s.pi"
  SPECNAME = "1:1 Calvera-M1-1.pi 1:2 Calvera-PN-1.pi 1:3 Calvera-M1-2.pi 1:4 Calvera-PN-2.pi  1:5 calvera-acis-cc1.pi 1:6 calvera-acis-cc2.pi 1:7 calvera-acis-s.pi"
  #SPECNAME = "1:1 psr1741-1.pi 1:2 psr1741-2.pi 1:3 psr1741-3.pi 1:4 psr1741_15542.pi 1:5 psr1741_15543.pi 1:6 psr1741_15544.pi 1:7 psr1741_15638.pi 1:8 spectrumPN.fits 1:9 spectrumM1.fits 1:10 spectrumM2.fits"
  #SPECNAME = "1:1 spectrumPN.fits 1:2 spectrumM1.fits 1:3 spectrumM2.fits"
########################################################################3Data
AllData(SPECNAME)
#s1.background = ""
AllData.ignore(ignore_lessXMM)
AllData.ignore(ignore_moreXMM)
#for i in range(1,5):
#  AllData(i).ignore(ignore_lessXMM)
#  AllData(i).ignore(ignore_moreXMM)
#for i in range(1,5):
#  AllData(i).ignore(ignore_lessXMM)
#  AllData(i).ignore(ignore_moreXMM)
#AllData.ignore("bad")
#for i in range(5,8):
#  AllData(i).ignore(ignore_less)
#  AllData(i).ignore(ignore_more)
#AllData(9).ignore("**-0.2")
#AllData(9).ignore("10.0-**")
#AllData(10).ignore("**-0.2")
#AllData(10).ignore("10.0-**")
AllData.ignore("bad")
##########################################################444
#################################################1####statistics
if therm == "nsmax":
  AllModels += "phabs*(powerlaw+nsmax)"
  m1 = AllModels(1)
  m2 = AllModels(2)
  par1 = m1.phabs.nH
  par2 = m1.powerlaw.PhoIndex
  par3 = m1.powerlaw.norm
  par4 = m1.nsmax.logTeff
  par5 = m1.nsmax.redshift
  par6 = m1.nsmax.specfile
  par7 = m1.nsmax.norm
  par1_1 = m2.phabs.nH
  par2_1 = m2.powerlaw.PhoIndex
  par3_1 = m2.powerlaw.norm
  par4_1 = m2.nsmax.logTeff
  par5_1 = m2.nsmax.redshift
  par6_1 = m2.nsmax.specfile
  par7_1 = m2.nsmax.norm
  ###################end
  #par1.values = 0.1
  #par1.frozen = True
  #par2.values = 1.22
  #par1.values = [0.06, 0.001,0.,0.,nh,nh]
  par1.values = [nh, 0.0001, 0.0001, 0.001, 0.3, 0.3]
  #par1.frozen = True
  par2.values = gamma_i
  par2.frozen = True
  par3.values = 0.0 #9.87e-6
  par3.frozen = True
  par4.values = [5.7,0.01,5.5,5.5,6.5,6.5]
  par5.values = [redshift,0.01,1.1,1.1,2.,2.]
  par5.frozen = True
  par6.values = int(whatatm)
  par6.frozen = True
  par4_1.untie()
  # par7.values = [(R/Dnsmax)**2, .01, (8./Dnsmax)**2,  (8./Dnsmax)**2, (20./Dnsmax)**2, #(20./Dnsmax)**2]
  #par7.values = [redshift*(Rns/Dnsmax)**2, .01, redshift*(8./(Dnsmax+0.5))**2,  redshift*(8./(Dnsmax+0.5))**2, redshift*(20./(Dnsmax-0.2))**2, redshift*(20./(Dnsmax-0.2))**2]
  #par6.values = 10
  #par6.frozen = True
  #par7.link = "1417*5"
elif therm == "bb":
  AllModels += "phabs*(powerlaw+bbodyrad)"
  m1 = AllModels(1)
  #m2 = AllModels(2)
  par1 = m1.phabs.nH
  par2 = m1.powerlaw.PhoIndex
  par3 = m1.powerlaw.norm
  par4 = m1.bbodyrad.kT
  par5 = m1.bbodyrad.norm
  #par1_2 = m2.phabs.nH
  #par2_2 = m2.powerlaw.PhoIndex
  #par3_2 = m2.powerlaw.norm
  #par4_2 = m2.bbodyrad.kT
  #par5_2 = m2.bbodyrad.norm
  #par6 = m1.bbodyrad_4.kT
  #par7 = m1.bbodyrad_4.norm
  par1.values = [nh,0.001, 0.001, 0.001, 0.3, 0.3]
  #par1_2
  #par1.frozen = True
  par2.values = [gamma_i,0.01,1.0,1.0,9.0,9.0]
  #par2_2.values = [gamma_i,0.01,1.0,1.0,3.0,3.0]
  #par3_2.values = 1.23337E-04
  ###################################en
  par4.values = [0.03, 0.001, 0.01, 0.01, 0.5, 0.5]
  par5.values = (redshift*Rns/Dbb)**2 #[,0.01,(redshift*(Rns-5.)/(Dbb+0.072))**2,(redshift*(Rns-5.)/(Dbb+0.072))**2,(redshift*(Rns+7.)/(Dbb-0.03))**2,(redshift*(Rns+7.)/(Dbb-0.03))**2]
  #par4.frozen = True
  #par5.frozen = True
  #par4.values = 0.1
elif therm == "notherm":
  AllModels += "phabs*(powerlaw)"
  m1 = AllModels(1)
  par1 = m1.phabs.nH
  par1.values = [nh, 0.001, 0.001, 0.001, 0.3, 0.3]
  #par1.frozen = True
  par2 = m1.powerlaw.PhoIndex
  par3 = m1.powerlaw.norm
  par2.values = 2.65
  par2.frozen = True

Fit.statMethod = Stat
Fit.statTest   = StatTest
#if iffit == "Yes":
Fit.query = "yes"
Fit.perform()

Fit.goodness(nRealizations = 300,sim = False)
AllModels.calcFlux("0.2 10.0")
#AllModels.calcFlux("0.3 1.0")
#print type(AllModels.calcFlux("2.0 10.0"))
#FluxNT = float(FluxNT)
Dns    = 380*3.086E18
LuNT   = 4*pi*Dns**2*1.2E-13 #FluxNT
LuGNT  =  4*pi*Dns**2*1.2E-10
EffNT  = LuNT/9.5E33
EffGNT = LuGNT/9.5E33
#print LuNT,EffNT
#print LuGNT,EffGNT
#print Fit.statistic
#print Fit.dof

#Plot.addCommand("Font Roman")
#Plot.addCommand("R Y2 -6. 6.")
Plot.addCommand("R Y1 1.E-5 1.0")
#Plot.addCommand("CS 0.8")
Plot.addCommand("Time Off")
Plot.addCommand('LA TOP')
Plot.addCommand("LWidth 2. on 5")
#Plot.addCommand("Hardcopy")
#Plot.addCommand('LA 1 Vpos 0.83 0.85 CS 1.3 "BB+PL"')
Plot.device = "/xs"
Plot.xAxis = "keV"
#Plot.add = True
#Plot.background = True
#Plot.yAxis = "Flux"
#Plot("data")
#iffit == "Yes":
  #Plot("ufspec","resid")
Plot("ldata","resid")
#Plot("ldata",)
#Plot("goodness")
if errorsyes == "Yes":
  Plot("contour")
#Plot("model")
#Plot.xAxis = "energy"
#Plot("energy")
#Plot()
#print redshift
AllData.clear()
AllModels.clear()
Xset.parallel.reset()
#Fit.reset()
