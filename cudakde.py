#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import matplotlib.pyplot as plt
import math
import numpy as np
from pylab import *
import random
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import kde
import scipy.ndimage
from numpy import linalg as la
from pycuda import driver, compiler, gpuarray, tools
import time
# -- initialize the device
import pycuda.autoinit

kernel_code_template_cov_1d = """
__global__ void gauss_kde1d ( float *a, float *b, float *pdf ) {
    int ndata = %(DATA_SIZE)s;
    float h = %(BAND_W)s;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float pi = 3.141592654;
    float sum_ker = 0;
    float prod_ker;
    for ( int m = 0; m < ndata; m++ ) {
        prod_ker = expf ( - powf ( a[id] - b[m], 2 ) / 2 / powf ( h, 2 ) ) / h;
        sum_ker += prod_ker;
    }
    pdf[id] = ( 1 / powf ( 2 * pi , 0.5 ) ) * sum_ker / ndata;
}
"""

kernel_code_template_cov = """
__global__ void gauss_kde ( float *a, float *b, float *c, float *pdf ) {
    int npars = %(N_PAR)s;
    int ndata = %(DATA_SIZE)s;
    float h = %(FACT_D)s;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float pi = 3.141592654;
    float sum_ker = 0;
    float prod_ker;
    for ( int m = 0; m < ndata; m++ ) {
        prod_ker = expf ( - ( c[0] * powf ( a[id*npars+0] - b[m*npars+0], 2 ) + c[1] * ( a[id*npars+0] - b[m*npars+0] ) * ( a[id*npars+1] - b[m*npars+1] ) + c[2] * ( a[id*npars+0] - b[m*npars+0] ) * ( a[id*npars+1] - b[m*npars+1] ) + c[3] * powf ( a[id*npars+1] - b[m*npars+1], 2 ) ) / 2 / h ) / powf ( h, 2 );
        sum_ker += prod_ker;
    }
    pdf[id] = (1/(2*pi))*%(DET_C)s*sum_ker/ndata;
}
"""

def read_data(FileName):
    lines = []
    with open(FileName) as fp:
        for line in iter(fp.readline, ''):
            lines.append(str(line))
    npars = len(lines[1].split())
    nsmpl = len(lines)
    pars = np.empty([npars,nsmpl])
    for i in range(nsmpl):
        for j in range(npars):
            pars[j,i] = lines[i].split()[j]
    return pars

def read_data_nsmpl(FileName,nsm):
    lines = []
    with open(FileName) as fp:
        for line in iter(fp.readline, ''):
            lines.append(str(line))
    random.shuffle(lines)
    npars = len(lines[1].split())
    pars = np.empty([npars,nsm])
    for j in range(npars):
        for i in range(nsm):
            pars[j,i] = lines[i].split()[j]
    return pars

def kde_gauss_cuda2d(x,y,nbins2D):
    sigmax = np.std(x)
    sigmay = np.std(y)
    data = np.vstack([x,y])
    xi,yi = np.mgrid[x.min():x.max():nbins2D*1j,y.min():y.max():nbins2D*1j]
    xiyi = np.vstack([xi.flatten(),yi.flatten()])
    npars = shape(xiyi)[0]
    nbins = shape(xiyi)[1]
    nsmpl = shape(data)[1]
    cdat = np.cov(data)
    factd = nsmpl**(-1./3.)
    invc = np.linalg.inv(cdat)
    invcf = invc.flatten()
    detinvc = np.linalg.det(invc)
    detsqrin = 1./math.sqrt(detinvc)
    invcfno = invcf
    xiyiT = xiyi.T.flatten()
    dataT = data.T.flatten()
    data_gpu = gpuarray.to_gpu(dataT.astype(np.float32))
    xiyi_gpu = gpuarray.to_gpu(xiyiT.astype(np.float32))
    pdf_gpu = gpuarray.zeros(nbins,np.float32)
    invcfno_gpu = gpuarray.to_gpu(invcfno.astype(np.float32))
    b_s = 512
    t_s = 1024*npars
    # get the kernel code from the template
    kernel_code = kernel_code_template_cov % {
        'DATA_SIZE': nsmpl,
        'N_PAR': npars,
        'DET_C': detsqrin,
        'FACT_D': factd,
        }
    # compile the kernel code
    mod = compiler.SourceModule(kernel_code)
    # get the kernel function from the compiled module
    cuda_gauss_kde = mod.get_function("gauss_kde")
    # call the kernel on the card
    cuda_gauss_kde(
        # inputs
        xiyi_gpu, data_gpu, invcfno_gpu,
        # output
        pdf_gpu,
        #
        grid = (nbins // b_s, 1),
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (b_s, 1, 1),
        )
    return xi,yi,pdf_gpu.get() #/(xi.flatten()[1]-xi.flatten()[0])/(yi.flatten()[1]-yi.flatten()[0])

def kde_gauss_cuda1d(x,nbins1D):
    nsmpl = len(x)
    nbins = nbins1D
    sigmax = np.std(x)
    bandwidth = 1.06*sigmax*nsmpl**(-1./5.)
    xi = linspace(x.min(),x.max(),nbins1D)
    x_gpu = gpuarray.to_gpu(x.astype(np.float32))
    xi_gpu = gpuarray.to_gpu(xi.astype(np.float32))
    pdf_gpu = gpuarray.zeros(nbins,np.float32)
    b_s = 16
    # get the kernel code from the template
    kernel_code = kernel_code_template_cov_1d % {
        'DATA_SIZE': nsmpl,
        'BAND_W': bandwidth,
        }
    # compile the kernel code
    mod = compiler.SourceModule(kernel_code)
    # get the kernel function from the compiled module
    cuda_gauss_kde = mod.get_function("gauss_kde1d")
    # call the kernel on the card
    cuda_gauss_kde(
        # inputs
        xi_gpu, x_gpu,
        # output
        pdf_gpu,
        #
        grid = (nbins // b_s, 1),
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (b_s, 1, 1),
        )
    return xi,pdf_gpu.get() #*(xi[1]-xi[0])

def kde_gauss_scipy(x,y,nbins2D):
    data = np.vstack([x,y])
    xi,yi = np.mgrid[x.min():x.max():nbins2D*1j,y.min():y.max():nbins2D*1j]
    xiyi = np.vstack([xi.flatten(),yi.flatten()])
    k = kde.gaussian_kde(data)
    zi = k(xiyi)
    return xi,yi,zi

def comp_lev(zi,quont):
    zisort = np.sort(zi)[::-1]
    zisum = np.sum(zi)
    zicumsum = np.cumsum(zisort)
    zicumsumnorm = zicumsum/zisum
    zinorm = zi/zisum
    zisortnorm = zisort/zisum
    levels = [zisortnorm[np.where(zicumsumnorm>qu)][0] for qu in quont]
    return levels,zinorm

def prc(xi,zi,qu):
    zisort = np.sort(zi)[::-1]
    zisum = np.sum(zi)
    zimax = np.max(zi)
    zicumsum = np.cumsum(zisort)
    zicumsumnorm = zicumsum/zisum
    zisortnorm = zisort/zisum
    zinorm = zi/zisum
    qlevel = zisortnorm[np.where(zicumsumnorm>qu)][0]
    quat = xi[np.where(zinorm>qlevel)]
    quatmax = xi[np.where(zi==zimax)][0]
    return zinorm,np.array([quat[0], quatmax, quat[-1]])
