#!/usr/bin/env python
from __future__ import division,with_statement

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import scipy

import pyfits
import pymc


#<------------------------fake models------------------------------------------>

def gen_lines(n,x,upperline=0,lowerline=-0.4,widthcen=20,widthsig=20):
    from astropysics.models import GaussianModel,LorentzianModel,VoigtModel
    tmap={0:'g',1:'l',2:'v'}
    type=[tmap[i] for i in 3*random.rand(n).astype(int)]
    peak=(upperline-lowerline)*random.rand(n)+lowerline
    xmax,xmin=max(x),min(x)
    center=(xmax-xmin)*random.rand(n)+xmin
    width=abs(widthsig*random.randn(n)+widthcen)
    lines=[]
    for t,p,c,w in  zip(type,peak,center,width):
        m=None
        if t == 'g':
            m=GaussianModel()
            m.sig = w
        if t == 'l':
            m=LorentzianModel()
            m.gamma = w
        if t == 'v':
            m=VoigtModel()
            m.gamma = m.sig = w/2
        assert m is not None
        
        m.peak = p
        m.mu = c
        lines.append(m)
    return lines
        
    
def lines_to_template(lines,x,T=5800):
    from astropysics.models import BlackbodyModel
    bm=BlackbodyModel(T=T)
    bm.peak = 1
    flux=bm(x)
    for l in lines:
        flux+=l(x)
        
    flux[flux<=0]=0.01
    
    #bm ignore
    return flux
    #return flux-bm(x)

def lines_to_spec(x,lines,T=5800,psnr=100,vel=200,ivarsamples=3):
    import astropysics.models,scipy.stats
    
    z=vel/3e5
    restx=10**(log10(x)-log10(z+1))
    
    
    bm=astropysics.models.BlackbodyModel(T=T)
    bm.peak = 1
    flux = bm(restx)
    for l in lines:
        flux+=l(restx)
        
    flux[flux<=0]=0.01
        
        
    pn=psnr*psnr
#    #dist=scipy.stats.poisson(pn)
#    #noise=dist.rvs(len(restx))-pn
#    noise = array([scipy.stats.poisson.rvs(pn*f)[0]/pn/f-1 for f in flux])
#    return flux+array([scipy.stats.poisson.rvs(pn*f)[0]/pn/f-1 for f in flux]),None
    
#    return flux+noise
    
    exps = array([[scipy.stats.poisson.rvs(pn*f)/pn for f in flux] for i in range(ivarsamples)])
    
    flux = mean(exps,axis=0)
    vars = var(exps,axis=0)
    minvar = (1/ivarsamples-1/ivarsamples/ivarsamples)*pn**-2 #min var of N/N0+N/N0+...+(N+1)/N0 is (1/k - 1/k^2)*N0^-2
    vars[vars< minvar]=  minvar
    
    #bm ignore
    return flux,1.0/vars
    #return flux-bm(restx),1.0/vars
    
def lines_to_spec_multi(x,lines,As,Ts,psnr=100,vel=200,ivarsamples=3):
    """
    does a linear combination of templateswith lines given in the sequence of 
    sequences lines, with coefficients As, and BB temperatures Ts
    """
    import astropysics.models,scipy.stats
    
    z=vel/3e5
    restx=10**(log10(x)-log10(z+1))
    
    flux = zeros_like(restx)
    for T,A,lineset in zip(Ts,As,lines):
        bm=astropysics.models.BlackbodyModel(T=T)
        bm.peak = 1
        fluxi = bm(restx)
        for l in lineset:
            fluxi+=l(restx)
        flux += A*fluxi    
            
    flux[flux<=0]=0.01  
        
    pn=psnr*psnr
#    #dist=scipy.stats.poisson(pn)
#    #noise=dist.rvs(len(restx))-pn
#    noise = array([scipy.stats.poisson.rvs(pn*f)[0]/pn/f-1 for f in flux])
#    return flux+array([scipy.stats.poisson.rvs(pn*f)[0]/pn/f-1 for f in flux]),None
    
#    return flux+noise
    
    exps = array([[scipy.stats.poisson.rvs(pn*f)[0]/pn for f in flux] for i in range(ivarsamples)])
    
    flux = mean(exps,axis=0)
    vars = var(exps,axis=0)
    minvar = (1/ivarsamples-1/ivarsamples/ivarsamples)*pn**-2 #min var of N/N0+N/N0+...+(N+1)/N0 is (1/k - 1/k^2)*N0^-2
    vars[vars< minvar]=  minvar
    
    #bm ignore
    return flux,1.0/vars

def generate_templates(n,x,nl=10,temps=5800,**kwargs):
    ts,lines=[],[]
    if isscalar(temps):
        temps=ones(n)*temps
    for i in range(n):
        lines.append(gen_lines(nl,x,**kwargs))
        ts.append(lines_to_template(lines[-1],x,temps[i]))
    return ts,lines

def fit_BB_and_sub(x,flux):
    from astropysics.models import BlackbodyModel
    bm = BlackbodyModel()
    bm.fitData(x,flux)
    return flux-bm(x)


#<----------------------------DEIMOS data-------------------------------------->
def get_templates(templatedir="./templates"):
    """
    returns (xstar,dstar),(xgal,dgal),(xab,dab)
    """
    import pyfits,glob,os
    if not templatedir.endswith(os.sep):
        templatedir+=os.sep
        
    #tfns=glob.glob(templatedir+'*.fits')
    
    xs,ds=[],[]
    for fn in ('deimos-021507.fits','deimos-aband.fits','deimos-galaxy.fits'):
        f=pyfits.open(templatedir+fn)
        try:
            ds.append(f[0].data)
            h=f[0].header
            xs.append(h['COEFF0']+h['COEFF1']*arange(ds[-1].shape[-1]))
        finally:
            f.close()
    
    xstar,xab,xgal=xs
    dstar,dab,dgal=ds
    return (xstar,dstar),(xgal,dgal),(xab,dab)

def mask_slit_to_fn(mask,slitnum):
    from glob import glob
    from os import sep
    
    fns = glob('marladata'+sep+mask+sep+'*')
    fnmatch = [fn for fn in fns if '%.3i'%slitnum in fn if 'serendip' not in fn]
    if len(fnmatch)>1:
        raise ValueError('multi match')
    return fnmatch[0]

def match_file_to_fns(fn='marladata/hirescomp'):
    from astropysics import io
    
    d = io.loadtxt_text_fields(fn)
    
    masks = d['mask']
    slitnums = d['slit']

    specfns = [mask_slit_to_fn(mask,slitnum) for mask,slitnum in zip(masks,slitnums)]
    
    return specfns,d['vzspec'],d['vcorr'],d['vhires']


#<-------------------Model/Vel Fitting----------------------------------------->
def pixshift_to_vel(shift,x,zout=False,logify=True,lincheck=True):
    """
    if x is in log(lambda), set logify = False
    """    
    if logify:
        x = log10(x)
    
    dlogx=(max(x)-min(x))/len(x)
    
    if lincheck:
        dxs=convolve(x,[1,-1],mode='valid')
        if max(dxs)/min(dxs) > 2 :
            from warnings import warn
            warn('average dx varies by factor of %f - not linear in %s?'%(max(dxs)/min(dxs),'log(lambda)' if logify else 'lambda'))
    
    z = 10**(shift*dlogx)-1

    if zout:
        return z
    else:
        return z*3e5
    
def vel_to_pixshift(vel,x,logify=True):
    from  scipy.optimize import fmin
    f = lambda *args:np.abs(vel-pixshift_to_vel(*args))
    return fmin(f,0,args=(x,False,logify))
    

def generate_MCMC_model(specobj,templates,offset=False,shiftout='velocity',v0=0,multitemps=False,copy=False):
    """
    Makes a PyMC model for the given data and x-axis to scale and offset from a
    template:
    
    offset can specify an offset, or be True to have it be a free variate
    
    shiftout can be 'vel','z',or 'pix'
    
    v0 determines the initial value of the pixshift (random if None)
    
    multitemps determines if a linear combination of templates should be used
    
    data=Normal(tau=ivar,center=A*template(x-shift)+offset)
    """
    import pymc
    
    x,flux,ivar = specobj.x.copy(),specobj.flux.copy(),specobj.ivar.copy()
    templates=array(templates,copy=copy)
    
    if any([t.shape!=templates[0].shape for t in templates]):
        raise ValueError("templates don't match")
    ntempix = templates[0].shape[0]
    npix = x.shape[0]
    
    #TODO: match pixels with alignment instead of just assuming middle
    tx = arange(ntempix)
    sx = arange(npix)+(ntempix-npix)/2 #spectrum x-value in terms of template coordinates
    
    tmax,tmin=templates.max(),templates.min()
    fmax,fmin=flux.max(),flux.min()
    imax,imin=ivar.max(),ivar.min()
    xmax,xmin=x.max(),x.min()
    ivar0 = imin/npix/100 #A small value to un-weight bad data points
    
    maxlshift=maxrshift=(ntempix-npix)/2 #TODO:fix for non-symmetric
    
    if offset:
        loff,uoff=(fmin+tmin,fmax+tmax) #TODO:fix
    else:
        loff=uoff=float(offset)
    
    offset=pymc.Uniform('offset',loff,uoff,trace=bool(offset),plot=bool(offset))
    
    pixshift=pymc.Uniform('pixshift',-round(npix/2),round(npix/2),trace=True,plot=False)
    if v0 is not None:
        pixshift.value = v0
    svar=None
    if 'vel' in shiftout:
        svar=pymc.Lambda('vel',lambda pixshift=pixshift:pixshift_to_vel(pixshift,x,zout=False,logify=True,lincheck=False))
    elif shiftout == 'z':
        svar=pymc.Lambda('z',lambda pixshift=pixshift:pixshift_to_vel(pixshift,x,zout=True,logify=True,lincheck=False))
    elif 'pix' in shiftout:
        svar=pixshift
    else:
        raise ValueError('unrecognized shiftout')
    svar.plot = True
    svar.trace = True
    
    elems={'offset':offset,'pixshift':pixshift}
    if svar is not pixshift:
        elems[svar.__name__]=svar
    
    if multitemps:
        #TODO:smarter initial
        A = pymc.Container([pymc.Uniform('A%i'%i,0,fmax/np.max(t)) for i,t in enumerate(templates)])
        elems['A']=A
        for e in A:
            e.plot = False
            e.value = 0.1
        A[0].value = 1
        
        @pymc.deterministic(trace=True,plot=False)
        def modelflux(A=A,offset=offset,pixshift=pixshift):
            #TODO:caching of some kind ?
            
            temp = sum((A*templates.T),1) #TODO:test
            
            
            #r = int(round(pixshift))
            
            shifted = interp(sx-pixshift+1,tx,temp) #TODO:why +1 ?
            #shifted = roll(temp,r)
            
#            if r > 0:
#            #    shifted[:r]=flux[:r] #do something smarter here for the edges
#                shifted[:r]=shifted[r]
#            elif r < 0:
#            #    shifted[r:]=flux[r:] #do something smarter here for the edges
#                shifted[r:]=shifted[r]
            
            
            return shifted+offset
        
        elems['modelflux']=modelflux
    else:
        A=pymc.Uniform('A',1,1,value=1) #TODO:smarter setting
        elems['A']=A
        templatei=pymc.DiscreteUniform('templatei',0,len(templates)-1,trace=True,plot=False)
        templatei.value = 0
        elems['templatei']=templatei
        
        @pymc.deterministic(trace=True,plot=False)
        def modelflux(A=A,offset=offset,pixshift=pixshift,templatei=templatei,templates=templates):
            """
            The flux expected from the template parameters
            """
            #TODO:caching of some kind ?
            
            temp = templates[templatei]
            tx = arange(len(temp))+1
            #r = int(round(pixshift))
            
            shifted = interp(sx-pixshift+1,tx,temp) #TODO:why +1 ?
            
            #shifted = roll(temp,r)
            
#Rendered unnecessary by ivar variable            
#            if r > 0:
#            #    shifted[:r]=flux[:r] #do something smarter here for the edges
#                shifted[:r]=shifted[r] 
#            elif r < 0:
#            #    shifted[r:]=flux[r:] #do something smarter here for the edges
#                shifted[r:]=shifted[r] 
            
            
            return A*shifted+offset
        
        elems['modelflux']=modelflux
        
        @pymc.potential
        def pixelcutoff(pixshift=pixshift):
            #TODO:rethink
            lcut=np.exp(-(pixshift-maxlshift)/npix)
            rcut=np.exp((pixshift-maxrshift)/npix)
            return lcut*rcut
        #elems['pixelcutoff']=pixelcutoff
    
#    @pymc.deterministic(trace=True,plot=False)
#    def ivar(pixshift=pixshift,ivararr=ivar,ivar0=ivar0):
#        """
#        The inverse variance is adjusted to be very small for points that are 
#        off the edge
#        """
#        r = int(round(pixshift))
#        if r > 0:
#            ivars = ivararr.copy()
#            ivars[:r] = ivar0
#        elif r < 0:
#            ivars = ivararr.copy()
#            ivars[r:] = ivar0
#        else:
#            ivars = ivararr #leave alone
            
#        return ivars
#    elems['ivar']=ivar
    
    #fluxvar = pymc.Poisson('flux',mu=modelflux,observed=True,value=flux)
    dmask = isfinite(ivar) & (ivar > 0)
    ivar[~dmask]=np.min(ivar[dmask])/1000 #TODO:test
    
    fluxvar = pymc.Normal('flux',mu=modelflux,tau=ivar,observed=True,value=flux)
    elems['fluxvar']=fluxvar
    
    m=pymc.MCMC(elems)
    m.ivar = ivar
    return m

def extract_template(m,speci):
    return m.modelflux.trace[:][speci]

def replotmc(m,xaspec=None,i=-1,close=True):
    import pymc
    
    m.deviance.plot = False
    m.A.plot = False
    
    plt.close('all')
    pymc.Matplot.plot(m)
    if xaspec:
        x,spec=xaspec
        plt.figure()
        plt.plot(x,spec)
        plt.plot(x,extract_template(m,i),'--')
        
def plotspec(m,i=None,x=None,dospec=True,dotemp=True,doerr=True,dotext=True,clf=True):
    if i is not None:
        ps=m.pixshift.trace[:][i]
        mf=m.modelflux.trace[:][i]
        da=m.fluxvar.value
        try:
            err=m.ivar.trace[:][i]**-0.5
        except (AttributeError,TypeError):
            err = m.ivar**-0.5
    else:
        ps=m.pixshift.value
        mf=m.modelflux.value
        da=m.fluxvar.value
        try:
            err=m.ivar.value**-0.5
        except AttributeError:
            err = m.ivar**-0.5
    
    if clf:
        plt.clf()
        
    if x is None:
        x = arange(len(da))
        s = 'pixshift=%i'%ps
    else:
        s = 'pixshift=%i,vel=%.1f'%(ps,pixshift_to_vel(ps,x))
    if dospec:
        plt.plot(x,da,c='b')
    if dotemp:
        plt.plot(x,mf,c='r')
    if doerr:
        plt.plot(x,err,c='g')
    if dotext:
        plt.text(.1,.9,s,transform=plt.gca().transAxes)
    
    plt.xlim(np.min(x),np.max(x))
    plt.ylim(0,1.05)
    plt.xlabel(r'$\lambda/\AA$')
    plt.ylabel(r'${\rm erg} s^{-1} cm^{-2}$')
    
#<-----------------spectrum loading---------------->
def load_zspec_spec(fns):
    """
    generates a sequence of Spectrum objects from the requested zspec-style
    files
    
    fn can be a single file, a list of files, or a unix-style pattern
    """
    import pyfits
    from astropysics.spec import Spectrum
    from operator import isSequenceType
    from contextlib import closing
    
    if isinstance(fns,basestring):
        from glob import glob
        fns = glob(fns)
        
    if not isSequenceType(fns):
        raise ValueError('improper filename format')
    
    specs = []
    
    for fn in fns:
        with closing(pyfits.open(fn)) as f:
            d = f[1].data
            s = Spectrum(d.LAMBDA.ravel(),d.SPEC.ravel(),ivar=d.IVAR.ravel())
            specs.append(s)
            
    return specs

def load_deimos_templates(fns,asdict=True):
    """
    This will generate a dictionary of Spectrum objects from the specified  
    templates 
    fns can be a single file, a list of files, or a unix-style pattern
    """
    import pyfits
    from astropysics.spec import Spectrum
    from operator import isSequenceType
    from warnings import warn
    
    if isinstance(fns,basestring):
        from glob import glob
        fns = glob(fns)
        
    if not isSequenceType(fns):
        raise ValueError('improper filename format')
    
    tempd={}
    xd={}
    for fn in fns:
        f = pyfits.open(fn)
        try:
            h=f[0].header
            d=f[0].data
            if len(d.shape) == 1:
                d = (d,)
            if len(d) == 1:
                if h.has_key('NAME0'):
                    k=h['NAME0'].strip()
                elif h.has_key('NAME'):
                    k=h['NAME'].strip()
                elif h.has_key('OBJECT'):
                    k=h['OBJECT'].strip()
                else:
                    k = fn.strip()
                    warn('could not infer spectrum name - using filename %s as key name'%k)
                    
                xd[k] = 10**(h['COEFF0']+h['COEFF1']*np.arange(d[0].shape[-1]))
                tempd[k] = d[0].copy()
            else:
                x = 10**(h['COEFF0']+h['COEFF1']*np.arange(d.shape[-1]))
                for i,spec in enumerate(d):
                    if  h.has_key('NAME%i'%i):
                        k=h['NAME%i'%i].strip()
                    elif h.has_key('OBJECT') and h.has_key('EIGEN%i'%i):
                        k = '%s%i'%(h['OBJECT'].strip(),h['EIGEN%i'%i])
                    elif h.has_key('NAME'):
                        k = '%s-%i'%(h['NAME'].strip(),i)
                    elif h.has_key('OBJECT'):
                        k = '%s-%i'%(h['OBJECT'].strip(),i)
                    else:
                        k = '%s-%i'%(fn.strip(),i)
                        warn('could not infer spectrum name - using filename %s as key name'%k)
                        
                    xd[k] = x
                    tempd[k]=spec.copy() #TODO: see if its better to not copy here
                    
        finally:
            f.close()
            
    d = dict(((k,Spectrum(xd[k],tempd[k])) for k in tempd))
    for n,s in d.iteritems():
        s.name = n
    if asdict:
        return d
    else:
        return d.values()

def combine_templates(temps,As=None,noise=None,scaling=None,vel=0,unit = 'wl'):
    """
    make a Spectrum from a linear combination of template Spectrum objects
    
    temps should be a sequence of Spectrum objects, and As either None 
    (random weights) or a sequence equal to the length of temps
    
    if scaling is not None, it sets the maximum value of the resultant spectrum
    
    noise can be:
    * None/False/0: no noise
    * 'poisson##.##': poisson noise where ##.## is a factor to multiply the flux 
      by before scaling
    * a float:random noise of given amplitude
    
    returns newspec,As
    """
    from astropysics.spec import Spectrum
    from scipy.interpolate import InterpolatedUnivariateSpline
    
    if As is None:
        As = np.random.rand(len(temps))
    
    As = np.array(As,copy=False)
    if len(As) != len(temps):
        raise ValueError('As does not match templates size')
    
    xs = []
    fs = []
    for temp in temps:
        oldunit = temp.unit
        try:
            temp.unit = unit
            xs.append(temp.x)
            fs.append(temp.flux)
        finally:
            temp.unit = oldunit
    
    x0 = xs[0]
    for i,x in enumerate(xs):
        if np.any(x != x0):
            raise ValueError('Spectrum %i x does not match'%i) 
    
    fx = (np.array(fs)*As.reshape((As.size,1))).sum(axis=0)
    
    err = None
    if noise:
        if isinstance(noise,basestring):
            if 'poisson' in noise:
                pscale = noise.replace('poisson','')
                if pscale.strip() == '':
                    pscale = 1
                else:
                    pscale = float(pscale)
                err = sqrt(pscale*fx)/pscale
                fx = np.random.poisson(pscale*fx)/pscale
            else:
                raise ValueError('noise parameter is an invalid string')
        else:
            noisescale = float(noise)
            fx += noisescale*np.random.randn(*fx.shape)
            err = noisescale
            
    if vel != 0:
        xnew = x0*(1+vel/3e5)
        bad = (x0>xnew.max())|(x0<xnew.min())
        
        s = InterpolatedUnivariateSpline(xnew,fx)
        fx = s(x0)
        fx[bad] = np.mean(fx[~bad])
        
        if err is not None:
            se = InterpolatedUnivariateSpline(xnew,err)
            err = se(x0)
            err[bad] = np.mean(err[~bad])
    
    return Spectrum(x=x0,flux=fx,unit=unit,err=err),As
    
#<-----------------testing--------------------------->



if __name__=='__main__':
    from astropysics.spec import *
#    #Fake spectra from made up models    
#    from sys import argv
#    npix = float(argv[argv.index('npix')+1]) if 'npix' in argv else 4096
#    nlines = float(argv[argv.index('nlines')+1]) if 'nlines' in argv else 10
#    ntemps = float(argv[argv.index('ntemps')+1]) if 'ntemps' in argv else 15
    
#    xl=logspace(log10(5650),log10(8250),2*npix)
#    xllin=linspace(5650,8250,2*npix)
#    midmask=slice(npix/2,3*npix/2)
#    x0=xl[midmask]
#    x0lin=xllin[midmask]
        
#    if 'vartemp' in argv:
#        T=linspace(3000,10000,ntemps)
#        tsl,lines=generate_templates(ntemps,xl,nl=nlines,temps=T)
#        speci=8 if ntemps>=8 else ntemps-1
#    else:
#        T=5800*np.ones(ntemps)
#        tsl,lines=generate_templates(ntemps,xl,nl=nlines,temps=T)
#        speci=0
    
#    ts=[t[midmask] for t in tsl]
        
#    psnr = float(argv[argv.index('psnr')+1]) if 'psnr' in argv else 20
#    vel = float(argv[argv.index('vel')+1])if 'vel' in argv else 200
#    isamps = float(argv[argv.index('samps')+1])if 'samps' in argv else 3
#    if 'A' in argv:
#        spAs = argv[argv.index('A')+1].split(',')
#        if len(spAs) == 1:
#            As = spAs[0]
#        else:
#            As = np.array(spAs,dtype=float)
#    else:
#        As = None
    
#    snrs=[2,5,20,50,100,200,500,1000]
#    if psnr not in snrs:
#        snrs.append(psnr)
#        snrs=sorted(snrs)
#    specfs = []
#    ivars = []
#    for snr in snrs:
#        if As is None:
#            specf,ivar=lines_to_spec(x0,lines[speci],psnr=snr,vel=vel,ivarsamples=isamps)
#        else:
#            if As == 'rand':
#                As = np.random.rand(len(lines))
#                As = As/sum(As)
#            elif As == 'randn':
#                As = np.random.randn(len(lines))
#                As = As/sum(As)
#            specf,ivar=lines_to_spec_multi(x0,lines,As,T,psnr=snr,vel=vel,ivarsamples=isamps)
#        specfs.append((snr,specf))
#        ivars.append((snr,ivar))
#    specfs=dict(specfs)
#    ivars=dict(ivars)
#    specfobjs=dict([(snr,Spectrum(x0,specfs[snr],ivar=ivars[snr])) for snr in specfs.keys()])
    
#    specf,ivar=specfs[psnr],ivars[psnr]
#    specfobj=Spectrum(x0,specf,ivar=ivar)
    
#    corrtemp=ts[speci]
    
#    #(xstar,dstar),(xgal,dgal),(xab,dab) = get_templates()
 
    dtemps = load_deimos_templates('templates/deimos-021507.fits')
    #fix for M8III spiking
    dtemps['M8III'].flux[:820] = np.mean(dtemps['M8III'].flux[820:])
    tss = dtemps.values()
    x0 = tss[0].x
    tfx = array([s.flux for s in tss]).T
    
    Asd = dict([(n,0) for n in dtemps.keys()])
    #Asd['K3III'] = 0.6
    Asd['K0III'] = 1
    #Asd['G8III'] = 0.3
    As = [Asd[n] for n in dtemps.keys()]
    s0,As = combine_templates(dtemps.values(),As,noise='poisson1000',vel=0)
    s,As = combine_templates(dtemps.values(),As,noise='poisson1000',vel=214.18)
    
    