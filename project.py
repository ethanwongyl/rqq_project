from central import *
import numpy as np,njj,warnings,os,sys,time,glob,cProfile
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.visualization import wcsaxes
from astropy.nddata import Cutout2D
from matplotlib import pyplot as plt
import matplotlib
from scipy.optimize import fmin
import scipy.optimize as opt
from scipy.ndimage import measurements,filters,gaussian_filter
from scipy import special
from scipy import ndimage
from scipy import interpolate
from fits_axzap import fits_axzap
import os,time
import sys
from radio_beam import Beam
from reproject import reproject_interp

from matplotlib.patches import Ellipse
matplotlib.rcParams.update({'font.size':12})



fitalpha, LOGSTART, LOGEND, LNUM = True, -7., 1., 300
TABLE1 = np.asarray(np.loadtxt('grid_search_B0850/table1',dtype='S')[:,2:],dtype='float')
FRAT,FERR,ABSEP,ASEP,BSEP,DERR = 6.15,1.0,0.68,0.1,0.1,0.01
dir = 'yesBH'

#Combining B0850 files
#Not to be used in these. Refer to total().
def combine():#(hdul1=fits.open('1130_comb.fits'),hdul2=fits.open('1201_comb.fits'),hdul3=fits.open('1212_comb.fits'),hdul4=fits.open('1214_comb.fits')):
    data1, data2, data3, data4 = hdul1[0].data, hdul2[0].data, hdul3[0].data, hdul4[0].data
    avg = (data1+data2+data3+data4)/4
    avg = avg[0][0]
    rms =np.sqrt(sum(i**2 for i in avg[412:512][:100].flatten())/len(avg[412:512][:100].flatten()))
    print(avg[257][256])
    plt.figure(figsize=(10,10))
    plt.imshow(avg,cmap='Greys',origin='lower',vmin=-5e-4,vmax=1e-3)
    plt.xlim(200,300)
    plt.ylim(200,300)
    #plt.colourbar()
    #plt.contour(avg,levels=[-3*rms,3*rms,6*rms,12*rms,24*rms,48*rms,96*rms,128*rms,256*rms],colors='black',alpha=0.5)
    plt.savefig('combine.png')
    plt.show()    

def plot_0435(file1='0435_CLEAN.fits',file2='HE0435_COMB_IMAGE.fits'):
    os.system('rm temp*.fits')
    fits_axzap (file1, 'temp1.fits')
    fits_axzap (file2, 'temp2.fits')
    hdul = fits.open('temp1.fits')[0]
    hdul2 = fits.open('temp2.fits')[0]

    header = hdul.header
    wcs=WCS(header,naxis=2)
    data=hdul.data
    rms=np.sqrt(sum(i**2 for i in data[:100][:100].flatten())/len(data[:100][:100].flatten()))     
    noise = rms*3

    header2 = hdul2.header
    wcs2 = WCS(header2,naxis=2)
    data2 = hdul2.data
    rms2 = np.sqrt(sum(i**2 for i in data2[:100][:100].flatten())/len(data2[:100][:100].flatten()))
    noise2 = rms2 * 3

    max2 = hdul2.data.max()
    levels2 = (max2/9)*np.sqrt(2)**np.arange(2,9)   #[1*noise2,2*noise2,4*noise2,8*noise2,16*noise2,32*noise2,64*noise2,128*noise2]
    print(hdul.data.min(),hdul.data.std()*7.0)
    print(levels2)

    #GAIA position
    ra = [69.56160351805,69.56230465346,69.56203780095,69.56188488232]
    dec = [-12.28748166882,-12.28731415348,-12.28776263363,-12.28716055854]
    sky = SkyCoord(ra,dec,unit='deg')
    x,y = wcs2.world_to_pixel(sky)
    ra_centre = [((ra[0]+ra[1]+ra[2]+ra[3])/4)]
    dec_centre = [((dec[0]+dec[1]+dec[2]+dec[3])/4)]
    centre_sky = SkyCoord(ra_centre,dec_centre,unit='deg')
    centre = wcs2.world_to_pixel(centre_sky)

    #reprojection
    reproj, footprint = reproject_interp(hdul2, header)

    plt.figure(figsize=(9,9))
    ax = plt.subplot(projection=wcs2)
    #ax.set_autoscale_on(False)
    plt.imshow(data,cmap=matplotlib.cm.Purples_r,vmin=hdul.data.min(),vmax=hdul.data.std()*7.0)
    plt.xlim(centre[0]-125,centre[0]+125)
    plt.ylim(centre[1]-125,centre[1]+125)
    plt.xlabel('Right Ascension (J2000)')
    plt.ylabel('Declination (J2000)')
    plt.contour(data2,levels=levels2,colors='yellow',linewidth=0.5)
    #plt.plot(x,y,'m+',linewidth=0.5,markersize=30)
    

    #Add beam e-MERLIN
    xpix, ypix = 0.95*(centre[0]-200)+0.05*(centre[0]+200), 0.95*(centre[0]-200)+0.05*(centre[0]+200)
    pixscale = 1.0/header['CDELT2']
    c_beam = Beam.from_fits_header(header)
    bmaj,bmin,bpa = c_beam.major.value, c_beam.minor.value, c_beam.pa.value
    ell = Ellipse((xpix,ypix),width=bmin*pixscale,height=bmaj*pixscale,angle=bpa, color='indianred')
    #ax.add_patch(ell)

    #Add beam VLA
    xpix2, ypix2 = 0.9*(centre[0]-200)+0.1*(centre[0]+200), 0.95*(centre[0]-200)+0.05*(centre[0]+200)
    pixscale = 1.0/header['CDELT2']
    c_beam2 = Beam.from_fits_header(header2)
    bmaj2,bmin2,bpa2 = c_beam2.major.value, c_beam2.minor.value, c_beam2.pa.value
    ell2 = Ellipse((xpix2,ypix2),width=bmin2*pixscale,height=bmaj2*pixscale,angle=bpa2, color='black',alpha=0.5)
    #ax.add_patch(ell2)

    plt.show()
    #plt.savefig('figures/HE0435-1223.png',bbox_inches='tight')


def plot_2211(file1='2211_evn.fits',file2='2211_emerlin1.fits',file3='HS2209+1914_nat.fits'):
    #get info print(header[:47])
    os.system('rm temp*.fits')
    fits_axzap (file1, 'temp1.fits')
    fits_axzap (file2, 'temp2.fits')
    fits_axzap (file3, 'temp3.fits')
    hdul1 = fits.open('temp1.fits')[0]
    hdul2 = fits.open('temp2.fits')[0]
    hdul3 = fits.open('temp3.fits')[0]

    header1 = hdul1.header
    wcs1 = WCS(header1,naxis=2)
    data1 = hdul1.data      #dimension removed in fits_axzap
    rms1 = np.sqrt(sum(i**2 for i in data1[:100][:100].flatten())/len(data1[:100][:100].flatten()))
    noise1 = rms1 * 3
    
    header2 = hdul2.header
    wcs2 = WCS(header2,naxis=2)
    data2 = hdul2.data      #dimension removed in fits_axzap
    rms2 = np.sqrt(sum(i**2 for i in data2[:100][:100].flatten())/len(data2[:100][:100].flatten()))
    noise2 = rms2 * 3

    header3 = hdul3.header
    wcs3 = WCS(header3,naxis=2)
    data3 = hdul3.data      #dimension removed in fits_axzap
    rms3 = np.sqrt(sum(i**2 for i in data3[:100][:100].flatten())/len(data3[:100][:100].flatten()))
    noise3 = rms3 * 3

    #GAIA position
    ra = [332.87641136285,332.87631302787]
    dec = [+19.48683478813,+19.48710841581]
    sky = SkyCoord(ra,dec,unit='deg')
    x,y = wcs1.world_to_pixel(sky)
    ra_centre = [(ra[0]+ra[1])/2]
    dec_centre = [(dec[0]+dec[1])/2]
    centre_sky = SkyCoord(ra_centre,dec_centre,unit='deg')
    centre = wcs1.world_to_pixel(centre_sky)

    #contour levels
    max2 = hdul2.data[200:300,200:300].max()
    levels2 = (max2/8)*np.sqrt(2)**np.arange(3,7)   #[1*noise2,2*noise2,4*noise2,8*noise2,16*noise2,32*noise2,64*noise2,128*noise2]
    max3 = hdul3.data.max()    
    levels3 = (max3/8)*np.sqrt(2)**np.arange(7)     #[1*noise3,2*noise3,4*noise3,8*noise3,16*noise3,32*noise3,64*noise3,128*noise3]
    
    #reprojection
    reproj, footprint = reproject_interp(hdul3, header1)

    print(hdul1.data.min()/2,hdul1.data.std()*8.0)
    print(levels2)
    print(levels3)

    plt.figure(figsize=(8,8))
    ax = plt.subplot(projection=wcs1)
    plt.imshow(data1,cmap=matplotlib.cm.Purples_r,vmin=hdul1.data.min()/2,vmax=hdul1.data.std()*8.0)
    plt.contour(data2,levels=levels2,colors='black',alpha=0.7,linewidth=0.5)      
    plt.contour(reproj,levels=levels3,colors='yellow',linewidth=0.5)  
    plt.xlim(centre[0]-50,centre[0]+50)
    plt.ylim(centre[1]-50,centre[1]+50)
    plt.xlabel('Right Ascension (J2000)')
    plt.ylabel('Declination (J2000)')
    #plt.plot(x,y,'m+',linewidth=0.5,markersize=30)

    #beam for e-MERLIN
    xpix, ypix = 0.75*(centre[0]-50)+0.25*(centre[0]+50), 0.9*(centre[0]-50)+0.1*(centre[0]+50)
    pixscale = 1.0/header1['CDELT2']
    c_beam = Beam.from_fits_header(header1)
    bmaj,bmin,bpa = c_beam.major.value, c_beam.minor.value, c_beam.pa.value
    ell = Ellipse((xpix,ypix),width=bmin*pixscale,height=bmaj*pixscale,angle=bpa, color='black')
    #ax.add_patch(ell)  

    #beam for EVN
    xpix2, ypix2 = 0.7*(centre[0]-50)+0.3*(centre[0]+50), 0.9*(centre[0]-50)+0.1*(centre[0]+50)
    pixscale = 1.0/header1['CDELT2']
    c_beam2 = Beam.from_fits_header(header2)
    bmaj2,bmin2,bpa2 = c_beam2.major.value, c_beam2.minor.value, c_beam2.pa.value
    ell2 = Ellipse((xpix2,ypix2),width=bmin2*pixscale,height=bmaj2*pixscale,angle=bpa2, color='indianred',alpha=0.5)
    #ax.add_patch(ell2)

    #beam for VLA
    xpix3, ypix3 = 0.88*(centre[0]-50)+0.12*(centre[0]+50), 0.9*(centre[0]-50)+0.1*(centre[0]+50)
    pixscale = 1.0/header1['CDELT2']
    c_beam3 = Beam.from_fits_header(header3)
    bmaj3,bmin3,bpa3 = c_beam3.major.value, c_beam3.minor.value, c_beam3.pa.value
    ell3 = Ellipse((xpix3,ypix3),width=bmin3*pixscale,height=bmaj3*pixscale,angle=bpa3, color='blue',alpha=0.4)
    #ax.add_patch(ell3)

    #plt.show()
    plt.savefig('figures/HS2209+1914.png',bbox_inches='tight')    


#Mapping of B0850 CLEANed .fits file
#Next step is to make contour map of B0850
def total(hdul=fits.open('0850_clean.fits')):
    lim=[150,350]
    header = hdul[0].header
    wcs=WCS(hdul[0].header)
    data=hdul[0].data
    data=data[0][0]
    rms=np.sqrt(sum(i**2 for i in data[:100][:100].flatten())/len(data[:100][:100].flatten()))     
    print(rms)
    noise = rms*3

    plt.figure(figsize=(8,8))
    ax = plt.subplot(projection=wcs,slices=('x','y',0,0))
    ax.set_autoscale_on(False)
    #plt.imshow(data,cmap='Greys',origin='lower',vmin=-5e-4,vmax=1e-3)
    plt.xlim(lim[0],lim[1])
    plt.ylim(lim[0],lim[1])
    plt.xlabel('Right Ascension (J2000)')
    plt.ylabel('Declination (J2000)')
    #plt.contour(data,levels=[-3*noise,3*noise,6*noise,12*noise,24*noise,48*noise,96*noise,128*noise,256*noise],colors='black',alpha=0.5)
    plt.contour(data,levels=[-1*noise,1*noise,2*noise,4*noise,8*noise,16*noise,32*noise,64*noise,128*noise],colors='black',alpha=0.5)

    #Add beam
    xpix, ypix = 0.95*lim[0]+0.05*lim[1], 0.95*lim[0]+0.05*lim[1]
    pixscale = 1.0/header['CDELT2']
    c_beam = Beam.from_fits_header(header)
    bmaj,bmin,bpa = c_beam.major.value, c_beam.minor.value, c_beam.pa.value
    ell = Ellipse((xpix,ypix),width=bmin*pixscale,height=bmaj*pixscale,angle=bpa, color='indianred')
    ax.add_patch(ell)

    print(bmaj,bmin,bpa)

    #label components A and B
    plt.text(240,260,'A',fontsize='x-large')
    plt.text(230,210,'B',fontsize='x-large')

    #Tried transform from fk5 but failed

    #sky = wcs.pixel_to_world(xpix,ypix,0,0)
    #ra,dec = sky[0].ra, sky[0].dec
    #c_beam = Beam.from_fits_header(header)
    #ell = Ellipse((ra,dec),width=c_beam.major,height=c_beam.minor,angle=c_beam.pa,transform=ax.get_transform('fk5'))

    plt.savefig('figures/0850_map.png')

#2-D gaussian model
def twoD_gaussian(xy,a,x0,y0,x_err,y_err,theta,off):
    x,y =xy
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta)**2)/(2*x_err**2) + (np.sin(theta)**2)/(2*y_err**2)
    b = -(np.sin(2*theta))/(4*x_err**2) + (np.sin(2*theta))/(4*y_err**2)
    c = (np.sin(theta)**2)/(2*x_err**2) + (np.cos(theta)**2)/(2*y_err**2)
    g = off + a*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                            + c*((y-y0)**2)))
    return g.ravel()

#gaussian model fit to components
def data_fit2D(hdul=fits.open('0850_clean.fits')):
    data=hdul[0].data
    data=data[0][0]
    x=np.linspace(0,data.shape[0]-1,data.shape[0])
    y=np.linspace(0,data.shape[1]-1,data.shape[1])
    x,y=np.meshgrid(x,y)
    initial_guess = (np.max(data),256,256,0,0,23,10)
    popt, pcov = opt.curve_fit(twoD_gaussian,(x,y),data.ravel(),p0=initial_guess)
    data_fitted= twoD_gaussian((x,y),*popt)

    fig,ax=plt.subplots(1,1)
    #ax.imshow(data)
    #ax.contour(x,y,data_fitted.reshape(512,512),8,colors='w')
    ax.imshow(data_fitted.reshape(512,512))
    plt.show()

def loop_match(gid=len(TABLE1),doplot=False,dorat=True,dosep=True):
    loop = np.array([]), np.array([])
    n = 0
    good=[]
    f = open('match_yesBH','w')
    for i in range(gid):
        match = match_ratio(i,True,doplot=doplot, dorat=dorat,dosep=dosep)
        if len(match[1])==3:
            for cmpt in range(3):
                f.write('%d %6.3f %6.3f %6.3f %6.3f %6.3f %10.5f \n' %(i,match[0],match[1][cmpt][0],match[1][cmpt][1],match[1][cmpt][2],match[1][cmpt][3],match[-1]))
        if match[-1] < 5:
            n+=1
            good.append(i)
        f.write('\n')
    f.write('good fit %d \n' %(n))
    for i in good:
        f.write('%d %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %d \n' %(TABLE1[i][0],TABLE1[i][1],TABLE1[i][2],TABLE1[i][3],TABLE1[i][4],TABLE1[i][5],TABLE1[i][6],TABLE1[i][7],TABLE1[i][8],TABLE1[i][9],TABLE1[i][10],TABLE1[i][11]))
            #f.write(str(i)+' '+str(match[0])+' '+str(match[1][cmpt])+' '+str(match[-1])+'\n')
#        f.write(str(i)+' '+''.join(str(x) for x in match)+'\n')
    f.close()

def einst_radius(einst=TABLE1[:,6]):
    n = 0
    for i in range(len(einst)):
        if pow(10,einst[i])>=0.68:
            n+=1
        print(pow(10,einst[i]))    
    return n



def plot_histogram(file=TABLE1[:,:6]):
    fig, axs = plt.subplots(3,2)
    axs[0,0].hist(TABLE1[:,0])
    axs[0,0].axvline(200,color='black')
    axs[0,0].set_title('Velocity Dispersion \u03C3 (km/s)')
    axs[0,1].hist(TABLE1[:,1])
    axs[0,1].axvline(2.78,color='black')#, label='M87 parameter values')
    axs[0,1].set_title('Sharpness of break \u03B1')
    axs[1,0].hist(TABLE1[:,2])
    axs[1,0].axvline(1.33,color='black')
    axs[1,0].set_title('Outer power-law index \u03B2')
    axs[1,1].hist(TABLE1[:,3])
    axs[1,1].axvline(0.14,color='black')
    axs[1,1].set_title('Inner power-law index \u03B3')
    axs[2,0].hist(TABLE1[:,4])
    axs[2,0].axvline(2.24,color='black')
    axs[2,0].set_title('Convergence $\u03BA_b$')
    axs[2,1].hist(TABLE1[:,5])
    axs[2,1].axvline(-2.63,color='black')
    axs[2,1].set_title('Break radius $r_b$')
    fig.tight_layout()
    plt.savefig('figures/keeton.png')
    #plt.show()

def plot_ratio_gamma_rb(logfile='gamma_rb.log',savefig='gamma_rb.png',glim=[0.0,0.025,40],rblim=[-3.,0.05,60],\
                        ls='solid',lw=1,mu=1/2250,sig=0,alpha=2,alphafit='fitted',plotall=True,subplot=321):
    glabel = np.linspace(glim[0],glim[0]+glim[1]*glim[2],glim[2]//20+1)
    rblabel = np.linspace(rblim[0],rblim[0]+rblim[1]*rblim[2],rblim[2]//20+1)
    gticks = np.linspace(0,glim[2],glim[2]//20+1)
    rbticks = np.linspace(0,rblim[2],rblim[2]//20+1)
    #xx,yy = np.meshgrid(g,rb)
    data = np.loadtxt(logfile)
    ratio = data[:,-2].reshape((rblim[2],glim[2]))
    if plotall:
        plt.subplot(subplot)
    else:
        plt.figure(figsize=(12,10))
    plt.imshow(ratio)
    plt.xticks(gticks,glabel)
    plt.yticks(rbticks,rblabel)
    plt.colorbar()
    plt.xlabel('Inner power law index')
    plt.ylabel('log(Break radius/arcsec)')
    CS = plt.contour(ratio,levels=[np.log10(mu)],linewidths=2.0,colors='Red',ls='solid')

    fmt={}
    strs = [2250]
    for l,s in zip(CS.levels,strs):
        fmt[l]=s
    #plt.clabel(CS,fmt=fmt,inline=True)
    plt.title('\u03C3='+str(sig)+' km/s, \u03B1='+str(alpha)+', \u03B1 '+alphafit)
    #plt.savefig(savefig)
    return None

def plot_all_subplot():
    #folders=['BH0_noalpha','BH0_yesalpha','BH100_noalpha','BH100_yesalpha','BH200_noalpha','BH200_yesalpha']
    #folders=['BH0_noalpha3','BH0_yesalpha3','BH100_noalpha3','BH100_yesalpha3','BH200_noalpha3','BH200_yesalpha3']
    #fig,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2)
    plt.figure(figsize=(10,8))
    plot_ratio_gamma_rb(logfile='BH0_noalpha/gamma_rb.log',savefig='figures/BH0_noalpha.png',sig=0,alpha=2,alphafit='not fitted',subplot=321)
    plot_ratio_gamma_rb(logfile='BH0_yesalpha/gamma_rb.log',savefig='figures/BH0_yesalpha.png',sig=0,alpha=2,alphafit='fitted',subplot=322)
    plot_ratio_gamma_rb(logfile='BH100_noalpha/gamma_rb.log',savefig='figures/BH100_noalpha.png',sig=100,alpha=2,alphafit='not fitted',subplot=323)
    plot_ratio_gamma_rb(logfile='BH100_yesalpha/gamma_rb.log',savefig='figures/BH100_yesalpha.png',sig=100,alpha=2,alphafit='fitted',subplot=324)
    plot_ratio_gamma_rb(logfile='BH200_noalpha/gamma_rb.log',savefig='figures/BH200_noalpha.png',sig=200,alpha=2,alphafit='not fitted',subplot=325)
    plot_ratio_gamma_rb(logfile='BH200_yesalpha/gamma_rb.log',savefig='figures/BH200_yesalpha.png',sig=200,alpha=2,alphafit='fitted',subplot=326)
    plt.tight_layout()
    #plt.show()
    plt.savefig('figures/alpha2_all.png')

def plot_all():
    folders=['BH0_noalpha','BH0_yesalpha','BH100_noalpha','BH100_yesalpha','BH200_noalpha','BH200_yesalpha',\
             'BH0_noalpha3','BH0_yesalpha3','BH100_noalpha3','BH100_yesalpha3','BH200_noalpha3','BH200_yesalpha3']
    for i in folders:
        plot_ratio_gamma_rb(logfile=i+'/gamma_rb.log',savefig='figures/'+i)
#def plot_twofig():
#    plt.subplot(121)
#    plot_ratio_gamma_rb(logfile='noBH/gamma_rb.log',ls='dashed')
#    plot_ratio_gamma_rb(logfile='yesBH/gamma_rb.log',ls='solid')
    
    
    
    
    
    

#def findim(s,gal,bh,r,defl,ddefl,logstart,logend):
#    t,im = defl-r-s, np.array([])
#    for i in range(len(r)-1):
#        if (t[i]<0.0 and t[i+1]>0.0) or (t[i]>0.0 and t[i+1]<0.0):
#            x = abs(t[i])/(abs(t[i])+abs(t[i+1]))
#            rret = r[i]+(r[i+1]-r[i])*x
#            dret = defl[i]+(defl[i+1]-defl[i])*x
#            ddret = ddefl[i]+(ddefl[i+1]-ddefl[i])*x
#            mu = 1./((1.-dret/rret)*(1.-ddret))
#            new = np.array([rret,dret ,ddret,mu])
#            try:
#                im = np.vstack((im,new))
#            except:
#                im = np.array([np.copy(new)])
#    return im,r,defl,ddefl


#def setup(logstart,logend,n,gal,bh,s,stype='nuker'):
#    r = np.logspace(logstart,logend,n)  
#    defl = defr_nuker(gal,r,bh)
#    r = np.append(-r[::-1],r)
#    defl = np.append(-defl[::-1],defl)
#    ddefl = np.gradient(defl) / np.gradient(r)
#    return findim(s,gal,bh,r,defl,ddefl,logstart,logend)
       
#def defr_nuker(r,gal,bh):
#    (sigma,alpha,beta,gamma,kb,rb)=gal
#    a,b = (2.-gamma)/alpha, (beta-gamma)/alpha
#    rb,kb = 10.**rb, 10.**kb
#    out = 2.**(1.+b)*kb*rb*(r/rb)**(1.-gamma)/(2.-gamma) * \
#          special.hyp2f1(a,b,1.+a,-(r/rb)**alpha)  \
#          #+ bh*0.686e-12*sigma**3.75/r
#    return out


#def func_opt_gamma_rb(x0,*x):
#    global fitalpha
#    if fitalpha:
#        sigma,beta,gamma,rb = x
#        alpha,s,kappa = x0
#    else:
#        sigma,alpha,beta,gamma,rb = x
#        s,kappa = x0
#    if alpha>15.0:
#        return 1.0e+9
#    gal = [sigma,alpha,beta,gamma,kb,rb]
#    im,r,defl,ddefl = setup(LOGSTART,LOGEND,LNUM,gal,False,s)
#    if len(im)>2:
#        mu = np.sort(abs(im[:,-1]))
#        fAB = np.log10(mu[-1]/mu[-2])
#        sepAB = abs(im[-1,0]-im[0,0])
#        sepA,sepB = abs(im[0,0]),abs(im[-1,0])
#        goodness = 100.*((np.log10(FRAT)-fAB)**2 + (ABSEP-sepAB)**2/DERR**2)
#    else:
#        goodness = 1.0e+9
#    return goodness

#def opt_gamma_rb(gamma,rb,alpha=2.0,sigma=0.0,beta=1.0):
#    global fitalpha
#    s,kappa = plot_gamma_rb(gamma,rb,alpha,sigma,beta,False)
#    if fitalpha:
#        x = (sigma,beta,gamma,rb)
#        x0 = np.array([alpha,s,kappa])
#        xopt = fmin(func_opt_gamma_rb,x0,args=x,ftol=0.0001,xtol=0.0001)
#        x = (sigma,beta,gamma,rb)
#    else:
#        x = (sigma,alpha,beta,gamma,rb)
#        x0 = np.array([s,kappa])
#        xopt = fmin(func_opt_gamma_rb,x0,args=x,ftol=0.0001,xtol=0.0001)
#        x = (sigma,alpha,beta,gamma,rb)
#    goodness = func_opt_gamma_rb(xopt,*x)
#    alpha = xopt[0] if fitalpha else alpha
#    s = xopt[1] if fitalpha else xopt[0]
#    kappa = xopt[2] if fitalpha else xopt[1]
#    gal = [sigma,alpha,beta,gamma,kb,rb]
#    im,r,defl,ddefl = setup(LOGSTART,LOGEND,LNUM,gal,False,s)
#    if len(im)>2 and goodness<5.:
#        mu = np.sort(abs(im[:,-1]))
#        nplot = len(glob.glob('gamma_rb*.png'))
#        fn = 'gamma_rb_%04d.png'%nplot
#        mkplot(-1,s,True,[sigma,alpha,beta,gamma,kappa,rb],noisy=True,\
#               fname=fn,x1=-1.5,x2=1.5)
#        f = open('gamma_rb.log','a')
#        f.write('%.3f %.2f %.3f %.3f %.3f %.3f %.3f  %.1f %.3f %.4f\n' % \
#                 (s,sigma,alpha,beta,gamma,kappa,rb,mu[-1]/mu[-2],\
#                   np.log10(mu[-3]/mu[-1]),goodness))
#        f.close()
#        return np.log10(mu[-3]/mu[-1])
#    else:
#        f = open('gamma_rb.log','a')
#        f.write('%.3f %.2f %.3f %.3f %.3f %.3f %.3f  NaN NaN  %.4f\n' % \
#                  (s,sigma,alpha,beta,gamma,kappa,rb,goodness))
#        f.close()
#        return np.nan

#def plot_gamma_rb(gamma,rb,alpha=2.0,sigma=0.0,beta=1.0,doplot=True):
