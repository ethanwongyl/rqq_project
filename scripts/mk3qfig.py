import astropy,matplotlib,numpy as np,sys,os,glob,scipy,warnings
import reproject; from reproject import reproject_interp
from astropy.io import fits
from astropy.io.fits import getdata,getheader
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from fits_axzap import fits_axzap
from matplotlib.patches import Ellipse
matplotlib.rcParams.update({'font.size':12})
warnings.filterwarnings('ignore')

def thisplot (file1, file2, pngfile='qfig.png'):
    os.system('rm temp*.fits')
    fits_axzap (file1, 'temp1.fits')
    fits_axzap (file2, 'temp2.fits')
    hdu1 = fits.open('temp1.fits')[0]
    hdu2 = fits.open('temp2.fits')[0]
    h = hdu1.header
    try:
        bmaj,bmin,bpa = 3600.*h['BMAJ'],3600.*h['BMIN'],h['BPA']+90
    except:
        for i in range(len(h)):
            if isinstance(h[i],str) and 'CLEAN BMAJ' in h[i]:
                hl = h[i].split()
                bmaj,bmin,bpa = 3600.*float(hl[3]),3600.*float(hl[5]),float(hl[7])+90
    
    rsiz, osiz = hdu1.data.shape[0], hdu2.data.shape[0]
    max2 = hdu2.data[460:560,460:560].max()
    levels = (max2/8)*np.sqrt(2)**np.arange(7)
    wcs = WCS(hdu2.header)
    hdu2r, footprint = reproject_interp(hdu2, h)
    wcs = WCS(hdu1.header)
    plt.subplot(projection=wcs)
    plt.imshow(hdu1.data,cmap=matplotlib.cm.Purples_r,vmin=hdu1.data.min(),vmax=hdu1.data.std()*7.0)
    plt.contour(hdu2r,levels=levels,colors='yellow',linewidths=0.5)
    plt.xlabel('Right ascension')
    plt.ylabel('Declination')
    plt.savefig(pngfile,bbox_inches='tight')

thisplot('0435_CLEAN.fits','HE0435_COMB_IMAGE.fits','HE0435-1223.png')
thisplot('2211_emerlin1.fits','HS2209+1914_nat.fits','HS2209+1914.png')
    
