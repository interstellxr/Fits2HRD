import os
from os import listdir
from astropy.visualization import simple_norm
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
#from astropy.io.fits import writeto
import astropy.units as u
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.aperture import aperture_photometry
from photutils.detection import DAOStarFinder
#from photutils.detection import IRAFStarFinder
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources
from photutils.background import Background2D, MedianBackground, SExtractorBackground
from photutils.utils import circular_footprint
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import ApertureStats
import cv2
from astropy.wcs import WCS
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from photutils.psf import IntegratedGaussianPRF
from photutils.psf import PSFPhotometry
from photutils.background import LocalBackground
from photutils.psf import IterativePSFPhotometry
from astropy.table import QTable
from astropy.coordinates import match_coordinates_sky

plt.rcParams['text.usetex'] = True

path = "C:/Users/thums/Pictures/Astro images/Heart Nebula 20-08-23/LIGHT/"
os.chdir(path)

Vizier.clear_cache()
Vizier.TIMEOUT = 120 #set timeout to 120s (60s by default)

#%% Calculations

table, table_r, table_g, table_b = [],[],[],[]
for index, images in enumerate(os.listdir(path)):

    if (images.endswith("0061.fits")): #images you want to use

        with fits.open(images) as img:

            data = img[0].data
            data = np.array(data)
            data = (data / np.max(data) * 65535).astype(np.uint16)
            
            
            #Debayering
            debayered_image = cv2.cvtColor(data, cv2.COLOR_BayerRGGB2RGB)
            
            b=debayered_image[:,:,0]
            g=debayered_image[:,:,1]
            r=debayered_image[:,:,2]
            
            
            #GLobal background substraction for all channels
            sigma_clip = SigmaClip(sigma=3.0, maxiters=10) 
            footprint = circular_footprint(radius=10)
            threshold, segment_img, mask, mean, median, std =[],[],[],[],[],[]
            bkg_estimator = MedianBackground()
            
            for i, channel in enumerate([data, r, g, b]):
            
                threshold.append(detect_threshold(channel, nsigma=2.0, sigma_clip=sigma_clip))
                segment_img.append(detect_sources(channel, threshold[i], npixels=10))
                mask.append(segment_img[i].make_source_mask(footprint=footprint))
                
                bkg = Background2D(channel, (50, 50), filter_size=(3, 3),mask=mask[i], sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
                
                #channel = channel - bkg.background #substract background
                
                mean.append(sigma_clipped_stats(channel, sigma=3.0,mask=mask[i])[0])
                median.append(sigma_clipped_stats(channel, sigma=3.0,mask=mask[i])[1])
                std.append(sigma_clipped_stats(channel, sigma=3.0,mask=mask[i])[2])
                
            
            #Source (star) detection
            daofind = DAOStarFinder(fwhm=5, threshold=10.*std[0])
            sources = daofind(data)   
            sources_r = daofind(r)
            sources_g = daofind(g)
            sources_b = daofind(b)
            
            for col in sources.colnames:  
                if col not in ('id', 'npix'):
                    sources[col].info.format = '%.2f'  #for consistent table output
            
            x = sources[1][:]
            y = sources[2][:]
    
            
            #Astrometry : find sky coordinates and convert sources
            wcs_header = img[0].header
            wcs = WCS(wcs_header)
            sky_coords = wcs.pixel_to_world(x,y)
            
            #Query Vizier catalog and obtain reference magnitudes
            result=Vizier.query_region(SkyCoord(sky_coords.ra, sky_coords.dec, frame='fk5'), radius=0.01*u.deg, catalog='APASS')
            
            ref_mag_V = result[0]['Vmag']
            ref_mag_B = result[0]['Bmag']
            ref_B_V = result[0]['B-V']
            
            ra_vizier = result[0]['RAJ2000']
            dec_vizier =result[0]['DEJ2000']
            
            coords_vizier = SkyCoord(ra=ra_vizier, dec=dec_vizier,unit=u.deg, frame='fk5')
            x_pix, y_pix = wcs.world_to_pixel(coords_vizier)
            positions_vizier = list(zip(x_pix, y_pix))
            
            #match sources and catalog
            #idx, d2d, _ = sky_coords.match_to_catalog_sky(coords_vizier)
            idx,d2d,_ = match_coordinates_sky(matchcoord=sky_coords, catalogcoord=coords_vizier)
            
            matched_sources = sky_coords[idx]
            x_match,y_match = wcs.world_to_pixel(matched_sources)
            
            positions_match = list(zip(x_match,y_match))
            
            #PSF photometry
            fit_shape = (5, 5)
            psf_model = IntegratedGaussianPRF()
            
            #local background extraction using annuli
            Local_bkg = LocalBackground(inner_radius = 10, outer_radius = 20)
            
            #Find common stars between all channels
            common_sources = min([sources, sources_r, sources_g, sources_b], key=len)
            
            stars_ = list(zip(common_sources['xcentroid'],common_sources['ycentroid']))
            stars = QTable(rows=stars_, names=('x','y'))
            
            stars_match = QTable(rows=positions_match, names=('x','y'))
            
            
            #psfphot = PSFPhotometry(psf_model, fit_shape, finder=daofind, aperture_radius=6, localbkg_estimator=Local_bkg)
            #psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=daofind, aperture_radius=6, localbkg_estimator=Local_bkg)
            psfphot = PSFPhotometry(psf_model, fit_shape, aperture_radius=6, localbkg_estimator=Local_bkg)
            
            phot = []  
            
            for i, channel in enumerate([data, r, g, b]):
            
                phot.append(psfphot(channel, init_params=stars_match)) #force use common stars
            
            table.append(phot[0])
            table_r.append(phot[1])
            table_g.append(phot[2])
            table_b.append(phot[3])
            
            #todo : - compare to catalog data
            # - build an effective PSF for more accuracy
            # - PSF matching ?


#%% Plots

#Plot data, fits and residual data
resid = psfphot.make_residual_image(data, (5, 5))

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
norm = simple_norm(data, 'sqrt', percent=99)
ax[0].imshow(data, origin='lower', norm=norm)
ax[1].imshow(data - resid, origin='lower', norm=norm)
im = ax[2].imshow(resid, origin='lower')
ax[0].set_title('Data')
ax[1].set_title('Model')
ax[2].set_title('Residual Image')
plt.tight_layout()


#Calculate apparent magnitudes 
mag, mag_r, mag_g, mag_b = [],[],[],[]

for t in table :
    mag.append(-2.5*np.log10(t['flux_fit'])) 
for t in table_r :
    mag_r.append(-2.5*np.log10(t['flux_fit']))
for t in table_g :
    mag_g.append(-2.5*np.log10(t['flux_fit']))
for t in table_b :
    mag_b.append(-2.5*np.log10(t['flux_fit']))

mag, mag_r, mag_g, mag_b = np.array(mag), np.array(mag_r), np.array(mag_g), np.array(mag_b)

valid_indices = ~np.isnan(mag_b)
mag_b = mag_b[valid_indices]
mag = mag[valid_indices]
mag_r = mag_r[valid_indices]
mag_g = mag_g[valid_indices]

valid_indices_g = ~np.isnan(mag_g)
mag_g = mag_g[valid_indices_g]
mag = mag[valid_indices_g]
mag_r = mag_r[valid_indices_g]
mag_b = mag_b[valid_indices_g]

valid_indices_r = ~np.isnan(mag_r)
mag_r = mag_r[valid_indices_r]
mag = mag[valid_indices_r]
mag_g = mag_g[valid_indices_r]
mag_b = mag_b[valid_indices_r]


#Calculate color indexes and temperatures
B_V = np.array([mag_b - mag for mag_b, mag in zip(mag_b, mag)])
G_V = np.array([mag_g - mag for mag_g, mag in zip(mag_g, mag)])
R_V = np.array([mag_r - mag for mag_r, mag in zip(mag_r, mag)])

T_eff_list = []
for bv in B_V:
    # Ballesteros' empirical formula
    T_eff = 4600 * ((1 / (0.92 * bv + 1.7)) + (1 / (0.92 * bv + 0.62)))
    T_eff_list.append(T_eff)

T_eff_list = np.array(T_eff_list)


#Experimental color-magnitude and HR diagrams
cmap = plt.get_cmap('RdBu_r')
norm = plt.Normalize(vmin=np.min(B_V), vmax=np.max(B_V))
norm2 = plt.Normalize(vmin=np.min(G_V), vmax=np.max(G_V))
norm3 = plt.Normalize(vmin=np.min(R_V), vmax=np.max(R_V))

plt.figure(figsize=(8, 6))
plt.scatter(B_V, mag,c=B_V,norm=norm, s=5, cmap=cmap, alpha=1)
cbar = plt.colorbar()
cbar.set_label(r'$B-V$ Color Index', fontsize=15)
plt.xlabel(r'$B-V$', fontsize = 15)
plt.ylabel(r'V', fontsize = 15)
plt.title(r'Color-Magnitude Diagram', fontsize = 15)
plt.gca().invert_yaxis()  # Invert y-axis (brighter stars at the top)
plt.grid(True)
plt.show()
plt.tick_params(axis='both', labelsize=13, direction="in")

plt.figure(figsize=(8, 6))
plt.scatter(np.log10(T_eff_list), mag,c=B_V,norm=norm, s=5, cmap=cmap, alpha=1)
cbar = plt.colorbar()
cbar.set_label(r'$B-V$ Color Index', fontsize=15)
plt.xlabel(r'$\log(T_{eff})$', fontsize = 15)
plt.ylabel(r'V', fontsize = 15)
plt.title(r'HR Diagram', fontsize = 15)
plt.gca().invert_yaxis() 
plt.gca().invert_xaxis() #since log10()
plt.grid(True)
plt.show()
plt.tick_params(axis='both', labelsize=13, direction="in")

plt.figure(figsize=(8, 6))
plt.scatter(G_V, mag,c=G_V,norm=norm2, s=5, cmap=cmap, alpha=1)
cbar = plt.colorbar()
cbar.set_label(r'$G-V$ Color Index', fontsize=15)
plt.xlabel(r'$G-V$', fontsize = 15)
plt.ylabel(r'V', fontsize = 15)
plt.title(r'Color-Magnitude Diagram', fontsize = 15)
plt.gca().invert_yaxis()  # Invert y-axis (brighter stars at the top)
plt.grid(True)
plt.show()
plt.tick_params(axis='both', labelsize=13, direction="in")

plt.figure(figsize=(8, 6))
plt.scatter(R_V, mag,c=R_V,norm=norm3, s=5, cmap=cmap, alpha=1)
cbar = plt.colorbar()
cbar.set_label(r'$R-V$ Color Index', fontsize=15)
plt.xlabel(r'$R-V$', fontsize = 15)
plt.ylabel(r'V', fontsize = 15)
plt.title(r'Color-Magnitude Diagram', fontsize = 15)
plt.gca().invert_yaxis()  # Invert y-axis (brighter stars at the top)
plt.grid(True)
plt.show()
plt.tick_params(axis='both', labelsize=13, direction="in")


#Compare data to theory
ref_B_V = np.array(ref_B_V)
ref_mag_V = np.array(ref_mag_V)

plt.figure(figsize=(8, 6))
plt.scatter(ref_B_V, ref_mag_V,color='black',s=5)
plt.xlabel(r'$(B-V)_{th}$', fontsize = 15)
plt.ylabel(r'$V_{th}$', fontsize = 15)
plt.title(r'Reference color-magnitude Diagram', fontsize = 15)
plt.gca().invert_yaxis()  # Invert y-axis (brighter stars at the top)
plt.grid(True)
plt.show()
plt.tick_params(axis='both', labelsize=13, direction="in")

plt.figure(figsize=(8, 6))
plt.scatter(B_V, ref_B_V,s=5, color='blue')
plt.xlabel(r'$(B-V)_{exp}$', fontsize = 15)
plt.ylabel(r'$(B-V)_{th}$', fontsize = 15)
plt.grid(True)
plt.show()
plt.tick_params(axis='both', labelsize=13, direction="in")

plt.figure(figsize=(8, 6))
plt.scatter(mag, ref_mag_V,s=5, color='blue')
plt.xlabel(r'$V_{exp}$', fontsize = 15)
plt.ylabel(r'$V_{th}$', fontsize = 15)
plt.grid(True)
plt.show()
plt.tick_params(axis='both', labelsize=13, direction="in")
