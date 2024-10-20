import os
from os import listdir
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

plt.rcParams['text.usetex'] = True

path = "C:/Users/thums/Pictures/Astro images/Crocodile eye galaxy 11-05-2024/LIGHT"
os.chdir(path)

Vizier.clear_cache()
Vizier.TIMEOUT = 120 #set timeout to 120s (60s by default)

#%% Calculations

table, table_r, table_g, table_b = [],[],[],[]
for index, images in enumerate(os.listdir(path)):

    if (images.endswith("0015.fits")): #images you want to use

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
                
                channel = channel - bkg.background #substract background
                
                mean.append(sigma_clipped_stats(channel, sigma=3.0,mask=mask[i])[0])
                median.append(sigma_clipped_stats(channel, sigma=3.0,mask=mask[i])[1])
                std.append(sigma_clipped_stats(channel, sigma=3.0,mask=mask[i])[2])
                
            
            #Source (star) detection
            daofind = DAOStarFinder(fwhm=5, threshold=10.*std[0])
            sources = daofind(data)  
            
            for col in sources.colnames:  
                if col not in ('id', 'npix'):
                    sources[col].info.format = '%.2f'  #for consistent table output
            
            x = sources[1][:]
            y = sources[2][:]
            
            
            #Astrometry : find sky coordinates
            wcs_header = img[0].header
            wcs = WCS(wcs_header)
            sky_coords = wcs.pixel_to_world(x,y)
            
            #Query Vizier catalog and obtain reference magnitudes
            result=Vizier.query_region(SkyCoord(sky_coords.rad, sky_coords.dec, frame='fk5'), radius=0.01*u.deg, catalog='APASS')
            
            ref_mag_V = result[0]['Vmag']
            ref_mag_B = result[0]['Bmag']
            ref_B_V = result[0]['B-V']
            
            ra_vizier = result[0]['RAJ2000']
            dec_vizier =result[0]['DEJ2000']
            
            
            #Create apertures and annuli
            coords_vizier = SkyCoord(ra=ra_vizier, dec=dec_vizier,unit=u.deg, frame='fk5')
            x_pix, y_pix = wcs.world_to_pixel(coords_vizier)
            positions_vizier = list(zip(x_pix, y_pix))
            
            positions = list(zip(x,y))
            
            R = 6.0

            apertures = CircularAperture(positions_vizier, R)
            annuli = CircularAnnulus(positions_vizier, r_in=10, r_out=20)
            
            
            #Photometry and local mean background substraction (w/ masks)
            aperstats, bkg_mean, apertures_area, total_bkg, phot = [],[],[],[],[]   
            
            for i, channel in enumerate([data, r, g, b]):
            
                aperstats.append(ApertureStats(channel, annuli))
                bkg_mean.append(aperstats[i].mean)
                apertures_area.append(apertures.area_overlap(channel,mask=mask[i]))
                total_bkg.append(bkg_mean[i] * apertures_area[i])
                
                phot.append(aperture_photometry(channel, apertures))
                phot_bkgsub = phot[i]['aperture_sum'] - total_bkg[i]
                phot[i]['total_bkg'] = total_bkg[i]
                phot[i]['aperture_sum_bkgsub'] = phot_bkgsub
            
                for col in phot[i].colnames:
                    phot[i][col].info.format = '%.8g'  # for consistent table output
            
            table.append(phot[0])
            table_r.append(phot[1])
            table_g.append(phot[2])
            table_b.append(phot[3])


#%% Plots

#Show apertures on image
plt.figure(figsize=(8, 6))
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data, cmap='gray', origin='lower', norm=norm, interpolation='nearest')
apertures.plot(color='blue', lw=1, alpha=1)


#Calculate apparent magnitudes 
mag, mag_r, mag_g, mag_b = [],[],[],[]
for t in table :
    mag.append(-2.5*np.log10(t['aperture_sum_bkgsub'])) 
for t in table_r :
    mag_r.append(-2.5*np.log10(t['aperture_sum_bkgsub']))
for t in table_g :
    mag_g.append(-2.5*np.log10(t['aperture_sum_bkgsub']))
for t in table_b :
    mag_b.append(-2.5*np.log10(t['aperture_sum_bkgsub']))

for mags in [mag, mag_r, mag_g, mag_b]:
    mags = np.array(mags)


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
plt.ylabel(r'V_{th}', fontsize = 15)
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
