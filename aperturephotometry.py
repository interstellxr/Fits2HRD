import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io.fits import writeto
import astropy.units as u
from astropy.units.photometric import zero_point_flux
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.aperture import aperture_photometry
from photutils.detection import DAOStarFinder
from photutils.detection import IRAFStarFinder
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources
from photutils.background import Background2D, MedianBackground, SExtractorBackground
from photutils.utils import circular_footprint
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import ApertureStats
import rawpy
import cv2
from photutils.psf import PSFPhotometry
from photutils.psf import IntegratedGaussianPRF
from PIL import Image
from skimage import io
from astropy.wcs import WCS
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord


plt.rcParams['text.usetex'] = True

#open fits image
path = "C:/Users/thums/Pictures/Astro images/Heart Nebula 20-08-23/LIGHT/"
#path = "C:/Users/thums/Pictures/Astro images/TIC 271167979/LIGHT/FITS/Calibrated_AIJ/aligned"
#path = "C:/Users/thums/Pictures/Astro images/Cocoon nebula 12-08-23/LIGHT"
#path = "C:/Users/thums/Pictures/Astro images/Bubble nebula 22-08-23/LIGHT"
#path ="C:/Users/thums/Desktop"

os.chdir(path)
table, table_r, table_g, table_b = [],[],[],[]
for index, images in enumerate(os.listdir(path)):

# img = os.listdir(path)[10]
# r = os.listdir(path)[11]
# g = os.listdir(path)[12]
# b = os.listdir(path)[13]

    if (images.endswith("0061.fits")): #index==0:
    
        with fits.open(images) as img:# io.imread(images) as img:

            # data = io.imread(images)
            
            data = img[0].data
            
            # r=fits.open(r)[0].data
            # g=fits.open(g)[0].data
            # b=fits.open(b)[0].data
            data = np.array(data)
            
            #debayer image
            data = (data / np.max(data) * 65535).astype(np.uint16)
            # r = (r / np.max(r) * 65535).astype(np.uint16)
            # g = (g / np.max(g) * 65535).astype(np.uint16)
            # b = (b / np.max(b) * 65535).astype(np.uint16)
            debayered_image = cv2.cvtColor(data, cv2.COLOR_BayerRGGB2RGB)
            #○debayered_image=data
            
            #color channels (cv2 has RGB reversed so BGR order)
            b=debayered_image[:,:,0]
            g=debayered_image[:,:,1]
            r=debayered_image[:,:,2]
            
            #save each channel
            #output_b = 

            #background substraction and star detection
            sigma_clip = SigmaClip(sigma=3.0, maxiters=10)  
            threshold = detect_threshold(data, nsigma=2.0, sigma_clip=sigma_clip)    
            segment_img = detect_sources(data, threshold, npixels=10)  
            thresholdb = detect_threshold(b, nsigma=2.0, sigma_clip=sigma_clip)    
            segment_imgb = detect_sources(b, thresholdb, npixels=10)    
            footprint = circular_footprint(radius=10)           
            mask = segment_img.make_source_mask(footprint=footprint)
            mean, median, std = sigma_clipped_stats(data, sigma=3.0,mask=mask)
            meanb, medianb, stdb = sigma_clipped_stats(b, sigma=3.0,mask=mask)
            
            #sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = SExtractorBackground()
            bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
            bkg_estimatorb = SExtractorBackground()
            bkgb = Background2D(b, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimatorb)
            
            # data=data-bkg.background#remove 2D background
            # b=b-bkgb.background
            
            daofind = DAOStarFinder(fwhm=7, threshold=20.*std)
            irafind = IRAFStarFinder(threshold=20.*std, fwhm=7)
            sources = daofind(data)  
            mag_dao = sources['mag']
        
            
            for col in sources.colnames:  
                if col not in ('id', 'npix'):
                    sources[col].info.format = '%.2f'  # for consistent table output
            
            #sources.pprint(max_width=76) 
            x = sources[1][:]
            y = sources[2][:]
            
            
            #astrometry
            wcs_header = img[0].header
            wcs = WCS(wcs_header)
            sky_coords = wcs.pixel_to_world(x,y)
            
            #VIZIER 
            result=Vizier.query_region(sky_coords, radius=0.1*u.deg, catalog='APASS') 
            ref_mag_V = result[0]['Vmag']
            ref_mag_B = result[0]['Bmag']
            ref_B_V = result[0]['B-V']
            
            
            #SIMBAD
            # simbad = Simbad()
            # ref_catalog = simbad.query_region(sky_coords, radius=1.0*u.deg)

            # identifiers = ref_catalog['MAIN_ID']

            # # Initialize an empty list to store magnitudes
            # magnitudes = []
            
            # # Loop through each identifier to query for magnitudes
        
            # # Query Simbad for each identifier to get additional information including magnitudes
            # query = """SELECT basic.ra, basic.dec, flux.V JOIN TAP_UPLOAD.ident ON basic.main_id=TAP_UPLOAD.ident """
            # star_info = Simbad.query_tap(query, identifier = identifiers)
            # # Extract magnitudes from the returned data
            # magnitude = star_info['FLUX_V'][0]  # Example for V-band magnitudes
            # # Append the magnitude to the list
            # magnitudes.append(magnitude)
            
            
            #create your apertures and annuli
            #ra_simbad = ref_catalog['RA'] # Right ascension in hms
            #dec_simbad = ref_catalog['DEC']  # Declination in degrees
            ra_vizier = result[0]['RAJ2000']
            dec_vizier =result[0]['DEJ2000']
            
            # Create a SkyCoord object representing the celestial coordinates
            #coords_simbad = SkyCoord(ra=ra_simbad, dec=dec_simbad,unit=[u.hourangle,u.deg], frame='fk5')
            coords_vizier = SkyCoord(ra=ra_vizier, dec=dec_vizier,unit=u.deg, frame='fk5')
            
            # Convert celestial coordinates to pixel coordinates
            x_pix, y_pix = wcs.world_to_pixel(coords_vizier)
            #positions = list(zip(x_pix,y_pix))#or np.transpose((x,y)) 
            positions = list(zip(x, y))
            R = 8.0
            apertures = CircularAperture(positions, R)
            annuli = CircularAnnulus(positions, r_in=15, r_out=25)
            
            
            #perform aperture photometry and background substraction
            
            #mean method
            aperstats = ApertureStats(data, annuli)
            bkg_mean = aperstats.mean
            apertures_area = apertures.area_overlap(data,mask=mask)
            total_bkg = bkg_mean * apertures_area
            
            #sigma clipping method
            # sigclip = SigmaClip(sigma=10.0, maxiters=10)
            # aper_stats = ApertureStats(data, apertures, sigma_clip=None)
            # bkg_stats = ApertureStats(data, annuli, sigma_clip=sigclip)
            # total_bkg = bkg_stats.median * aper_stats.sum_aper_area.value
            
            
            phot = aperture_photometry(data, apertures)
            phot_bkgsub = phot['aperture_sum'] - total_bkg
            phot['total_bkg'] = total_bkg
            phot['aperture_sum_bkgsub'] = phot_bkgsub
            
            phot_r = aperture_photometry(r, apertures)
            phot_bkgsub_r = phot_r['aperture_sum'] - total_bkg
            phot_r['total_bkg'] = total_bkg
            phot_r['aperture_sum_bkgsub'] = phot_bkgsub_r
            
            aperstats_g = ApertureStats(g, annuli)
            bkg_mean_g = aperstats_g.mean
            apertures_area_g = apertures.area_overlap(g,mask=mask)
            total_bkg_g = bkg_mean_g * apertures_area_g
            
            # aper_stats_g = ApertureStats(g, apertures, sigma_clip=None)
            # bkg_stats_g = ApertureStats(g, annuli, sigma_clip=sigclip)
            # total_bkg_g = bkg_stats_g.median * aper_stats_g.sum_aper_area.value
            
            phot_g = aperture_photometry(g, apertures)
            phot_bkgsub_g = phot_g['aperture_sum'] - total_bkg_g
            phot_g['total_bkg'] = total_bkg_g
            phot_g['aperture_sum_bkgsub'] = phot_bkgsub_g
            
            aperstats_b = ApertureStats(b, annuli)
            bkg_mean_b = aperstats_b.mean
            apertures_area_b = apertures.area_overlap(b,mask=mask)
            total_bkg_b = bkg_mean_b * apertures_area_b
            
            # aper_stats_b = ApertureStats(b, apertures, sigma_clip=None)
            # bkg_stats_b = ApertureStats(b, annuli, sigma_clip=sigclip)
            # total_bkg_b = bkg_stats_b.median * aper_stats_b.sum_aper_area.value
            
            phot_b = aperture_photometry(b, apertures)
            phot_bkgsub_b = phot_b['aperture_sum'] - total_bkg_b
            phot_b['total_bkg'] = total_bkg_b
            phot_b['aperture_sum_bkgsub'] = phot_bkgsub_b
            
            for col in phot.colnames:
                phot[col].info.format = '%.8g'  # for consistent table output
            for col in phot_r.colnames:
                phot_r[col].info.format = '%.8g'  # for consistent table output
            for col in phot_g.colnames:
                phot_g[col].info.format = '%.8g'  # for consistent table output
            for col in phot_b.colnames:
                phot_b[col].info.format = '%.8g'  # for consistent table output
            
            table.append(phot)
            table_r.append(phot_r)
            table_g.append(phot_g)
            table_b.append(phot_b)
            
            
            #psf photometry
            # psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
            # fit_shape = (5, 5)
            # finder = DAOStarFinder(6.0, 2.0)
            # psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder, aperture_radius=4)
            # psf_phot_data = psfphot(data)
            # psf_phot_data['flux_fit'].info.format = '.4f'
            
            # psfphot_b = PSFPhotometry(psf_model, fit_shape, finder=finder, aperture_radius=4)
            # psf_phot_b = psfphot(b)
            # psf_phot_b['flux_fit'].info.format = '.4f'
        

#plot queried stars from simbad/vizier and compare to daofind
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data - median, cmap='gray', origin='lower', norm=norm, interpolation='nearest')
apertures.plot(color='blue', lw=0.1, alpha=0.5)
CircularAperture(list(zip(x, y)), 10).plot(color='red', lw=0.2, alpha=0.8)
plt.legend(["Vizier stars","Daofind stars"], loc=1)

#print(table[-1])


#apparent magnitudes 
mag, mag_r, mag_g, mag_b = [],[],[],[]
for t in table :
    mag.append(-2.5*np.log10(t['aperture_sum_bkgsub'])) 
for t in table_r :
    mag_r.append(-2.5*np.log10(t['aperture_sum_bkgsub']))
for t in table_g :
    mag_g.append(-2.5*np.log10(t['aperture_sum_bkgsub']))
for t in table_b :
    mag_b.append(-2.5*np.log10(t['aperture_sum_bkgsub']))


# psfmag = -2.5*np.log10(psf_phot_data['flux_fit'])
# psfmag_b = -2.5*np.log10(psf_phot_b['flux_fit'])


#convert lists to arrays
mag=np.array(mag)
mag_b=np.array(mag_b)

# psfmag=np.array(psfmag)
# psfmag_b=np.array(psfmag_b)


#Color index and temperature
B_V = np.array([mag_b - mag for mag_b, mag in zip(mag_b, mag)])
G_V = np.array([mag_g - mag for mag_g, mag in zip(mag_g, mag)])
R_V = np.array([mag_r - mag for mag_r, mag in zip(mag_r, mag)])

#B_V_psf = np.array([psfmag_b - psfmag for psfmag_b, psfmag in zip(psfmag_b, psfmag)])


#offset_color = B_V - ref_B_V  # Compute color offset

# Apply zero-point correction to measured magnitudes
#zero_point_offset = np.mean(mag - ref_mag_V)  # Compute zero-point offset
#calibrated_V = mag - zero_point_offset  # 

T_eff_list = []

for bv in B_V:
    # Ballesteros' empirical formula
    T_eff = 4600 * ((1 / (0.92 * bv + 1.7)) + (1 / (0.92 * bv + 0.62)))
    T_eff_list.append(T_eff)

T_eff_list = np.array(T_eff_list)


#colormaps and figures

cmap = plt.get_cmap('Blues_r')
norm = plt.Normalize(vmin=np.min(B_V), vmax=np.max(B_V))
norm2 = plt.Normalize(vmin=np.min(G_V), vmax=np.max(G_V))
#norm_psf = plt.Normalize(vmin=np.min(B_V_psf), vmax=np.max(B_V_psf))

plt.figure(figsize=(8, 6))
plt.scatter(B_V, mag,c=B_V,norm=norm, s=5, cmap=cmap, alpha=1)
cbar = plt.colorbar()
cbar.set_label(r'$B-V$ Color Index', fontsize=15)
plt.xlabel(r'$B-V$', fontsize = 15)
plt.ylabel(r'Apparent Magnitude', fontsize = 15)
plt.title(r'Color-Magnitude Diagram', fontsize = 15)
plt.gca().invert_yaxis()  # Invert y-axis (brighter stars at the top)
#plt.gca().invert_xaxis()
plt.grid(True)
plt.show()
plt.tick_params(axis='both', labelsize=13, direction="in")

plt.figure(figsize=(8, 6))
plt.scatter(np.log10(T_eff_list), mag,c=B_V,norm=norm, s=5, cmap=cmap, alpha=1)
cbar = plt.colorbar()
cbar.set_label(r'$B-V$ Color Index', fontsize=15)
plt.xlabel(r'$\log(T_{eff})$', fontsize = 15)
plt.ylabel(r'Apparent Magnitude', fontsize = 15)
plt.title(r'HR Diagram', fontsize = 15)
plt.gca().invert_yaxis()  # Invert y-axis (brighter stars at the top)
plt.gca().invert_xaxis()
plt.grid(True)
plt.show()
plt.tick_params(axis='both', labelsize=13, direction="in")

plt.figure(figsize=(8, 6))
plt.scatter(G_V, mag,c=G_V,norm=norm2, s=5, cmap=cmap, alpha=1)
cbar = plt.colorbar()
cbar.set_label(r'$G-V$ Color Index', fontsize=15)
plt.xlabel(r'$G-V$', fontsize = 15)
plt.ylabel(r'Apparent Magnitude', fontsize = 15)
#plt.title(r'Color-Magnitude Diagram', fontsize = 15)
plt.gca().invert_yaxis()  # Invert y-axis (brighter stars at the top)
#plt.gca().invert_xaxis()
plt.grid(True)
plt.show()
plt.tick_params(axis='both', labelsize=13, direction="in")

# plt.figure(figsize=(8, 6))
# plt.scatter(B_V_psf, psfmag,c=B_V_psf,norm=norm_psf, s=5, cmap=cmap, alpha=1)
# cbar = plt.colorbar()
# cbar.set_label(r'$B-V$ Color Index', fontsize=15)
# plt.xlabel(r'$B-V$', fontsize = 15)
# plt.ylabel(r'Apparent Magnitude', fontsize = 15)
# plt.title(r'Color-Magnitude Diagram', fontsize = 15)
# plt.gca().invert_yaxis()  # Invert y-axis (brighter stars at the top)
# #plt.gca().invert_xaxis()
# plt.grid(True)
# plt.show()
# plt.tick_params(axis='both', labelsize=13, direction="in")
