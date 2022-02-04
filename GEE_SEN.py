#Inladen van de noodzakelijke packages
import ee
import geemap
import geetools
from ipygee import *
import numpy as np
import pandas as pd
import folium
import math
import matplotlib as plt

# Functie aanmaken die S2_SR collectie koppelt aan de s2cloudless info
def get_s2_sr_cld_col(aoi, start_date, end_date, CLOUD_FILTER):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


def add_cloud_bands(img,CLD_PRB_THRESH):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
    
    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')
    
    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img,NIR_DRK_THRESH,CLD_PRJ_DIST):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)
    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
    
    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));
    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
                .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
                .select('distance')
                .mask()
                .rename('cloud_transform'))

    # Identify the intersection of dark pixels withcloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img):
    CLOUD_FILTER = 60
    CLD_PRB_THRESH = 50
    NIR_DRK_THRESH = 0.15
    CLD_PRJ_DIST = 2
    BUFFER = 100
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img,NIR_DRK_THRESH)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud,NIR_DRK_THRESH,CLD_PRJ_DIST)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(BUFFER*2/20)
                   .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
                   .rename('cloudmask'))
    # Add the final cloud-shadow mask to the image.

    return img.addBands(is_cld_shdw)

def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)

#Nodig om naar UINt16() te brengen (= S2-banden)
def addIndices(image):
    #NDVI
    ndvi = image.normalizedDifference(['B8','B4']).rename('NDVI')
    
    #MNDWI
    mndwi = image.normalizedDifference(['B3','B12']).rename('MNDWI')
    
    #NDWI
    ndwi = image.normalizedDifference(['B3','B8']).rename('NDWI')
    
   
    ireci = (image.select('B7').subtract(image.select('B4'))).divide((image.select('B5').divide(image.select('B6')))).float().rename('IRECI')
      
    #SAVI
    savi = image.expression('((B8-B4)/(B8+B4+0.5))*(1+0.5)',
                            {
                                'B8':image.select('B8'),
                                'B4':image.select('B4'),
                            }
                           ).rename('SAVI');
    #S2REP
    s2rep = image.expression('705 + 35 * ((B4 + B7)/2 - B5) / (B6 - B5)',
                            {
                                'B6':image.select('B6'),
                                'B7':image.select('B7'),
                                'B4':image.select('B4'),
                                'B5':image.select('B5'),
                            }
                           ).rename('S2REP');
    

    #Indices aan Image toevoegen
    return image.addBands(ndvi).addBands(mndwi).addBands(ndwi).addBands(ireci).addBands(savi).addBands(s2rep)

def createS2_med(ROI, START_DATE, END_DATE, bands):
    s2_sr_cld_col = get_s2_sr_cld_col(AOI, START_DATE, END_DATE)
    S2_coll = (s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask))
    S2_med = S2_coll.map(addIndices).select(bands).median().clip(ROI)
    return S2_med

def addS2Texturebands(S2, window_size,TextureMetrics):
    GLCM = S2.toInt32().glcmTexture(**{'size': window_size})
    Texture = S2
    TextureBands = []
    for i in TextureMetrics:
        for j in S2.bandNames().getInfo():
            globals()[f"{j}_{i}"] = GLCM.select(j+'_'+i)
            TextureBands.append(j+'_'+i)
            Texture =Texture.addBands(globals()[f"{j}_{i}"])        
    Texture = Texture.select(TextureBands)       
    S2 = S2.addBands(Texture)
    return S2    

def maskEdges(image):
    edge = image.lt(-30.0)
    maskedImage = image.mask().And(edge.Not())
    return image.updateMask(maskedImage)

def createS1coll(ROI):
    S1_coll = (ee.ImageCollection('COPERNICUS/S1_GRD')
               .filter(ee.Filter.eq('instrumentMode', 'IW'))
               .filterBounds(ROI)
               .filterMetadata('resolution_meters','equals', 10)
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
               .map(maskEdges))
    
    return S1_coll.select(['VV','VH'])

    

def add_S1_asc(S1_coll, start_date, end_date):
    imgVV_asc= (S1_coll.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).select('VV').filterDate(start_date,end_date).mean())
    imgVH_asc = (S1_coll.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).select('VH').filterDate(start_date,end_date).mean())
    
    return imgVV_asc.addBands(imgVH_asc)

def add_S1_desc(S1_coll, start_date, end_date):
    imgVV_desc= (S1_coll.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')).select('VV').filterDate(start_date,end_date).mean())
    imgVH_desc = (S1_coll.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')).select('VH').filterDate(start_date,end_date).mean())
    
    return imgVV_desc.addBands(imgVH_desc)
                

def terrainCorrection(image): 
    imgGeom = ee.image.geometry();
    srtm = ee.Image('USGS/SRTMGL1_003').clip(imgGeom); # 30m srtm 
    sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0));
    
    #Article ( numbers relate to chapters) 
    # 2.1.1 Radar geometry 
    theta_i = ee.image.select('angle');
    phi_i = ee.Terrain.aspect(theta_i).reduceRegion(ee.Reducer.mean(), theta_i.get('system:footprint'), 1000).get('aspect');
 
  # 2.1.2 Terrain geometry
    alpha_s = ee.Terrain.slope(srtm).select('slope');
    phi_s = ee.Terrain.aspect(srtm).select('aspect');
    #2.1.3 Model geometry
    # reduce to 3 angle
    phi_r = ee.Image.constant(phi_i).subtract(phi_s);
    
    #convert all to radians
    phi_rRad = phi_r.multiply(np.pi / 180);
    alpha_sRad = alpha_s.multiply(np.pinp.pi / 180);
    theta_iRad = theta_i.multiply(np.pinp.pinp.pi / 180);
    ninetyRad = ee.Image.constant(90).multiply(np.pinp.pinp.pinp.pi / 180);
    
    # slope steepness in range (eq. 2)
    alpha_r = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan();
    
    #slope steepness in azimuth (eq 3)
    alpha_az = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan();
    
    #local incidence angle (eq. 4)
    theta_lia = (alpha_az.cos().multiply((theta_iRad.subtract(alpha_r)).cos())).acos();
    theta_liaDeg = theta_lia.multiply(180 / Math.PI);
    
    # 2.2 
    # Gamma_nought_flat
    gamma0 = sigma0Pow.divide(theta_iRad.cos());
    gamma0dB = ee.Image.constant(10).multiply(gamma0.log10());
    ratio_1 = gamma0dB.select('VV').subtract(gamma0dB.select('VH'));
    
    
    # Volumetric Model
    nominator = (ninetyRad.subtract(theta_iRad).add(alpha_r)).tan();
    denominator = (ninetyRad.subtract(theta_iRad)).tan();
    volModel = (nominator.divide(denominator)).abs();
    
    #apply model
    gamma0_Volume = gamma0.divide(volModel);
    gamma0_VolumeDB = ee.Image.constant(10).multiply(gamma0_Volume.log10());
    
    # we add a layover/shadow maskto the original implmentation
    # layover, where slope > radar viewing angle 
    alpha_rDeg = alpha_r.multiply(180 / Mnp.pi);
    layover = alpha_rDeg.lt(theta_i);
    
    #shadow where LIA > 90
    shadow = theta_liaDeg.lt(85);
    
    #calculate the ratio for RGB vis
    ratio = gamma0_VolumeDB.select('VV').subtract(gamma0_VolumeDB.select('VH'));
    
    output = gamma0_VolumeDB.addBands(ratio).addBands(alpha_r).addBands(phi_s).addBands(theta_iRad).addBands(layover).addBands(shadow).addBands(gamma0dB).addBands(ratio_1);
    
    return image.addBands(output.select(['VV', 'VH'], ['VV', 'VH']),null,true);


 
def powerToDb(img):
    return ee.Image(10).multiply(img.log10());
 
def dbToPower(img):
    return ee.Image(10).pow(img.divide(10));
 

def refinedLee(image):
    bandNames = image.bandNames();
    image = dbToPower(image);
    
    def mapBandnames(b):
        
        img = image.select([b]);
           
        #img must be in natural units, i.e. not in dB!
        #Set up 3x3 kernels 
        weights3 = ee.List.repeat(ee.List.repeat(1,3),3);
        kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, False);

        mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3);
        variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3);

        # Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
        sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]]);

        sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, False);

        # Calculate mean and variance for the sampled windows and store as 9 bands
        sample_mean = mean3.neighborhoodToBands(sample_kernel); 
        sample_var = variance3.neighborhoodToBands(sample_kernel);

        # Determine the 4 gradients for the sampled windows
        gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs();
        gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs());
        gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs());
        gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs());

        #And find the maximum gradient amongst gradient bands
        max_gradient = gradients.reduce(ee.Reducer.max());

        # Create a mask for band pixels that are the maximum gradient
        gradmask = gradients.eq(max_gradient);

        # duplicate gradmask bands: each gradient represents 2 directions
        gradmask = gradmask.addBands(gradmask);

        # Determine the 8 directions
        directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1);
        directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2));
        directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3));
        directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4));
        # The next 4 are the not() of the previous 4
        directions = directions.addBands(directions.select(0).eq(0).multiply(5));
        directions = directions.addBands(directions.select(1).eq(0).multiply(6));
        directions = directions.addBands(directions.select(2).eq(0).multiply(7));
        directions = directions.addBands(directions.select(3).eq(0).multiply(8));

        # Mask all values that are not 1-8
        directions = directions.updateMask(gradmask);

        # "collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
        directions = directions.reduce(ee.Reducer.sum());  

        #pal = ['ffffff','ff0000','ffff00', '00ff00', '00ffff', '0000ff', 'ff00ff', '000000'];
        #Map.addLayer(directions.reduce(ee.Reducer.sum()), {min:1, max:8, palette: pal}, 'Directions', false);

        sample_stats = sample_var.divide(sample_mean.multiply(sample_mean));

        #Calculate localNoiseVariance
        sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0]);

        # Set up the 7*7 kernels for directional statistics
        rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4));

        diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0], 
                                [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]]);

        rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, False);
        diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, False);

        #Create stacks for mean and variance using the original kernels. Mask with relevant direction.
        dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1));
        dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1));

        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)));
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)));

        # and add the bands for rotated kernels
        for i in range(1,4):
            dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)));
            dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)));
            dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)));
            dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)));

        # "collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional band, and is otherwise masked)
        dir_mean = dir_mean.reduce(ee.Reducer.sum());
        dir_var = dir_var.reduce(ee.Reducer.sum());

        #A finally generate the filtered value
        varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0));

        b = varX.divide(dir_var);

        return dir_mean.add(b.multiply(img.subtract(dir_mean))).arrayProject([0]).arrayFlatten([['sum']]).float();
            
    result = ee.ImageCollection(bandNames.map(mapBandnames)).toBands().rename(bandNames);

    return powerToDb(ee.Image(result))

#Apply filter to reduce speckle
def ApplyLeeFilter(S1):
    S1_filtered = refinedLee(S1)
    return S1_filtered


def addS1Texture(S1,TextureMetrics= ['asm','contrast','corr','var','savg','idm','diss','ent'],window_size=5):
    Text_S1 = S1
    # Number of Texture bands currently in the collection:
    GLCM = (S1.toInt32().glcmTexture(**{'size': window_size}))
    TextureBands = []
    for i in TextureMetrics:
        for j in ['VV','VH']:
            globals()[f"{j}_{i}"] = GLCM.select(j+'_'+i)
            TextureBands.append(j+'_'+i)
            Text_S1=Text_S1.addBands(globals()[f"{j}_{i}"])
    Text_S1 = Text_S1.select(TextureBands)
    return S1.addBands(Text_S1)
