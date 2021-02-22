from pprint import pprint
import pickle
import sys
import os
import datetime
import itertools
from contextlib import contextmanager
import zipfile
import re

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
from tqdm import tqdm
import gdal
from sentinelsat import SentinelAPI
import rasterio
import rasterio.warp
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from rasterio.plot import show
from rasterio.mask import mask


# Machine learning 
import lightgbm as lgb
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import preprocessing

from eolearn.core import EOTask, EOPatch, EOExecutor, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk, SaveTask
from eolearn.io import S2L1CWCSInput, ExportToTiff
from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector, AddValidDataMaskTask
from eolearn.geometry import VectorToRaster, PointSamplingTask, ErosionTask
from eolearn.features import LinearInterpolation, SimpleFilterTask
from sentinelhub import BBoxSplitter, BBox, CRS, CustomUrlParam


class S2L2AImages:
    '''
    A class to get Sentinel-2 L2A images.
    For now, this class can only obtain images for patches under a single Sentinel-2 scene,
    under a single UTM zone and where L2A coverage is available. The script does not mosaic
    automatically. 
    '''
    def __init__(self, patch_gdf, path_to_download):
        self.patch_gdf = patch_gdf
        self.path_to_download = path_to_download
        self.bbox = None
        self.available = pd.DataFrame()
        self.downloaded = pd.DataFrame()
        self.api = None
        self.products = None
    
    def get_patches_bbox(self):
        '''
        Gets total bounding box of selected patches for downloading S2L2A images
        through sentinelsat API. Only adjacent or very close patches should can be chosen for now.
        '''
        
        self.bbox = box(*self.patch_gdf.to_crs('EPSG:4326').geometry.total_bounds)
        return self.bbox
    
    def get_available(self, esa_sci_hub_credentials=('username', 'password'), date_range=('20190620', '20190627'), 
                       cloudcov=(0, 100)):
        '''
        According to the user-specified parameters, available Sentinel-2 L2A images are obtained from Sentinelsat API.
        '''
        
        assert self.bbox != None, 'Bbox for patches is empty. Use self.get_patches_bbox to obtain one.'
        self.api = SentinelAPI(*esa_sci_hub_credentials)
        self.products = self.api.query(self.bbox,
                     date=date_range,
                     platformname='Sentinel-2',
                     cloudcoverpercentage=cloudcov,
                     producttype='S2MSI2A')
        assert not self.products == None, 'There were no sentinelsat api or available images found.'        
        self.available = self.api.to_geodataframe(self.products).sort_values(by='ingestiondate')
        return self.available
    
    def select(self, selected=None):
        '''
        Selects desired images by date. For example, if we do not want the whole time series but some particular dates.
        '''
        
        if selected:
            condition = self.available['ingestiondate'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d')).isin(selected)
            self.downloaded.append(self.available[condition]).sort_values(by='ingestiondate')
        else:
            self.downloaded = self.available
        return self.downloaded
            
    def download(self):
        '''
        Wrapper for downloading images.
        '''
        
        assert not self.available.empty, 'There were no sentinelsat api or available images found.'       
        self.downloaded.apply(self.download_aux, axis=1)
                
    def download_aux(self, image_row):
        '''
        Auxiliary method that controls if the zip file already exists. If not, it downloads a single-image 
        zipfile and moves on to the next one as this method is invoked by self.download -> DataFrame.apply.
        '''
        zip_file = image_row[0]+'.zip'
        if not os.path.isdir(zip_file):
            self.api.download(image_row[33], directory_path=self.path_to_download)
 
            

class CustomInput(EOTask):
    '''
    A class that prepares Sentinel-2 images to further work in the pipeline.
    '''
    
    # Bands in the original Sentinel-2 zipfile (SAFE file) are groupped by geometric resolutions
    # Within these groups, they are not ordered by the band number
    # BAND_ORDER is a hash table for ordering the bands by band number later in the process and storing
    # the hash information to the META_DATA FeatureType
    BAND_ORDER = np.array([2,1,0,6,3,4,5,7,8,9])
    BAND_NAMES = np.array(['B4', 'B3', 'B2', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'])
    
    
    def __init__(self, images_gdf, path_to_images, custom_input_name):
        self.images = images_gdf
        self.path_to_images = path_to_images
        self.custom_input_name = custom_input_name
        
    def execute(self, **kwargs):
        '''
        Mandatory EOTask method, orchestrator for the Sentinel-2 imagery processing, using the methods below.
        '''
        # Initializing a new EOPatch
        eopatch = EOPatch()
        
        # Processing S2L2A data and CLM
        band_sets = self.images.apply(lambda row: self.process_bands(*self.load_bands(row), kwargs.get('patch_bbox')), axis=1) 
        
        # Adding new features to EOPatch
        # Besides bands (BANDS) and cloud mask (CLM), the respective bounding box and image timestamps are added.
        # Meta-information about the band name and its resulting order number are also attached.
        eopatch.data['BANDS'] = np.transpose(np.stack(band_sets.to_numpy()), (0,2,3,1))
        eopatch.bbox = BBox(kwargs.get('patch_bbox'), crs=CRS.UTM_33N)
        eopatch.timestamp = self.get_images_dates()
        eopatch.meta_info = {band:order for band, order in zip(CustomInput.BAND_NAMES, CustomInput.BAND_ORDER)}
        
        return eopatch
    
    def load_bands(self, row):
        '''
        Loads 10m and 20m Sentinel-2 bands without unzipping the file. 
        '''
        zip_name = row[0]+'.zip'
        
        # Load S2A zipfile as a GDAL Dataset
        S2L2A_image = gdal.Open(os.path.join(self.path_to_images, zip_name))
        
        # Retrieving the S2A's subdatasets (band groups)
        subdatasets = S2L2A_image.GetSubDatasets()
        l2a10m, l2a20m = subdatasets[0][0], subdatasets[1][0]
        del S2L2A_image
        
        return l2a10m, l2a20m
    
    def process_bands(self, l2a10m, l2a20m, clip_patch):
        '''
        Processes 10m and 20m bands using BandOperations class-interface (further down below) and rasterio MemoryFiles.
        '''
        with rasterio.open(l2a10m) as src:
            with BandOperations.clip_by_patch(src, clip_patch) as clipped:                             
                ten = clipped.read()
        with rasterio.open(l2a20m) as src:
            with BandOperations.upscale(src) as resampled:   
                with BandOperations.clip_by_patch(resampled, clip_patch) as clipped:                    
                    twenty = clipped.read()
        
        return np.concatenate((ten, twenty), axis=0)[CustomInput.BAND_ORDER,...]
    
    def get_images_dates(self):
        '''
        Auxiliary function to get image timestamps from their metadata. Ingestion data is used for the  timestamp.
        '''
        return self.images.ingestiondate.to_list()
    
    
class AddMask(EOTask):
    '''
    A class to retrieve, process and assign a SCL-based mask to all imagery within EOPatches.
    '''
    
    def __init__(self, images_gdf, path_to_images, mask_name='CLM'):
        self.images = images_gdf
        self.path_to_images = path_to_images
        self.mask_name = mask_name
        
    def execute(self, eopatch, **kwargs):
        clms = self.images.apply(lambda row: self.process_clm(self.load_clm(row), kwargs.get('patch_bbox')), axis=1)
        eopatch.mask[self.mask_name] = np.vstack(clms.to_numpy())[..., np.newaxis]
        
        return eopatch
        
    def load_clm(self, row):
        '''
        Loads SCL using GDAL vsizip Virtual File.
        
        This could be similar to loading bands, however, GDAL v2.4.3 Sentinel-2 drivers do not have the
        capability to load SCL as a subdataset. This is resolved in GDAL v3.1.0, which could not be easily 
        installed to the server due to dependency issues.
        '''    
        zip_name = row[0]+'.zip'
        zz = zipfile.ZipFile(os.path.join(self.path_to_images, zip_name))
        scl_path = [f.filename for f in zz.filelist if f.filename.find('SCL_20m') >= 0][0]
        
        assert len(scl_path) != 1, 'Unexpected behaviour of the Sentinel-2 image zipfile when trying to load SCL.'        
        scl = os.path.join('/vsizip', self.path_to_images, zip_name, scl_path)
        del zz
        
        return scl
        
    def process_clm(self, scl, clip_patch):
        '''
        Processes SCL in a similar fashion as process_bands method processes bands.
        '''
        with rasterio.open(scl) as src:
            with BandOperations.upscale(src) as resampled:   
                with BandOperations.clip_by_patch(resampled, clip_patch) as clipped:                       
                    scl = clipped.read()
        
        return BandOperations.scl_to_mask(scl)

    
class BandOperations:
    '''
    Auxiliary class-interface for operations with Sentinel-2 bands and SCL layer.
    '''
    
    # Predetermined reclassification structure for the SCL layer (Baetens et al. 2019)
    SCL_RECLASS = {
        0: False,
        1: True,
        2: True,
        3: False,
        4: True,
        5: True,
        6: True,
        7: True,
        8: False,
        9: False,
        10: False,
        11: True
    }
    
    @contextmanager
    def upscale(raster, upscale_factor=2):
        '''
        Upscales Sentinel-2 20m bands to 10 m pixel size.
        '''
        t = raster.transform
        # Rescale the metadata
        transform = Affine(t.a / upscale_factor, t.b, t.c, t.d, t.e / upscale_factor, t.f)
        height = raster.height * upscale_factor
        width = raster.width * upscale_factor
        profile = raster.profile
        profile.update(transform=transform, driver='GTiff', height=height, width=width)

        # Resample data to target shape
        data = raster.read(
            out_shape=(
                raster.count,
                int(raster.height * upscale_factor),
                int(raster.width * upscale_factor)
            ),
            resampling=Resampling.nearest
        )

        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset: 
                dataset.write(data)
                del data
            with memfile.open() as dataset:  
                yield dataset  

    @contextmanager
    def clip_by_patch(raster, clip_patch):
        '''
        Clips a set of bands by the respective EOPatch's bounding box
        '''
        out_img, out_transform = mask(raster, shapes=[clip_patch], crop=True)
        profile = raster.profile
        profile.update(transform=out_transform, driver='GTiff', height=out_img.shape[1], width=out_img.shape[2])
    
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset: 
                dataset.write(out_img)
                del out_img
            with memfile.open() as dataset:  
                yield dataset  
                
    def scl_to_mask(scl):
        '''
        Reclassifies SCL to a cloud mask. 
        No-data pixels, thus those places where Sentinel-2 scene is not captured, are added to the mask.
        '''
        return np.vectorize(BandOperations.SCL_RECLASS.get)(scl)


class DerivateProduct(EOTask):   
    """
    Custom EOTask that calculates a user-chosen derivate product (index or similar).
    The band placholders must be called the same name as found in EOPatch.meta_info
    """
    
    # Regex expressions to check whether the user-specified eqution contains allowed features (bands)
    ALLOWED_BANDS = re.compile(r"[B][11|12]{2}|[B][8][A]|[B][2|3|4|5|6|7|8]{1}")
    FALSE_BANDS = re.compile(r"[B](?<!\d)[9](?!\d)|[B](?<!\d)[1](?!\d)|[B]10")
    
    def __init__(self, derivate_name, equation):
        self.derivate_name = derivate_name
        self.equation = equation
        self.extracted_features = set(re.findall(DerivateProduct.ALLOWED_BANDS, self.equation))
    
    def check_equation(self):
        '''
        Checks if the equation contains correct features.
        '''
        disallowed = set(re.findall(DerivateProduct.FALSE_BANDS, self.equation))
        assert len(self.extracted_features) > 0, 'There are invalid or no band features in the equation'  
        assert len(disallowed) == 0, 'Bands 1, 9, 10 are not applicable at this point'  
    
    def equation_features_as_variables(self, band_arrays):
        '''
        Extracts feature names to create the variable-like strings that the Python eval method recognizes
        and processes.
        '''
        features_as_variables = list(map(lambda band: 'band_arrays[\"'+ band +'\"]', band_arrays.keys()))
        return dict(zip(band_arrays.keys(), features_as_variables))
    
    def repopulate_equation(self, feature_variables):
        '''
        Creates a new equation for the Python eval function where user-specified band names are exchanged
        for variable-like names.
        '''
        new_equation = self.equation
        for band, variable in feature_variables.items():
             new_equation = new_equation.replace(band, variable)   
        return new_equation
    
    def execute(self, eopatch):
        # Checking equation
        self.check_equation()
        
        # Retrieving the right bands with the help of meta info FeatureType mapping, using extracted bands from the equation
        band_arrays = {band:eopatch.data['BANDS'][..., eopatch.meta_info[band]] for band in self.extracted_features}
        
        # Synthetizing user-specified equation band names with variables to correctly index the band_arrays array
        feature_variables = self.equation_features_as_variables(band_arrays)
        new_equation = self.repopulate_equation(feature_variables)
        
        # Evaluating the new equation where band names are variable-like strings to retrieve genuine band arrays
        derivate_product = eval(new_equation)
        
        # Saving multi-image feature to EOPatch
        eopatch.data[self.derivate_name] = derivate_product[..., np.newaxis]
        
        return eopatch
    

class MaskValidation:
    """ 
    Vaidation of each SCL mask. If the there are more than a threshold of False pixels, it is removed.
    Adapted from the method of Lubej (2019a).
    """
    def __init__(self, threshold):
        self.threshold = threshold
        
    def __call__(self, mask):
        '''
        Return if non-valid pixels constitute less than threshold.
        '''
        valid = np.count_nonzero(mask) / np.size(mask)
        return (1 - valid) < self.threshold

    
class NanRemover(EOTask):
    '''
    An auxiliary class to remove Nan values from sampled
    '''
    def __init__(self, sampled_features_name, sampled_sampled_lulc_name):
        self.sampled_features_name = sampled_features_name
        self.sampled_sampled_lulc_name = sampled_lulc_name
    
    def execute(self, eopatch):
        '''
        Removes no-data values from sampled data.
        '''
        features = eopatch.data[self.sampled_features_name]
        lulc = eopatch.mask_timeless[self.sampled_lulc_name]
        unique_nans = self.get_all_unique_nan_indices(features)
        
        new_features = []
        for timeframe in features:
            timeframe_killed_nans = [np.delete(timeframe[...,0,i], unique_nans) for i in range(11)]
            new_features.append(np.stack(timeframe_killed_nans))        
        eopatch.data[self.sampled_features_name] = np.transpose(np.stack(np.array(new_features))[np.newaxis], (1,3,0,2))
        
        new_classes = np.delete(lulc[..., 0, 0], unique_nans)
        eopatch.mask_timeless[self.sampled_lulc_name] = new_classes[..., np.newaxis, np.newaxis]
        
        return eopatch
    
    def get_all_unique_nan_indices(self, features):
        '''
        Gets all unique nan indices within NumPy array of sampled features.
        '''
        nan_indices = []
        for timeframe in features:
            indices = [np.argwhere(np.isnan(timeframe[..., 0, band])) for band in range(11)]
            indices = np.unique(np.concatenate(indices).ravel())
            nan_indices.append(indices)
        return np.unique(np.concatenate(nan_indices).ravel())
    
    
class EstimatorParser:
    '''
    A class that reduces the last dimensions of the merged-feature time frames and class labels
    to prepare them for the Scikit-learn estimator.
    '''
    
    def __init__(self, eopatches, patch_ids, features='FEATURES_SAMPLED', classes='LULC'):
        self.eopatches = eopatches
        self.patch_ids = patch_ids
        self.features = features
        self.classes = classes
        
    def merge_features(self):
        '''
        Stacks sampled features from EOPatches on the top each other.
        '''
        f_list =  [self.eopatches[pid].data[self.features] for pid in self.patch_ids]
        merged = []
        for i in range(len(f_list)):
            t, px, w, b = f_list[i].shape
            merged.append(f_list[i].reshape(px, t*b))
        return np.concatenate(merged)
    
    def merge_classes(self):
        '''
        Stacks LULC labels from EOPatches on the top each other.
        '''
        return np.concatenate([self.eopatches[pid].mask_timeless[self.classes][..., 0, 0] for pid in self.patch_ids])
    
    def __call__(self):
        # Creating a single vector of pixels and labels in the correct order
        merged_features = self.merge_features()
        merged_classes = self.merge_classes()
        
        return merged_features, merged_classes