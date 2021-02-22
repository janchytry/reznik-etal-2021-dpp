import sys
import os
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
from sentinelhub import BBoxSplitter, BBox, CRS


class Project:
    '''
    An auxiliary class to create and manage the pipeline's data in the project folder.
    '''
    
    def __init__(self, path_to_project):
        self.FOLDER = self.set_FOLDER(path_to_project)
    
    def set_FOLDER(self, path):
        '''
        Prompts to set a project folder or create a new one.
        Should be set at any Jupyter Notebook which is an instance of the pipeline project.
        '''
        if Project.is_project(path):
            decision = input(f'Project {path} already exists, do you want to use it (1) or create a new one (2)? ')
            if int(decision) == 1:
                print(f'Project {path} has been set to be used.')
                return Project.get_existing_project(path)
            elif int(decision) == 2:
                print('Change the project name at project instance __init__.')
                return None
        os.mkdir(path)
        print(f'Project created at: {path}')
        return path
    
    @staticmethod
    def is_project(path):
        '''
        Checks if project exists.
        '''
        return any(folder == path for folder in os.listdir('.'))
    
    @staticmethod
    def get_existing_project(path):
        '''
        Gets the existing project path, if it exists.
        '''
        for folder in os.listdir('.'):
            if folder == path:
                return folder
    
    def __str__(self):
        return f'Project used: {self.FOLDER}'
        
           
class AOI: 
    '''
    A class to set the area of interest and prepare some essential variables for the Patcher class.
    The area of interest must be in the ESRI Shapefile or GeoJSON formats.
    '''
    
    def __init__(self, path_to_aoi, crs=CRS.UTM_33N):
        self.crs = crs
        self.path = self.set_AOI(path_to_aoi)
        self.gdf = self.set_AOI_gdf()
        self.shape = self.get_AOI_shape()
        self.dimensions = self.get_AOI_dimensions()
        
    def set_AOI(self, path):
        '''
        Sets filepath to the file containing the AOI.
        '''
        while not os.path.isfile(path):
            path = input(f'Path {path} for your area of interest does not exist. Change it: ')
        return path

    def set_AOI_gdf(self):
        '''
        Loads a GeoJSON or Shapefile to a Geopandas GeoGataframe.
        '''
        return gpd.read_file(self.path)  
    
    def get_AOI_crs(self):
        return crs
            
    def get_AOI_shape(self):
        '''
        Extracts shape from the geodataframe.
        '''
        return self.gdf.geometry.values[0]
    
    def get_AOI_dimensions(self):
        '''
        Obtains dimensions of the AOI.
        '''
        shape = self.shape
        return (shape.bounds[2] - shape.bounds[0], shape.bounds[3] - shape.bounds[1])
        
    def convert_desired_CRS(self):
        '''
        For safety, the AOI is converted to the UTM33N CRS. This has to be done manually now.
        The user has to know which UTM zone is their imagery in.
        '''
        self.gdf = self.gdf.to_crs(crs={'init': CRS.ogc_string(self.crs)})
    
    def __str__(self):
        return f'''AOI:
- path: {self.crs}
- dimensions: {self.dimensions[0]} x {self.dimensions[1]} m
- CRS: {self.crs}'''

    
class Patcher:
    '''
    A class that splits the AOI into patches based on the given dimensions. Creates list of bboxes
    and info list with parent bbox (bbox of the aoi) and spatial indexes of bboxes.
    '''
    def __init__(self, project_folder, crs=CRS.UTM_33N):
        self.crs = crs
        self.project_folder = project_folder
        self.bbox_list, self.info_list = None, None
        self.xy_splitters = None
        self.gdf = None
        self.selected_patches = None
        self.patch_gdf_bboxes = None
        
    def get_xy_splitters(self, aoi_dimensions, patch_factor=1):
        '''
        Diminishes the dimensions of the AOI to less-than-hundred numbers to
        represent number of bounding boxes in West-East and North-South directions.
        '''
        x, y = aoi_dimensions
        while x >= 100 or y >= 100: 
            x /= 10
            y /= 10
        self.xy_splitters = (int(x*patch_factor),int(y*patch_factor))
        print(f'Area will be split into {self.xy_splitters[0]} x {self.xy_splitters[1]} patches. You can adjust it by patch_factor.')
        return self.xy_splitters
                    
    def split_bboxes(self, shape):
        '''
        Uses Sentinel Hub BBoxSplitter to split the AOI to bounding boxes
        intersecting or being within the AOI.
        The list of bounding boxes and information about them is retrieved.
        '''
        bbox_splitter = BBoxSplitter([shape], self.crs, self.xy_splitters)
        self.bbox_list = np.array(bbox_splitter.get_bbox_list())
        self.info_list = np.array(bbox_splitter.get_info_list())
    
    def get_patch_gdf(self, subset_patch=None, save=False):
        '''
        It creates a GeoDataFrame from patches according to the subset.
        '''       
        subset = self.select_patch_subset(subset_patch)
        geometry = [Polygon(bbox.get_polygon()) for bbox in self.bbox_list[subset]]
        idxs_x = [info['index_x'] for info in self.info_list[subset]]
        idxs_y = [info['index_y'] for info in self.info_list[subset]]
        df = pd.DataFrame({'index_x': idxs_x, 
                           'index_y': idxs_y,
                           'patch_id': np.arange(0, len(geometry))
                           })
        gdf = gpd.GeoDataFrame(df,
                               crs={'init': CRS.ogc_string(self.crs)}, 
                               geometry=geometry)
        gdf['centre_point'] = gdf.apply(lambda row: Patcher.getXY(row.geometry.centroid), axis=1)
        self.gdf = gdf
        if save: self.save_patches_as_shp()
            
    def select_patch_subset(self, ID):
        '''
        Selects central patch ID and get also IDs of those patches that surround the central patch.
        '''
        if ID is None:
            return np.array([ID for ID in range(len(self.info_list))])
        aux = [idx for idx, [bbox, info] in enumerate(zip(self.bbox_list, self.info_list)) 
                if (abs(info['index_x'] - self.info_list[ID]['index_x']) <= 1 and abs(info['index_y'] - self.info_list[ID]['index_y']) <= 1)]
        
        # Renumbering the patches
        return np.transpose(np.fliplr(np.array(aux).reshape(3, 3))).ravel()

    @staticmethod
    def getXY(centroid):
        '''
        Auxiliary method to get the centroid of each patch for attributing and map-making.
        '''
        return (centroid.x, centroid.y)

    def get_patch_map(self, aoi, save=False):
        '''
        Auxiliary method to print an overview map of bboxes of the AOI.
        '''
        fig, ax = plt.subplots(figsize=(20, 20))
        aoi.plot(ax=ax)
        self.gdf.plot(ax=ax, facecolor='None', edgecolor='r')
        for i in range(len(self.gdf)):
            plt.annotate(s=self.gdf.patch_id[i], xy=self.gdf.centre_point[i], color='r')
        if save: plt.savefig(input())  

    def save_patches_as_shp(self):
        '''
        Saves a GeoDataFrame of selected patches as an ESRI Shapefile.
        '''
        path = f'{self.project_folder}/patches{self.xy_splitters[0]}x{self.xy_splitters[1]}'
        gdf_to_save = self.gdf
        gdf_to_save = gdf_to_save.drop('centre_point', axis=1)
        if not os.path.isdir(path):
            os.mkdir(path)
        gdf_to_save.to_file(f'{path}/patches{self.xy_splitters[0]}x{self.xy_splitters[1]}.shp', driver='ESRI Shapefile')