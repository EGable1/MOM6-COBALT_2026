
import numpy as np
import dask as ds
import xarray as xr
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import requests
import pandas as pd
import PyCO2SYS as pyco2
import cartopy.crs as ccrs          # ccrs handles projections
import cartopy.feature as cfeature  # cfeature handles land, ocean, coastlines
import cartopy
from shapely.ops import unary_union
import cartopy.feature as cfeature
from datetime import date
from scipy.spatial import KDTree
from datasets import Dataset
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping
from setuptools import setup, find_packages
import statsmodels.api as sm
import seaborn as sns


print("module import success")


sys.path.append(r'C:\\Users\\edmund.gable\\Documents\\Hollings_2026\\RESOURCES\\python\\utilities\\bootstrap\\')
print("path import success")

import pprint
print("success")


from pathlib import Path

root = Path(r"C:\Users\edmund.gable\Documents\Hollings_2026\GitHub\MOM6-COBALT_2026")
raw_data = Path(r"C:\Users\edmund.gable\Documents\Hollings_2026\GitHub\MOM6-COBALT_2026\data\Raw_Data")
model_data = Path(r"Z:\\MOM6\Hindcasts\\BTM-O2-REGRID")
processed_data=Path(r"C:\Users\edmund.gable\Documents\Hollings_2026\GitHub\MOM6-COBALT_2026\data\Processed_Data\comparison")




def open_file(file):
    ds = xr.open_dataset(file)
    df = ds.to_dataframe().reset_index()
    print("Available flattened columns:", df.columns.to_list())
    return df


def classify_stress(df, metric_column, status_col):
    if metric_column == 'saturated_aragonite' or metric_column == 'omega_bottom':
        classified_df=df.copy()
        classified_df.loc[classified_df[metric_column]>3.0, status_col]='Optimal'
        classified_df.loc[classified_df[metric_column]>1.4, status_col]='Saturated'
        classified_df.loc[classified_df[metric_column]<1.4, status_col]='Suboptimal'
        classified_df.loc[classified_df[metric_column]<1.19, status_col]='Atlantic Cod threshold surpassed'
        classified_df.loc[classified_df[metric_column]<1.1, status_col]='Atlantic Sea Scallop threshold surpassed'
        classified_df.loc[classified_df[metric_column]<1, status_col]='Critical threshold surpassed'
        classified_df=classified_df.reset_index()
        
    elif metric_column == 'dissolved_oxygen':
        clean_df = df[df[metric_column] > 0].copy()
        classified_df = clean_df.copy()

        classified_df.loc[classified_df[metric_column] >= 3.9, status_col] = 'Low Oxygen'
        classified_df.loc[classified_df[metric_column] > 5.0, status_col] = 'Oxygenated' 
        classified_df.loc[classified_df[metric_column] < 3.9, status_col] = 'Mildly Hypoxic'
        classified_df.loc[classified_df[metric_column] < 2.41, status_col] = 'Hypoxia'
        classified_df.loc[classified_df[metric_column] < 0.2, status_col] = 'Anoxia'
        classified_df=classified_df.reset_index()
    else:
        clean_df = df[df[metric_column] > 0].copy()
        classified_df = clean_df.copy()

        classified_df.loc[classified_df[metric_column] >= 3.9, status_col] = 'Low Oxygen'
        classified_df.loc[classified_df[metric_column] > 5.0, status_col] = 'Oxygenated' 
        classified_df.loc[classified_df[metric_column] < 3.9, status_col] = 'Mildly Hypoxic'
        classified_df.loc[classified_df[metric_column] < 2.41, status_col] = 'Hypoxia'
        classified_df.loc[classified_df[metric_column] < 0.2, status_col] = 'Anoxia'
        classified_df=classified_df.reset_index()
    
    return classified_df
    
def prepare_OA_df(df, OA_column, id_col, obs_depth_col, bottom_depth_col, lat_col, lon_col, time_col, mom6_grid_df):
    df = df.dropna(subset=[OA_column])

    depth_mismatch = abs(df[obs_depth_col] - df[bottom_depth_col]).astype(float)
    max_mismatch = np.minimum(0.1 * df[bottom_depth_col], 20)
    df_filtered = df[depth_mismatch < max_mismatch].copy()

    df_filtered["matched_time"] = pd.to_datetime(df_filtered[time_col]).dt.normalize() + pd.Timedelta(hours=12)
   
    pt_lats = xr.DataArray(df_filtered[lat_col], dims='points')
    pt_lons = xr.DataArray(df_filtered[lon_col], dims='points')


    dsg = xr.open_dataset(r"Z:\MOM6\V0.0\SOURCE\NES_9KM_HINDCAST\ARAG\CEFI-ARAG-HINDCAST-20070101.nc")
    ny, nx = len(dsg.lat), len(dsg.lon)
    grid_id_matrix = np.array([[f"{y},{x}" for x in range(nx)] for y in range(ny)], dtype=str)
    grid_ds = dsg.assign_coords(grid_id=(['lat', 'lon'], grid_id_matrix))

    nearest_grid = grid_ds.sel(lat=pt_lats, lon=pt_lons, method='nearest')

    df_filtered['matched_lat'] = nearest_grid.lat.values
    df_filtered['matched_lon'] = nearest_grid.lon.values
    df_filtered['grid_id'] = nearest_grid.grid_id.values.astype(str)
    df = df_filtered

    rename_dict = {
        OA_column: "saturated_aragonite",
        id_col: "id",
        obs_depth_col: "obs_depth",
        bottom_depth_col: "bottom_depth",
        lat_col: "lat",
        lon_col: "lon",
        time_col: "time",
    }
    df = df.rename(columns=rename_dict)
    
    agg_rules = {
        "saturated_aragonite": "mean",    
        "obs_depth": "mean",           
        "bottom_depth": "mean",        
        "lat": "mean",                 
        "lon": "mean",                 
        "id": "first",                 
        "matched_lat": "first",        
        "matched_lon": "first"         
    }
      
    df = df.groupby(['matched_time', 'grid_id'], as_index=False).agg(agg_rules)
    df = classify_stress(df, 'saturated_aragonite', 'arag_status')
    
    requested_columns = [
        "id", "lat", "lon", "obs_depth", "bottom_depth",
        "saturated_aragonite", "arag_status", "grid_id",
        "matched_time", "matched_lat", "matched_lon"
    ]
    return df[requested_columns]


def prepare_OA_df_no_depth(df, OA_col, id_col, lat_col, lon_col, time_col, mom6_grid_df):
    df_filtered = df.dropna(subset=[OA_col]).copy()

    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered["matched_time"] = pd.to_datetime(df_filtered[time_col]).dt.normalize() + pd.Timedelta(hours=12)
     
    pt_lats = xr.DataArray(df_filtered[lat_col], dims='points')
    pt_lons = xr.DataArray(df_filtered[lon_col], dims='points')

    nearest_grid = grid_ds.sel(lat=pt_lats, lon=pt_lons, method='nearest')

    df_filtered['matched_lat'] = nearest_grid.lat.values
    df_filtered['matched_lon'] = nearest_grid.lon.values
    raw_ids = nearest_grid.grid_id.values.astype(str)
    
    # 2. Strip any negative signs and ensure a clean space after the separator
    clean_ids = [str(rid).replace('-', '') for rid in raw_ids]
    
    # 3. Save it to the dataframe
    df_filtered['grid_id'] = clean_ids
    # FIX: Cleaned up the assignment typo and converted cleanly to text string arrays
    df_filtered['grid_id'] = nearest_grid.grid_id.values.astype(str)

    rename_dict = {
        OA_col: "saturated_aragonite",
        id_col: "id",
        lat_col: "lat",
        lon_col: "lon",
        time_col: "time",
    }
    df_mapped = df_filtered.rename(columns=rename_dict)
    
    agg_rules = {
        "saturated_aragonite": "mean",     
        "lat": "mean",                  
        "lon": "mean",                  
        "id": "first",                  
        "matched_lat": "first",         
        "matched_lon": "first"          
    }
    
    # FIX: added as_index=False so pandas doesn't convert grid_id into a MultiIndex row layout
    df_agg = df_mapped.groupby(['matched_time', 'grid_id'], as_index=False).agg(agg_rules)
    df_classified = classify_stress(df_agg, 'saturated_aragonite', 'arag_status')
    
    requested_columns = [
        "id", "lat", "lon", "saturated_aragonite", "arag_status",  
        "grid_id", "matched_time", "matched_lat", "matched_lon"
    ]
    return df_classified[requested_columns]


def prepare_DO_df(df, DO_col, id_col, obs_depth_col, bottom_depth_col, lat_col, lon_col, time_col, mom6_grid_df):
    df = df.dropna(subset=[DO_col])

    depth_mismatch = abs(df[obs_depth_col] - df[bottom_depth_col]).astype(float)
    max_mismatch = np.minimum(0.1 * df[bottom_depth_col], 20)
    df_filtered = df[depth_mismatch < max_mismatch].copy()

    df_filtered["matched_time"] = pd.to_datetime(df_filtered[time_col]).dt.normalize() + pd.Timedelta(hours=12)
   
    pt_lats = xr.DataArray(df_filtered[lat_col], dims='points')
    pt_lons = xr.DataArray(df_filtered[lon_col], dims='points')

    dsg = xr.open_dataset(r"Z:\MOM6\V0.0\SOURCE\NES_9KM_HINDCAST\ARAG\CEFI-ARAG-HINDCAST-20070101.nc")
    ny, nx = len(dsg.lat), len(dsg.lon)
    grid_id_matrix = np.array([[f"{y},{x}" for x in range(nx)] for y in range(ny)], dtype=str)
    grid_ds = dsg.assign_coords(grid_id=(['lat', 'lon'], grid_id_matrix))

    nearest_grid = grid_ds.sel(lat=pt_lats, lon=pt_lons, method='nearest')

    df_filtered['matched_lat'] = nearest_grid.lat.values
    df_filtered['matched_lon'] = nearest_grid.lon.values
    df_filtered['grid_id'] = nearest_grid.grid_id.values.astype(str)
    df = df_filtered

    rename_dict = {
        DO_col: "dissolved_oxygen",
        id_col: "id",
        obs_depth_col: "obs_depth",
        bottom_depth_col: "bottom_depth",
        lat_col: "lat",
        lon_col: "lon",
        time_col: "time",
    }
    df = df.rename(columns=rename_dict)
    
    agg_rules = {
        "dissolved_oxygen": "mean",    
        "obs_depth": "mean",           
        "bottom_depth": "mean",        
        "lat": "mean",                 
        "lon": "mean",                 
        "id": "first",                 
        "matched_lat": "first",        
        "matched_lon": "first"         
    }
      
    df = df.groupby(['matched_time', 'grid_id'], as_index=False).agg(agg_rules)
    df = classify_stress(df, 'dissolved_oxygen', 'DO_status')
    
    requested_columns = [
        "id", "lat", "lon", "obs_depth", "bottom_depth",
        "dissolved_oxygen", "DO_status", "grid_id",
        "matched_time", "matched_lat", "matched_lon"
    ]
    
    # Clean return statement (no trailing comma!)
    return df[requested_columns]


def prepare_df_no_depth(df, DO_col, id_col, lat_col, lon_col, time_col, mom6_grid_df):
    df_filtered = df.dropna(subset=[DO_col]).copy()

        
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered["matched_time"] = pd.to_datetime(df_filtered[time_col]).dt.normalize() + pd.Timedelta(hours=12)
     
    pt_lats = xr.DataArray(df_filtered[lat_col], dims='points')
    pt_lons = xr.DataArray(df_filtered[lon_col], dims='points')

    nearest_grid = grid_ds.sel(lat=pt_lats, lon=pt_lons, method='nearest')

    df_filtered['matched_lat'] = nearest_grid.lat.values
    df_filtered['matched_lon'] = nearest_grid.lon.values
    raw_ids = nearest_grid.grid_id.values.astype(str)
    
    # 2. Strip any negative signs and ensure a clean space after the separator
    clean_ids = [str(rid).replace('-', '') for rid in raw_ids]
    
    # 3. Save it to the dataframe
    df_filtered['grid_id'] = clean_ids
    # FIX: Cleaned up the assignment typo and converted cleanly to text string arrays
    df_filtered['grid_id'] = nearest_grid.grid_id.values.astype(str)

    rename_dict = {
        DO_col: "dissolved_oxygen",
        id_col: "id",
        lat_col: "lat",
        lon_col: "lon",
        time_col: "time",
    }
    df_mapped = df_filtered.rename(columns=rename_dict)
    
    agg_rules = {
        "dissolved_oxygen": "mean",     
        "lat": "mean",                  
        "lon": "mean",                  
        "id": "first",                  
        "matched_lat": "first",         
        "matched_lon": "first"          
    }
    
    # FIX: added as_index=False so pandas doesn't convert grid_id into a MultiIndex row layout
    df_agg = df_mapped.groupby(['matched_time', 'grid_id'], as_index=False).agg(agg_rules)
    df_classified = classify_stress(df_agg, 'dissolved_oxygen', 'DO_status')
    
    requested_columns = [
        "id", "lat", "lon", "dissolved_oxygen", "DO_status",  
        "grid_id", "matched_time", "matched_lat", "matched_lon"
    ]
    return df_classified[requested_columns]



def load_combined_OA_mom6_data(df, model_data, grid_ds=None):
    model_data = Path(r"Z:\MOM6\Hindcasts")
    df_list = []
    
    df = df.copy()
    # Ensure standard datetime formats for matching
    df['matched_time'] = pd.to_datetime(df['matched_time'])
    df['lat'] = df['matched_lat']
    df['lon'] = df['matched_lon']

    ion_path = model_data / "ocean_cobalt_daily_2d.19930101-20231231.btm_co3_ion.nc"
    sol_path = model_data / "ocean_cobalt_daily_2d.19930101-20231231.btm_co3_sol_arag.nc"

    if not (ion_path.exists() and sol_path.exists()):
        raise FileNotFoundError(f"Missing NetCDF files in {model_data}")

    unique_dates = df['matched_time'].dt.normalize().unique()
    print(f"Processing {len(unique_dates)} unique dates...")
    
    for center_date in unique_dates:
        # 1. Target a 3-day window around the center date
        target_days = [center_date - pd.Timedelta(days=1), center_date, center_date + pd.Timedelta(days=1)]
        daily_slices = []
        
        # Consistent file names across loops
        ion_file_name = "ocean_cobalt_daily_2d.19930101-20231231.btm_co3_ion.nc"
        sol_file_name = "ocean_cobalt_daily_2d.19930101-20231231.btm_co3_sol_arag.nc"
        
        ion_target_file_path = model_data / ion_file_name
        sol_target_file_path = model_data / sol_file_name

        if ion_target_file_path.exists() and sol_target_file_path.exists():
            # Open lazily without bringing 30GB into RAM
            ion_ds = xr.open_dataset(ion_target_file_path) 
            sol_ds = xr.open_dataset(sol_target_file_path)
            
            for d in target_days:
                try:
                    # Slice the exact day from both datasets lazily
                    ion_day = ion_ds['btm_co3_ion'].sel(time=d, method='nearest')
                    sol_day = sol_ds['btm_co3_sol_arag'].sel(time=d, method='nearest')
                    
                    # Compute aragonite saturation state lazily via vector division
                    arag_day = ion_day / sol_day
                    
                    # Compute/load ONLY this day's 2D slice into memory
                    daily_slices.append(arag_day.load())
                except KeyError:
                    # Catches dates out of the NetCDF time bound range gracefully
                    continue
                    
        if not daily_slices:
            continue
            
        # Combine the 3 days into a small 3D local in-memory cube
        ds_window = xr.concat(daily_slices, dim='time')
        
        if grid_ds is not None:
            # Match MOM6 grid array dimensions if a grid file mapping is passed
            ds_window = ds_window.assign_coords(grid_id=(['yh', 'xh'], grid_ds.grid_id.values))
            ds_window['grid_id'] = ds_window['grid_id'].astype(str)
            
        # Map back to MOM6 coordinate designations (yh, xh)
        yh_arr, xh_arr = ds_window.yh.values, ds_window.xh.values
        df_day = df[df['matched_time'].dt.normalize() == center_date]
        
        # 2. Extract space-time metrics for each observation point on this day
        for idx, row in df_day.iterrows():
            # Find closest center indices
            yh_idx = int(np.argmin(np.abs(yh_arr - row['lat'])))
            xh_idx = int(np.argmin(np.abs(xh_arr - row['lon'])))
            
            # Slice out the 3x3 horizontal window from the combined aragonite window
            yh_slice = slice(max(0, yh_idx - 1), yh_idx + 2)
            xh_slice = slice(max(0, xh_idx - 1), xh_idx + 2)
            
            cube_3x3x3 = ds_window.isel(yh=yh_slice, xh=xh_slice)
            arag_cube = cube_3x3x3.values 
            
            if np.all(np.isnan(arag_cube)):
                continue
                
            # --- EXACT SPATIOTEMPORAL MATCH ---
            time_deltas = np.abs(pd.to_datetime(cube_3x3x3.time.values) - center_date)
            t_exact_idx = int(np.argmin(time_deltas))
            
            y_exact_idx = int(np.argmin(np.abs(cube_3x3x3.yh.values - row['lat'])))
            x_exact_idx = int(np.argmin(np.abs(cube_3x3x3.xh.values - row['lon'])))
            
            exact_match_val = float(arag_cube[t_exact_idx, y_exact_idx, x_exact_idx])

            # --- EXTRACT RAW ION AND SOL VALUES FOR EXACT MATCH ---
            # We look at the exact same spatial/temporal indices in the daily datasets
            exact_time_str = str(cube_3x3x3.time.values[t_exact_idx])
            exact_yh = cube_3x3x3.yh.values[y_exact_idx]
            exact_xh = cube_3x3x3.xh.values[x_exact_idx]
            
            # Pull single scalar values directly out of the open handles
            exact_ion_val = float(ion_ds['btm_co3_ion'].sel(time=exact_time_str, yh=exact_yh, xh=exact_xh, method='nearest').values)
            exact_sol_val = float(sol_ds['btm_co3_sol_arag'].sel(time=exact_time_str, yh=exact_yh, xh=exact_xh, method='nearest').values)

            # --- CLOSEST 3x3x3 NEIGHBOR MATCH ---
            abs_diff = np.abs(arag_cube - row['saturated_aragonite'])
            flat_idx = np.nanargmin(abs_diff) 
            grid_t, grid_y, grid_x = np.unravel_index(flat_idx, arag_cube.shape)
            
            closest_val = float(arag_cube[grid_t, grid_y, grid_x])
            closest_lat = float(cube_3x3x3.yh.values[grid_y])
            closest_lon = float(cube_3x3x3.xh.values[grid_x])
            closest_time = pd.Timestamp(cube_3x3x3.time.values[grid_t])
            
            # Extract Grid ID
            if 'grid_id' in cube_3x3x3.coords:
                text = str(cube_3x3x3.grid_id.values[grid_y, grid_x]).replace('-', '')
                sep = ';' if ';' in text else ','
                y_val, x_val = text.split(sep)
                closest_grid_id = f"({y_val.strip()}{sep} {x_val.strip()})"
            else:
                closest_grid_id = f"({closest_lat}, {closest_lon})"
                
            # --- MERGE INTO DATA DICTIONARY ---
            matched_row = row.to_dict()
            matched_row.update({
                "exact_model_val": exact_match_val,
                "model_btm_ion": exact_ion_val,       # <--- NEW COLUMN
                "model_btm_sol": exact_sol_val,       # <--- NEW COLUMN
                "cube_mean": float(np.nanmean(arag_cube)),
                "cube_std": float(np.nanstd(arag_cube)),
                "cube_min": float(np.nanmin(arag_cube)),
                "cube_max": float(np.nanmax(arag_cube)),
                "closest_model_val": closest_val,
                "closest_grid_id": closest_grid_id,
                "closest_lat": closest_lat,
                "closest_lon": closest_lon,
                "closest_time": closest_time
            })
            df_list.append(matched_row)
            
    return pd.DataFrame(df_list).reset_index(drop=True)

"""def load_combined_OA_mom6_data(df, model_data, grid_ds=None):
    model_data = Path(r"Z:\MOM6\Hindcasts")
    df_list = []
    
    df = df.copy()
    df['matched_time'] = pd.to_datetime(df['matched_time'])
    df['lat'] = df['matched_lat']
    df['lon'] = df['matched_lon']
    
    # Define file paths
    ion_path = model_data / "ocean_cobalt_daily_2d.19930101-20231231.btm_co3_ion.nc"
    sol_path = model_data / "ocean_cobalt_daily_2d.19930101-20231231.btm_co3_sol_arag.nc"
    
    if not (ion_path.exists() and sol_path.exists()):
        raise FileNotFoundError(f"Missing NetCDF files in {model_data}")
        
    # 1. Open datasets lazily ONCE outside the date loop
    with xr.open_dataset(ion_path) as ion_ds_full, xr.open_dataset(sol_path) as sol_ds_full:
        
        unique_dates = df['matched_time'].dt.normalize().unique()
        print(f"Processing {len(unique_dates)} unique dates...")
        
        for center_date in unique_dates:
            target_days = [center_date - pd.Timedelta(days=1), center_date, center_date + pd.Timedelta(days=1)]
            
            # Slice both ion and sol for the 3-day window lazily
            try:
                ion_win = ion_ds_full['btm_co3_ion'].sel(time=target_days, method='nearest')
                sol_win = sol_ds_full['btm_co3_sol_arag'].sel(time=target_days, method='nearest')
                
                # Compute aragonite saturation state and load small 3D slice into memory
                arag_win = (ion_win / sol_win).load()
                ion_win = ion_win.load()
                sol_win = sol_win.load()
            except Exception as e:
                print(f"Skipping date {center_date}: {e}")
                continue

            if grid_ds is not None:
                arag_win = arag_win.assign_coords(grid_id=(['yh', 'xh'], grid_ds.grid_id.values))
                arag_win['grid_id'] = arag_win['grid_id'].astype(str)

            yh_arr, xh_arr = arag_win.yh.values, arag_win.xh.values
            df_day = df[df['matched_time'].dt.normalize() == center_date]
            
            for idx, row in df_day.iterrows():
                yh_idx = int(np.argmin(np.abs(yh_arr - row['lat'])))
                xh_idx = int(np.argmin(np.abs(xh_arr - row['lon'])))
                
                yh_slice = slice(max(0, yh_idx - 1), yh_idx + 2)
                xh_slice = slice(max(0, xh_idx - 1), xh_idx + 2)
                
                cube_3x3x3 = arag_win.isel(yh=yh_slice, xh=xh_slice)
                arag_cube = cube_3x3x3.values 
                
                if np.all(np.isnan(arag_cube)):
                    continue
                    
                # Exact spatiotemporal match
                time_deltas = np.abs(pd.to_datetime(cube_3x3x3.time.values) - center_date)
                t_exact_idx = int(np.argmin(time_deltas))
                y_exact_idx = int(np.argmin(np.abs(cube_3x3x3.yh.values - row['lat'])))
                x_exact_idx = int(np.argmin(np.abs(cube_3x3x3.xh.values - row['lon'])))
                
                exact_match_val = float(arag_cube[t_exact_idx, y_exact_idx, x_exact_idx])

                # Extract ion and sol directly from loaded in-memory slices
                ion_cube = ion_win.isel(yh=yh_slice, xh=xh_slice).values
                sol_cube = sol_win.isel(yh=yh_slice, xh=xh_slice).values
                
                exact_ion_val = float(ion_cube[t_exact_idx, y_exact_idx, x_exact_idx])
                exact_sol_val = float(sol_cube[t_exact_idx, y_exact_idx, x_exact_idx])

                # Closest neighbor match
                abs_diff = np.abs(arag_cube - row['saturated_aragonite'])
                flat_idx = np.nanargmin(abs_diff) 
                grid_t, grid_y, grid_x = np.unravel_index(flat_idx, arag_cube.shape)
                
                closest_val = float(arag_cube[grid_t, grid_y, grid_x])
                closest_lat = float(cube_3x3x3.yh.values[grid_y])
                closest_lon = float(cube_3x3x3.xh.values[grid_x])
                closest_time = pd.Timestamp(cube_3x3x3.time.values[grid_t])
                
                if 'grid_id' in cube_3x3x3.coords:
                    text = str(cube_3x3x3.grid_id.values[grid_y, grid_x]).replace('-', '')
                    sep = ';' if ';' in text else ','
                    y_val, x_val = text.split(sep)
                    closest_grid_id = f"({y_val.strip()}{sep} {x_val.strip()})"
                else:
                    closest_grid_id = f"({closest_lat}, {closest_lon})"
                    
                matched_row = row.to_dict()
                matched_row.update({
                    "exact_model_val": exact_match_val,
                    "model_btm_ion": exact_ion_val,
                    "model_btm_sol": exact_sol_val,
                    "cube_mean": float(np.nanmean(arag_cube)),
                    "cube_std": float(np.nanstd(arag_cube)),
                    "cube_min": float(np.nanmin(arag_cube)),
                    "cube_max": float(np.nanmax(arag_cube)),
                    "closest_model_val": closest_val,
                    "closest_grid_id": closest_grid_id,
                    "closest_lat": closest_lat,
                    "closest_lon": closest_lon,
                    "closest_time": closest_time
                })
                df_list.append(matched_row)
                
    return pd.DataFrame(df_list).reset_index(drop=True)"""

def load_combined_DO_mom6_data(df, model_data, mom6_var='btm_o2', grid_ds=None):
    model_data = Path(model_data)
    df_list = []
    
    df = df.copy()
    df['matched_time'] = pd.to_datetime(df['matched_time'])
    unique_dates = df['matched_time'].dt.normalize().unique()
    
    print(f"Processing {len(unique_dates)} unique dates...")
    
    for center_date in unique_dates:
        # 1. Load the 3-day window around the center date
        target_days = [center_date - pd.Timedelta(days=1), center_date, center_date + pd.Timedelta(days=1)]
        daily_slices = []
        
        for d in target_days:
            file_name = f"CEFI-BTM_O2-HINDCAST-{d.year}{d.month:02d}{d.day:02d}.nc"
            target_file_path = model_data / f"{d.year}" / f"{d.month:02d}" / file_name
            
            if target_file_path.exists():
                with xr.open_dataset(target_file_path) as ds:
                    # Apply unit conversion uniformly if needed
                    o2_converted = ds[mom6_var] * 31998.8 * 1.025
                    daily_slices.append(o2_converted.load()) # Load to memory for performance
                    
        if not daily_slices:
            continue
            
        ds_window = xr.concat(daily_slices, dim='time')
        if grid_ds is not None:
            ds_window = ds_window.assign_coords(grid_id=(['lat', 'lon'], grid_ds.grid_id.values))
            ds_window['grid_id'] = ds_window['grid_id'].astype(str)
            
        lat_arr, lon_arr = ds_window.lat.values, ds_window.lon.values
        df_day = df[df['matched_time'].dt.normalize() == center_date]
        
        # 2. Extract space-time metrics for each observation point on this day
        for idx, row in df_day.iterrows():
            # Find closest center indices
            lat_idx = int(np.argmin(np.abs(lat_arr - row['lat'])))
            lon_idx = int(np.argmin(np.abs(lon_arr - row['lon'])))
            
            # Slice out the 3x3 horizontal window
            lat_slice = slice(max(0, lat_idx - 1), lat_idx + 2)
            lon_slice = slice(max(0, lon_idx - 1), lon_idx + 2)
            cube_3x3x3 = ds_window.isel(lat=lat_slice, lon=lon_slice)
            o2_cube = cube_3x3x3.values 
            
            if np.all(np.isnan(o2_cube)):
                continue
                
            # --- EXACT SPATIOTEMPORAL MATCH ---
            # Index exact center (Time=center_date, and center cell of our slice)
            time_deltas = np.abs(pd.to_datetime(cube_3x3x3.time.values) - center_date)
            t_exact_idx = int(np.argmin(time_deltas))
           # t_exact_idx = np.where(pd.to_datetime(cube_3x3x3.time.values).date == center_date.date())[0][0]
            y_exact_idx = int(np.argmin(np.abs(cube_3x3x3.lat.values - row['lat'])))
            x_exact_idx = int(np.argmin(np.abs(cube_3x3x3.lon.values - row['lon'])))
            exact_match_val = float(o2_cube[t_exact_idx, y_exact_idx, x_exact_idx])

            # --- CLOSEST 3x3x3 NEIGHBOR MATCH ---
            abs_diff = np.abs(o2_cube - row['dissolved_oxygen'])
            flat_idx = np.nanargmin(abs_diff) 
            grid_t, grid_y, grid_x = np.unravel_index(flat_idx, o2_cube.shape)
            
            closest_val = float(o2_cube[grid_t, grid_y, grid_x])
            closest_lat = float(cube_3x3x3.lat.values[grid_y])
            closest_lon = float(cube_3x3x3.lon.values[grid_x])
            closest_time = pd.Timestamp(cube_3x3x3.time.values[grid_t])
            
            # Extract Grid ID
            if 'grid_id' in cube_3x3x3.coords:
                text = str(cube_3x3x3.grid_id.values[grid_y, grid_x]).replace('-', '')
                sep = ';' if ';' in text else ','
                y_val, x_val = text.split(sep)
                closest_grid_id = f"({y_val.strip()}{sep} {x_val.strip()})"
            else:
                closest_grid_id = f"({closest_lat}, {closest_lon})"
                
            # --- MERGE INTO DATA DICTIONARY ---
            matched_row = row.to_dict()
            matched_row.update({
                "exact_model_val": exact_match_val,
                "cube_mean": float(np.nanmean(o2_cube)),
                "cube_std": float(np.nanstd(o2_cube)),
                "cube_min": float(np.nanmin(o2_cube)),
                "cube_max": float(np.nanmax(o2_cube)),
                "closest_model_val": closest_val,
                "closest_grid_id": closest_grid_id,
                "closest_lat": closest_lat,
                "closest_lon": closest_lon,
                "closest_time": closest_time
            })
            df_list.append(matched_row)
            
    return pd.DataFrame(df_list).reset_index(drop=True)



def delta_difference(df, metric_col):
    # 1. Only calculate where model values exist
    df = df.dropna(subset=['closest_model_val']).copy()
    
    # 2. Calculate point-by-point delta difference
    df['delta_difference_point'] = df['exact_model_val'] - df[metric_col]
    df['delta_difference_grid'] = df['closest_model_val'] - df[metric_col]
    print(f"Mean of delta difference between model 3x3x3 grid data and observational data is: {np.nanmean(df['delta_difference_grid']):4f}")
    print(f"Mean of delta difference between model exact grid data and observational data is: {np.nanmean(df['delta_difference_point']):4f}")
    # 3. Calculate Global Dataset RMSE
    # RMSE is the square root of the mean of squared differences between model and observations
    global_rmse_grid = np.sqrt(np.nanmean(df['delta_difference_grid'] ** 2))
    print(f"Global RMSE (3x3x3 grid vs obs) between model and observations is: {global_rmse_grid:.4f}")
    
    # Exact Match RMSE (Slices down to only where valid exact pairs exist)
    valid_exact = df.dropna(subset=['delta_difference_point'])
    global_rmse_exact = np.sqrt((valid_exact['delta_difference_point'] ** 2).mean())
    print(f"Global RMSE (Exact point vs Obs): {global_rmse_exact:.4f}")
    print(f"-> Calculated using {len(valid_exact)} valid exact points (out of {len(df)} total pairs).")

    # 4. Calculate Local Z-score (point relative to its 3x3x3 grid)
    # Note: Added 1e-9 to prevent division-by-zero if a cube std is 0 
    df['local_z_score'] = (df['cube_mean'] - df[metric_col]) / (df['cube_std'] + 1e-9)
    
    # 5. Calculate Global Statistics using Grand Variance formulas (assuming N=27 for each grid)
    global_mean = df['cube_mean'].mean()
    obs_mean=df[metric_col].mean()
    
    # Combined variance formula: mean of variances + variance of means
    mean_of_variance=(df['cube_std']**2).mean()
    variance_of_means=df['cube_mean'].var()

    global_variance = mean_of_variance + variance_of_means
    global_std = np.sqrt(global_variance)
    
    print(f"Calculated Observational Mean: {obs_mean:.4f}")
    print(f"Calculated Model Global Mean: {global_mean:.4f}")
    print(f"Calculated Global Std Dev: {global_std:.4f}")
    over_est_pct=(df['delta_difference_grid']>0).mean()*100
    under_est_pct=(df['delta_difference_grid']<0).mean()*100
    print(f"Model grid overestimates the data {over_est_pct} percent of the time")
    print(f"Model grid underestimates the data {under_est_pct} percent of the time")
    avg_overestimation = df[df['delta_difference_grid'] > 0]['delta_difference_grid'].mean()
    avg_underestimation = df[df['delta_difference_grid'] < 0]['delta_difference_grid'].mean()

    print(f"When overestimating, the average excess oxygen is: {avg_overestimation:.4f}")
    print(f"When underestimating, the average deficit oxygen is: {avg_underestimation:.4f}")

    # Calculate the standard deviation of your real-world observations
    obs_std = df[metric_col].std()

    print(f"Real-World Observational STD: {obs_std:.4f}")
    print(f"Model Global STD: {global_std:.4f}")

    # Compare them
    std_ratio = global_std / obs_std
    print(f"Model-to-Observation Variance Ratio: {std_ratio:.2f}")
    print(f"Model RSR: ({global_rmse_grid/obs_std})")
    
    # 6. Calculate Global Z-score for each observational point
    df['global_z_score'] = (df[metric_col] - global_mean) / global_std

    if metric_col=='dissolved_oxygen':
        # 1. Define your subsets (Stressed = Anything that is NOT 'Oxygenated')
        stressed_obs = df[df['DO_status'] != 'Oxygenated']
        stressed_mod = df[df['model_DO_status'] != 'Oxygenated']

        if len(stressed_obs) > 0:
            # --- RAW COUNTS ---
            # Total Stressed Predictions by the model
            total_predicted_stress = len(stressed_mod)
            
            # True Positives: Model predicted stress, and the observation WAS ALSO stressed 
            # (Regardless of whether they match exactly, e.g., Model='Hypoxic', Obs='Anoxic' is a hit)
            true_positives = (stressed_mod['DO_status'] != 'Oxygenated').sum()
            
            # False Positives: Model predicted stress, but it was actually 'Oxygenated'
            false_positives = (stressed_mod['DO_status'] == 'Oxygenated').sum()
            
            # False Negatives: Observation showed stress, but model missed it (predicted 'Oxygenated')
            false_negatives = (stressed_obs['model_DO_status'] == 'Oxygenated').sum()
            
            # --- PERCENTAGES & METRICS ---
            mod_len = total_predicted_stress if total_predicted_stress > 0 else 1
            
            # General Stress Precision: Out of all times the model predicted stress, how often was there ACTUALLY stress?
            stress_precision = (true_positives / mod_len) * 100
            
            # False Positive Rate: Out of all times the model predicted stress, how often was it completely fine?
            fpp = (false_positives / mod_len) * 100
            
            # False Negative Rate: Out of all actual stressed events, how many did the model completely miss?
            fnp = (false_negatives / len(stressed_obs)) * 100
            
            # Recall / Sensitivity: Out of all actual stressed events, how many did the model flag as stressed?
            stress_recall = (true_positives / len(stressed_obs)) * 100
            
            print(f"\n===========================================================")
            print(f"GENERAL STRESS DETECTION METRICS (Binary: Stress vs No Stress)")
            print(f"===========================================================")
            print(f"Total observations showing stressed conditions: {len(stressed_obs)}")
            print(f"Total model outputs showing stressed conditions: {total_predicted_stress}")
            print(f'False Positives: Of the {total_predicted_stress} times the model predicted stress, {fpp:.2f}% were false alarms (actually oxygenated)')
            print(f'False Negatives: Of the {len(stressed_obs)} times the observations detected stress, {fnp:.2f}% were completely missed (predicted oxygenated)')
            print(f'Stress Precision: Of the {total_predicted_stress} times the model predicted stress, {stress_precision:.2f}% correctly co-occurred with an observed stress event')
            print(f"True Positive Capture Rate: Of the {len(stressed_obs)} actual stress events, the model flagged {stress_recall:.2f}% of them")
    
    if metric_col == 'saturated_aragonite':
        ###Must run classify_arag on model combined specifically
        stressed_obs = df[df['saturated_aragonite'] <1.4]
        stressed_mod=df[df['closest_model_val'] <1.4]
        critical_obs=df[df['saturated_aragonite'] <1]
        critical_mod=df[df['closest_model_val']<1]
        obs_is_stressed = df['saturated_aragonite'] < 1.4
        mod_is_stressed = df['closest_model_val'] < 1.4
        total_predicted_stress = len(stressed_mod)
        
        if len(stressed_obs) > 0:
            # 2. Calculate how often the model status matches the observation status exactly
            status_matches = (obs_is_stressed & mod_is_stressed).sum()
            match_rate = (status_matches / max(len(stressed_obs), 1)) * 100
            match_precision = (status_matches / max(len(stressed_mod), 1)) * 100
            false_positive = (mod_is_stressed & ~obs_is_stressed).sum()
            stress_precision = (true_positives / mod_len) * 100
            # False Negative: Obs saw stress (< 1.4), but Model says normal (>= 1.4)
            false_negative = (obs_is_stressed & ~mod_is_stressed).sum()
            stress_recall = (true_positives / len(stressed_obs)) * 100
            fpp = (false_positive / max(len(stressed_mod), 1)) * 100
            fnp = (false_negative / max(len(stressed_obs), 1)) * 100
            # Missed Critical: Obs was critical (< 1) but Model failed to see it as critical (>= 1)
            missed_critical = ((df['saturated_aragonite'] < 1) & ~(df['closest_model_val'] < 1)).sum()
            fcp = (missed_critical / max(len(critical_obs), 1)) * 100
        # --- PRINT STATS ---
            print(f"\n===========================================================")
            print(f"GENERAL STRESS DETECTION METRICS (Binary: Stress vs No Stress)")
            print(f"===========================================================")
            print(f"Total observations showing stressed conditions: {len(stressed_obs)}")
            print(f"Total model outputs showing stressed conditions: {total_predicted_stress}")
            print(f'False Positives: Of the {total_predicted_stress} times the model predicted stress, {fpp:.2f}% were false alarms (actually oxygenated)')
            print(f'False Negatives: Of the {len(stressed_obs)} times the observations detected stress, {fnp:.2f}% were completely missed (predicted oxygenated)')
            print(f'Critical threshold surpassed: Of the {len(critical_obs)} times the observations detected acute dissolution, {fcp:.2f}% were missed by the model')
            print(f'Stress Precision: Of the {total_predicted_stress} times the model predicted stress, {stress_precision:.2f}% correctly co-occurred with an observed stress event')
            print(f"True Positive Capture Rate: Of the {len(stressed_obs)} actual stress events, the model flagged {stress_recall:.2f}% of them")
    return df

def map_delta_difference(csv_file, bathymetry_nc_path, metric_col, title):
    # 1. Load delta data and geospatial shapes
    final = pd.read_csv(csv_file)
    
    # Load shapefile (already known to be in EPSG:4326 via your snippet)
    shapefile = gpd.read_file('https://github.com/hsynan/READ-EDAB-Synan_hydrographic_climatologies/raw/refs/heads/main/data/shapefiles/NES_5REGIONS.zip')
    if shapefile.crs is None:
        shapefile.set_crs("epsg:4326", inplace=True)
    elif shapefile.crs != "epsg:4326":
        shapefile = shapefile.to_crs("epsg:4326")

    # Load MOM6 static depth / bathymetry
    ds_bathym = xr.open_dataset(bathymetry_nc_path)
    
    # Setup base figure
    fig = plt.figure(figsize=(16, 9))
    map_projection = ccrs.PlateCarree()
    ax = plt.axes(projection=map_projection)

    # 2. Dynamic Symmetric Colorbar Normalization (Centered perfectly at 0)
    max_dev = max(abs(final['delta_difference_grid'].min()), abs(final['delta_difference_grid'].max()))
    div_norm = mcolors.TwoSlopeNorm(vmin=-max_dev, vcenter=0.0, vmax=max_dev)
    base_rdbu=plt.cm.get_cmap('RdBu_r')(np.linspace(0, 1, 256))

    neg_bound_pct=div_norm(-0.05)
    pos_bound_pct=div_norm(0.05)

    idx_start = int(neg_bound_pct * 255)
    idx_end = int(pos_bound_pct * 255)

    base_rdbu[idx_start:idx_end+1] = [0.85, 0.85, 0.85, 1.0] 

    # Convert the modified color array back into a usable Matplotlib colormap
    yellow_center_rdbu = mcolors.ListedColormap(base_rdbu)


    # 3. Scatter Plot of Point Observations vs Grid Delta
    sc = ax.scatter(
        final['matched_lon'].astype(float), 
        final['matched_lat'].astype(float), 
        c=final['delta_difference_grid'], 
        cmap=yellow_center_rdbu, 
        norm=div_norm,
        s=4.5, 
        edgecolors= None, 
        linewidths=0.2, 
        transform=map_projection,
        zorder=3,
        marker='s'
    )

    cb = plt.colorbar(sc, shrink=0.8, label='Delta Difference (Observed - Model grid omega)')

# 4. OVERLAY: Bathymetry Contour Lines from MOM6 NetCDF
    lon_var = 'lonh' if 'lonh' in ds_bathym.variables else ('lon' if 'lon' in ds_bathym.variables else 'xh')
    lat_var = 'lath' if 'lath' in ds_bathym.variables else ('lat' if 'lat' in ds_bathym.variables else 'yh')

    # Extract raw 2D depth matrices and define target line thresholds
    depth_data = ds_bathym['deptho'].values
    contour_levels = [25, 150, 400]

    # Plot uncolored neutral gray contour line isobaths
    # FIX: Pass clip_path=ax.patch directly inside the contour function call
    bathym_contours = ax.contour(
        ds_bathym[lon_var], ds_bathym[lat_var], depth_data,
        levels=contour_levels,
        colors='#7f7f7f',       # Clean, uncolored neutral gray lines
        linewidths=1.0,         # Slim line weight
        linestyles='-',         # Solid contour lines
        transform=map_projection,
        clip_path=ax.patch,     # Forces exact clipping to map frame boundaries natively
        zorder=1
    )

    # Custom legend matching the clean line format
    depth_labels = ['Shallow Limit (25m)', 'Midshelf Limit (150m)', 'Shelf-Slope Limit (400m)']
    
    # Generate line handles for the legend matching our contour lines
    legend_lines = [
        plt.Line2D([0], [0], color='#7f7f7f', linewidth=1.0, linestyle='-', label=label)
        for label in depth_labels
    ]
    



    # 5. OVERLAY: Regional Shapefile Boundaries & Text Labels
    # Draw regional outlines
    shapefile.geometry.boundary.plot(
        ax=ax, 
        color='black', 
        linewidth=1.4, 
        linestyle='--', 
        alpha=0.6,
        transform=map_projection,
        zorder=10
    )

    # Define dictionary maps matching your shapefile index/id rows
    region_names = {0: "MAB_south", 1: "MAB_north", 2: "GB", 3: "GOM_west", 4: "GOM_east"}


    region_offsets = {
        "MAB_south":  {"lon": +2, "lat":  -0.4}, # Shift left and up slightly
        "MAB_north":  {"lon":  0.6, "lat": -1.3}, # Shift down slightly
        "GB":         {"lon":  1, "lat":  -1.1}, # Shift right and down
        "GOM_west":   {"lon": -2.6, "lat":  0.1}, # Shift left slightly
        "GOM_east":   {"lon":  1.9, "lat": 0.1}  # Shift right and down
    }

    regional_means = final.groupby('region')['delta_difference_grid'].mean().to_dict()

    for idx, row in shapefile.iterrows():
        name = region_names.get(idx, f"Region {region_names}")
        #display_text = name

        raw_name = region_names.get(idx, f"{idx}") 

        name_mapping={
            "MAB_south": "Mid Atlantic Bight South",
            "MAB_north": "Mid Atlantic Bight",
            "GB": "Georges Bank",
            "GOM_west": "Gulf of Maine West",
            "GOM_east": "Gulf of Maine East"
        }
        
        # Simply replace the underscore with a space
        #display_text = raw_name.replace('_', ' ')
        
        display_clean = name_mapping.get(raw_name, raw_name.replace('_', ' '))
        r_mean=regional_means.get(name, 0.0)
        display_text = f"{display_clean} \n△{r_mean:.2f}"
        
        # Get the default center coordinates
        centroid = row.geometry.centroid
        lon_pos = centroid.x
        lat_pos = centroid.y
        
        # CHANGED: Apply custom offset if the region exists in our dictionary
        if name in region_offsets:
            lon_pos += region_offsets[name]["lon"]
            lat_pos += region_offsets[name]["lat"]
        
        ax.text(
            lon_pos, lat_pos, display_text, # Using the adjusted coordinates
            color='black', fontsize=11, fontweight='bold',
            ha='center', va='center', transform=map_projection,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', linewidth=0.5, boxstyle='round,pad=0.2'),
            zorder=11
        )

    # 6. Map features and basemaps
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.LAND, zorder=2, facecolor='lightgrey')
    
    # Set explicit spatial scope frame
    ax.set_extent([-77, -64, 34.5, 45], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14)

    # 7. Descriptive Stats Block Placement
    mean_delta = final['delta_difference_grid'].mean()
    median_delta = final['delta_difference_grid'].median()  
    total_pts = len(final['delta_difference_grid'].dropna())
    global_mean = final['cube_mean'].mean()
    obs_mean = final[metric_col].mean()   
    obs_std = final[metric_col].std()     
    
    global_variance = (final['cube_std']**2).mean() + final['cube_mean'].var()  
    global_std = np.sqrt(global_variance)
    global_rmse = np.sqrt((final['delta_difference_grid'] ** 2).mean())

    text_str = (
        f"Mean $\Delta$: {mean_delta:+.3f} mg/L\n"
        f"Median $\Delta$: {median_delta:+.3f} mg/L\n"
        f"Model $\mu$: {global_mean:.3f} mg/L\n"
        f"Obs $\mu$: {obs_mean:.3f} mg/L\n"
        f"Obs $\sigma$: {obs_std:.3f} mg/L\n"
        f"Model $\sigma$: {global_std:.3f} mg/L\n"
        f"\n"
        f"RMSE: {global_rmse:.3f} mg/L\n"
        f"RSR: {(global_rmse / obs_std):.4f}\n"
        f"$N$: {total_pts}"
    )

    box_style = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)
    ax.text(
        0.65, 0.02, 
        text_str, 
        transform=ax.transAxes, 
        fontsize=12, 
        fontweight='bold',
        va='bottom', 
        ha='left', 
        bbox=box_style,
        zorder=105 
    )

    plt.show()

def classify(df, depth_col, lat_col='lat', lon_col='lon'):
    from geopandas import GeoDataFrame
    from shapely.geometry import Point
    classified_df = df.copy()

    # 1. Depth Binning (Using pd.cut to avoid overlapping logic)
    bins = [-float('inf'), 25, 150, 400, float('inf')]
    labels = ['Shallow (<25m)', 'Inner-midshelf (<150m)', 'Shelf_slope (<400m)', 'Deep (>400m)']
    classified_df['depth_bin'] = pd.cut(classified_df[depth_col], bins=bins, labels=labels, right=True)

    # 2. Season Classification
    # Convert to datetime, extract month, and map to season    
   
    classified_df['matched_time'] = pd.to_datetime(classified_df['matched_time'], errors='coerce')
    # Map months directly using pd.cut (12 is rolled into Winter using a trick, or simple mapping)
    # Or stick to a vectorized map:
    month_to_season = {
     1: 'Winter', 2: 'Winter', 12: 'Winter',
     3: 'Spring', 4: 'Spring', 5: 'Spring',
     6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    classified_df['season'] = classified_df['matched_time'].dt.month.map(month_to_season)
    # 3. Spatial Join for Regions
    # Load shapefile
    shapefile = gpd.read_file('https://github.com/hsynan/READ-EDAB-Synan_hydrographic_climatologies/raw/refs/heads/main/data/shapefiles/NES_5REGIONS.zip')
    region_names = {0: "MAB_south", 1: "MAB_north", 2: "GB", 3: "GOM_west", 4: "GOM_east"}
    
 
    shapefile['region'] = shapefile.index.map(region_names)

    # Convert DataFrame to GeoDataFrame
    geometry = [Point(xy) for xy in zip(classified_df[lon_col], classified_df[lat_col])]
    gdf = gpd.GeoDataFrame(classified_df, geometry=geometry, crs=shapefile.crs)

    # Spatial join (points within polygons)
    # 'predicate=inner' keeps only rows that match, 'how=left' keeps all data rows
    joined_gdf = gpd.sjoin(gdf, shapefile[['geometry', 'region']], how='left', predicate='within')

    # Drop the geometry and spatial join index columns before returning a clean pandas DataFrame
    final_df = pd.DataFrame(joined_gdf).drop(columns=['geometry', 'index_right'])

    return final_df




def stats(df, metric_col, subset, baseline_std=None):
    # 1. Display point-by-point delta difference
    print(f"Mean of delta difference between model 3x3x3 grid data and observational data in {subset} is: {np.nanmean(df['delta_difference_grid']):4f}")
    print(f"Mean of delta difference between model exact grid data and observational data in {subset} is: {np.nanmean(df['delta_difference_point']):4f}")
    
    # 2. Calculate Global Dataset RMSE
    global_rmse_grid = np.sqrt(np.nanmean(df['delta_difference_grid'] ** 2))
    print(f"Global RMSE (3x3x3 grid vs obs) between model and observations in {subset} is: {global_rmse_grid:.4f}")
    
    # Exact Match RMSE (Slices down to only where valid exact pairs exist)
    valid_exact = df.dropna(subset=['delta_difference_point'])
    global_rmse_exact = np.sqrt((valid_exact['delta_difference_point'] ** 2).mean())
    print(f"Global RMSE (Exact point vs Obs) between model and observations in {subset} is: {global_rmse_exact:.4f}")
    print(f"-> Calculated using {len(valid_exact)} valid exact points in {subset} (out of {len(df)} total pairs).")

   

    # Calculate statistics directly from the matched dataset pairs
    closest_mean = df['closest_model_val'].mean()
    obs_mean = df[metric_col].mean()
    
    # Use standard variation directly for an accurate scale comparison
    closest_std = df['closest_model_val'].std()
    obs_std = df[metric_col].std()
    
    print(f"Calculated Observational Mean: {obs_mean:.4f}")
    print(f"Calculated Model Closest Mean: {closest_mean:.4f}")
    print(f"Calculated Closest Model Std Dev: {closest_std:.4f}")
    
    over_est_pct = (df['delta_difference_grid'] > 0).mean() * 100
    under_est_pct = (df['delta_difference_grid'] < 0).mean() * 100
    print(f"Model grid overestimates the data {over_est_pct:.2f} percent of the time")
    print(f"Model grid underestimates the data {under_est_pct:.2f} percent of the time")
    
    avg_overestimation = df[df['delta_difference_grid'] > 0]['delta_difference_grid'].mean()
    avg_underestimation = df[df['delta_difference_grid'] < 0]['delta_difference_grid'].mean()

    print(f"When overestimating, the average excess {metric_col} is: {avg_overestimation:.4f}")
    print(f"When underestimating, the average deficit {metric_col} is: {avg_underestimation:.4f}")

    # Calculate the standard deviation of your real-world observations
    obs_std = df[metric_col].std()
    rsr_denominator = baseline_std if baseline_std is not None else obs_std

    print(f"Real-World Observational STD: {obs_std:.4f}")
    print(f"Closest Match Model STD: {closest_std:.4f}")

    # Compare them
    std_ratio = closest_std / obs_std if obs_std != 0 else np.nan
    print(f"Model-to-Observation Variance Ratio: {std_ratio:.2f}")
    
    # Print RSR directly to console
    rsr_grid = global_rmse_grid / rsr_denominator if rsr_denominator != 0 else np.nan
    print(f"Model Grid RSR: {rsr_grid:.4f}")
    
    # 6. Calculate Global Z-score for each observational point
    df['global_z_score'] = (df[metric_col] - closest_mean) / closest_std if closest_std != 0 else np.nan

    if metric_col=='dissolved_oxygen':
        # ---  STRESS DETECTION LOGIC ---
        stressed_obs = df[df['dissolved_oxygen']<5]
        stressed_mod = df[df['closest_model_val'] <5]
        obs_is_stressed = df['dissolved_oxygen'] < 5
        mod_is_stressed = df['closest_model_val'] < 5
    if metric_col=='saturated_aragonite':
        stressed_obs = df[df['saturated_aragonite'] <1.4]
        stressed_mod=df[df['closest_model_val'] <1.4]
        obs_is_stressed = df['saturated_aragonite'] < 1.4
        mod_is_stressed = df['closest_model_val'] < 1.4
    
    # Default metric initializations in case len(stressed_obs) == 0
    fpp = np.nan
    fnp = np.nan
    stress_precision = np.nan
    
    #Default values for pnp, fpp, true positives to fill values in case if loop does not run
    true_positives =0
    fpp=np.nan
    fnp=np.nan
    stress_precision = np.nan
    
    if len(stressed_obs) > 0:
        mod_len = len(stressed_mod) if len(stressed_mod) > 0 else 1
        
        # True Positives: Model said stress, and obs was also any form of stress
        true_positives = (obs_is_stressed & mod_is_stressed).sum()
        
        # False Positives: Model said stress, but obs was completely fine (Oxygenated)
        false_positive = (mod_is_stressed & ~obs_is_stressed).sum()
        
        # False Negatives: Obs was stressed, but model completely missed it (said Oxygenated)
        false_negative = (obs_is_stressed & ~mod_is_stressed).sum()
        
        # Percentages
        fpp = (false_positive / mod_len) * 100 if mod_len>0 else np.nan
        fnp = (false_negative / len(stressed_obs)) * 100 if len(stressed_obs)>0 else np.nan
        stress_precision = (true_positives / mod_len) * 100 if mod_len>0 else np.nan

        print(f"\n===========================================================")
        print(f" STRESSED STATUS MATCH METRICS (Binary: Stress vs No Stress)")
        print(f"===========================================================")
        print(f"Total observations showing stressed conditions: {len(stressed_obs)}")
        print(f"Total model outputs showing stressed conditions: {len(stressed_mod)}")
        print(f'False Positives: Of the {len(stressed_mod)} times the model predicted stress, {fpp:.2f}% were false alarms')
        print(f'False Negatives: Of the {len(stressed_obs)} times the observations detected stress, {fnp:.2f}% were completely missed')
        print(f"Stress Precision: Of the {len(stressed_mod)} times the model predicted stress, {stress_precision:.2f}% co-occurred with an observed stress event")
        print(f"----------------------------------------------------------------------------------------------------------------------")
   
    # 7. Gather all metrics into a dictionary
    metrics = {
        'Subset': subset,
        'Mean_Delta_Grid': np.nanmean(df['delta_difference_grid']),
        'Mean_Delta_Point': np.nanmean(df['delta_difference_point']),
        'Global_RMSE_Grid': global_rmse_grid,
        'Global_RMSE_Exact': global_rmse_exact if df['delta_difference_point'].notna().any() else np.nan,
        'RSR_Grid': rsr_grid,
        'Model_Closest_Mean': closest_mean,
        'Obs_Mean': obs_mean,
        'Closest_Std_Dev': closest_std,
        'Obs_Std_Dev': obs_std,
        'Variance_Ratio': std_ratio if obs_std != 0 else np.nan,
        'Bias_Ratio': closest_mean / obs_mean if obs_mean != 0 else np.nan,
        'Variability_Ratio': ((closest_std/closest_mean)/(obs_std/obs_mean)) if (obs_mean != 0 and obs_std != 0) else np.nan,
        'Overestimate_Pct': over_est_pct,
        'Underestimate_Pct': under_est_pct,
        'Avg_Overestimation': avg_overestimation,
        'Avg_Underestimation': avg_underestimation,
        'Total_Obs': len(valid_exact),
        'Total_Stressed_Obs': len(stressed_obs),
        'Total_Stressed_Mod': len(stressed_mod),
        'False_Positives_Pct': fpp,
        'False_Negatives_Pct': fnp,
        'Match_Precision': stress_precision,
        'True_Positive_Accuracy': (true_positives / len(stressed_obs)) * 100 if len(stressed_obs) > 0 else np.nan
    }
    
    return metrics

def plot_model_vs_obs_with_trend(df, metric_col, label):
    # Drop NaNs to ensure clean plotting and calculations
    plot_df = df.dropna(subset=[metric_col, 'closest_model_val']).copy()
    
    n_samples=len(plot_df)
    if n_samples<3:
        return {'slope': np.nan , 'r_value': np.nan , 'N_samples': n_samples}
    x_data = plot_df[metric_col]
    y_data = plot_df['closest_model_val']
    r_val=float(x_data.corr(y_data))
    r2=r_val**2

    results_rma=regress2(x_data.values, y_data.values, 
                           _method_type_2="reduced major axis", 
                           _need_intercept=True)
    
    slope = results_rma['slope']
    intercept = results_rma['intercept']

    region_palette = {
        'MAB_south': '#ff9896',
        'MAB_north': '#d62728',
        'GOM_east': '#75bbfd',
        'GOM_west': '#1f77b4',
        'GB': "#b667bd",
    }
    if 'Region' in plot_df.columns:
        point_colors = plot_df['Region'].map(region_palette).fillna('#1f77b4')
    elif 'Region_Name' in plot_df.columns:
        point_colors = plot_df['Region_Name'].map(region_palette).fillna('#1f77b4')
    else:
        point_colors = '#1f77b4'  # Fallback if no region columns are found

    plt.figure(figsize=(8, 7))
    
    # 1. Core Scatter plot
   # plt.scatter(
       # x_data, 
        #y_data, 
       # s=20, 
       # color='#1f77b4', 
       # alpha=0.4, 
       # edgecolors='none', 
      #  label='Data Points'
   # )
 
    plt.scatter(x_data, y_data, s=3, c=point_colors, alpha=1, edgecolors='none')
    plt.plot([0, 10], [0, 10], color='red', linestyle='--', linewidth=2, label='1:1 Line')
    # 2. Dynamic 1:1 Line of Perfect Agreement
    x_trend = np.linspace(0, 10, 100)
    y_trend = slope * x_trend + intercept
    plt.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label=f'RMA Trend (Slope = {slope:.2f})')
    
    # 3. Calculate and Plot the TYPE 2 (RMA) Linear Relationship Line
   # r_val = x_data.corr(y_data)
   # sign_r = 1 if r_val >= 0 else -1
    
    # RMA Slope = sign(R) * (StdDev(Y) / StdDev(X))
   # slope = sign_r * (y_data.std() / x_data.std())
    
    # Intercept forces the line through the mean of X and Y
    #intercept = y_data.mean() - (slope * x_data.mean())

    
    plt.plot(
        x_trend,            
        y_trend, 
        color='black', 
        linestyle='-', 
        linewidth=2, 
        label=f'N= {n_samples}'
    )       
    rmse=np.sqrt(np.mean((y_data-x_data)**2))
    obs_std=np.std(x_data)
    rsr=rmse/obs_std
    stats_text=(
        f"RMSE: {rmse:.3f} mg/L\n"
        f"Observed std: {obs_std:.2f} mg/L \n"
        f"RSR: {rsr:.3f}\n"
        f"Intercept: {intercept:+.3f} mg/L"
    )

    box_style = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)
    
    plt.gca().text(
        0.95, 0.05, 
        stats_text, 
        transform=plt.gca().transAxes,  # Locks coordinates to 0-1 screen space
        fontsize=11, 
        fontweight='bold',
        va='bottom', 
        ha='right', 
        bbox=box_style
    )

    # 5. Formatting
    plt.title(f"Bottom {metric_col.replace('_', ' ')}: Model vs. Observations\nR = {r_val:.2f} | R^2 = {r2:.2f} |Trend Slope = {slope:.2f}", fontsize=12)
    plt.xlabel("Observations", fontsize=11)
    plt.ylabel("Model", fontsize=11)
    
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    return {'slope':slope , 'r_value': r_val, 'N_samples': n_samples}

def analyze_spatiotemporal_data(df, metric_col, output_path):
    # Ensure there are no null dates
    valid_dates_df = df.dropna(subset=['matched_time']).copy()
    all_summary_rows = []

    region_names = ["MAB_south", "MAB_north", "GB", "GOM_west", "GOM_east"]
    regional_baselines = {}
    
    for region in region_names:
        r_subset = valid_dates_df[valid_dates_df['Region_Name'] == region]
        if len(r_subset) > 0:
            # Store the stable, long-term standard deviation for this region to later calculate RSR from RMSE 
            regional_baselines[region] = r_subset[metric_col].std()
        else:
            regional_baselines[region] = None

    # Updated inner function with tracking arguments defaulting to "ALL"
    def run_complete_evaluation(dataframe_subset, subset_label, year="ALL", season="ALL", region="ALL", depth="ALL", baseline_std=None):
        # 1. FIX: Pass baseline_std down into the stats function!
        metrics_dict = stats(dataframe_subset.copy(), subset=subset_label, baseline_std=baseline_std)
        
        # 2. Extract trend parameters from plot module
        trend_dict = plot_model_vs_obs_with_trend(dataframe_subset, label=subset_label)
        

        r=trend_dict['r_value']
        B=metrics_dict['Bias_Ratio']
        Y=metrics_dict['Variability_Ratio']
        # 3. Merge outputs
        metrics_dict['RMA_Slope'] = trend_dict['slope']
        metrics_dict['Pearson_R'] = trend_dict['r_value']
        metrics_dict['r^2'] = trend_dict['r_value']**2
        metrics_dict['KGE']=1-np.sqrt((r-1)**2+(B-1)**2+(Y-1)**2)
        
        # --- Calculate exact p-value using statsmodels ---
        n = trend_dict['N_samples']
        if n > 2:
            clean_df = dataframe_subset.dropna(subset=[metric_col, 'closest_model_val'])
            x = clean_df[metric_col]
            y = clean_df['closest_model_val']
            x_with_const = sm.add_constant(x)
            model = sm.OLS(y, x_with_const).fit()
            metrics_dict['p_val'] = model.pvalues.iloc[1]
        else:
            metrics_dict['p_val'] = np.nan
        
        metrics_dict['N_Samples'] = n
        metrics_dict['Year'] = year
        metrics_dict['Season'] = season
        metrics_dict['Region'] = region
        metrics_dict['Depth'] = depth
        
        if 'id' in dataframe_subset.columns:
            unique_ids = dataframe_subset['id'].dropna().unique()
            metrics_dict['Aggregated_IDs'] = ", ".join(map(str, unique_ids))
        else:
            metrics_dict['Aggregated_IDs'] = "N/A"
        
        all_summary_rows.append(metrics_dict)
     
    def get_season(month):
        if month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        elif month in [9, 10, 11]: return 'Fall'
        else: return 'Winter'

    valid_dates_df['season'] = valid_dates_df['matched_time'].dt.month.map(get_season)
    depth_bins_order = ['Shallow(<25m)', 'Inner-midshelf(<150m)', 'Shelf_slope(<400m)', 'Deep(>400m)']

    # ----------------------------------
    # PART 1: GLOBAL BY YEAR
    # ----------------------------------
    unique_years = sorted(valid_dates_df['matched_time'].dt.year.unique())
    for year in unique_years:
        print(f"\n===========================================================")
        print(f"               RUNNING STATISTICS FOR YEAR: {year}        ")
        print(f"===========================================================")
        year_df = valid_dates_df[valid_dates_df['matched_time'].dt.year == year]
        run_complete_evaluation(year_df, str(year), year=str(year))

    # ----------------------------------
    # PART 2: GLOBAL BY SEASON    
    # ----------------------------------
    seasons_order = ['Spring', 'Summer', 'Fall', 'Winter']
    for season in seasons_order:
        season_df = valid_dates_df[valid_dates_df['season'] == season]
        if len(season_df) == 0: continue
        print(f"\n===========================================================")
        print(f"               RUNNING STATISTICS FOR SEASON: {season}     ")
        print(f"===========================================================")
        run_complete_evaluation(season_df, f"Season_{season}", season=season)

    # ----------------------------------
    # PART 3: GLOBAL BY REGION
    # ----------------------------------
    if 'region' in valid_dates_df.columns:
        for region in region_names:
            region_df = valid_dates_df[valid_dates_df['region'] == region]
            if len(region_df) == 0: continue
            print(f"\n===========================================================")
            print(f"               RUNNING STATISTICS FOR REGION: {region}     ")
            print(f"===========================================================")
            r_baseline = regional_baselines.get(region)
            run_complete_evaluation(region_df, str(region), region=region, baseline_std=r_baseline)

    # ----------------------------------
    # PART 4: GLOBAL BY DEPTH
    # ----------------------------------
    if 'depth_bin' in valid_dates_df.columns:
        for d_bin in depth_bins_order:
            depth_df = valid_dates_df[valid_dates_df['depth_bin'] == d_bin]
            if len(depth_df) == 0: continue
            print(f"\n===========================================================")
            print(f"               RUNNING STATISTICS FOR DEPTH: {d_bin}       ")
            print(f"===========================================================")
            run_complete_evaluation(depth_df, f"Depth_{d_bin}", depth=d_bin)

    # ----------------------------------
    # PART 5: REGION BY YEAR 
    # ----------------------------------
    print(f"\n Running Region x Year intersections...")
    if 'region' in valid_dates_df.columns:
        for region in region_names:
            for year in unique_years:
                sub_df = valid_dates_df[(valid_dates_df['region'] == region) & (valid_dates_df['matched_time'].dt.year == year)]
                if len(sub_df) < 3: continue  
                label = f"{region}_{year}"
                r_baseline = regional_baselines.get(region)
                run_complete_evaluation(sub_df, label, year=str(year), region=region, baseline_std=r_baseline)

    # ----------------------------------
    # PART 6: REGION BY SEASON
    # ----------------------------------
    print(f"\nRunning Region x Season intersections...")
    if 'region' in valid_dates_df.columns:
        for region in region_names:
            for season in seasons_order:
                sub_df = valid_dates_df[(valid_dates_df['region'] == region) & (valid_dates_df['season'] == season)]
                if len(sub_df) < 3: continue
                label = f"{region}_{season}"
                r_baseline = regional_baselines.get(region)
                # FIX: Added missing baseline_std argument here!
                run_complete_evaluation(sub_df, label, season=season, region=region, baseline_std=r_baseline)

    # ----------------------------------
    # PART 7: REGION BY DEPTH
    # ----------------------------------
    print(f"\nRunning Region x Depth intersections...")
    if 'region' in valid_dates_df.columns and 'depth_bin' in valid_dates_df.columns:
        for region in region_names:
            for depth in depth_bins_order:
                sub_df = valid_dates_df[(valid_dates_df['region'] == region) & (valid_dates_df['depth_bin'] == depth)]
                if len(sub_df) < 3: continue
                label = f"{region}_{depth}"
                r_baseline = regional_baselines.get(region)
                print(f"\n>>> Analyzing Combination: {label} (N={len(sub_df)})")
                run_complete_evaluation(sub_df, label, region=region, depth=depth, baseline_std=r_baseline)

    # ----------------------------------
    # FINAL: EXPORT TO CSV
    # ----------------------------------
    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Matrix compiled. All cross-statistics successfully exported to: {output_path}")
def plot_delta_histograms(csv_file, subset_col):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd    
    # 1. Load data
    df = pd.read_csv(csv_file)

    # 2. Filter data to ensure we drop missing values
    df_clean = df.dropna(subset=["delta_difference_grid", subset_col])

    # 3. Setup styling
    sns.set_theme(style="whitegrid")

    # Define a consistent 5-panel layout (one for each region)
    # Using row or col layout depending on preference; a single column stack works well for comparison
    if subset_col == 'region':
    
        g = sns.FacetGrid(
            df_clean,
            row="region",
            row_order=["MAB_south", "MAB_north", "GB", "GOM_west", "GOM_east"],
            hue="region",
            palette={
                "MAB_south": "#ff9896",
                "MAB_north": "#d62728",
                "GOM_east": "#75bbfd",
                "GOM_west": "#1f77b4",
                "GB": "#9467bd",
            },
            aspect=4,
            height=2.2,
            sharey=False,  # Allows each region's count to scale dynamically
        )

    elif subset_col == 'depth_bin':
        depth_order = ['Shallow (<25m)', 'Inner-midshelf (<150m)', 'Shelf_slope (<400m)', 'Deep (>400m)']

        g = sns.FacetGrid(
            df_clean,
            row="depth_bin",
            row_order=depth_order,
            hue="depth_bin",
            palette={
                "Shallow (<25m)": "#96ffff",
                "Inner-midshelf (<150m)": "#96bbff",
                "Shelf_slope (<400m)": "#2747d6",
                "Deep (>400m)": "#7775fd",
            },
            aspect=4,
            height=2.2,
            sharey=False,  # Allows each depth's count to scale dynamically
        )
    
    elif subset_col == 'season':
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']

        g = sns.FacetGrid(
            df_clean,
            row="season",
            row_order=season_order,
            hue="season",
            palette={
                
                "Spring": "#2ca02c",
                "Summer": "#ff7f0e",
                "Fall": "#d62728",
                "Winter": "#1f77b4"
            },
            aspect=4,
            height=2.2,
            sharey=False,  # Allows each season's count to scale dynamically
        )

    # 4. Map the histogram
    # binwidth=0.4 creates the exact bin sizing increment you requested
    g.map_dataframe(
        sns.histplot,
        x="delta_difference_grid",
        binwidth=0.2,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.75,
    )
    g.set(xlim=(-4, 5))
    # 5. Add a reference line at 0 (Perfect Model-Obs Agreement)
    g.refline(x=0, color="black", linestyle="--", linewidth=1.2)

     # 6. Dynamically add Mean Lines, Mean Text, and Clean Titles
    for bin_name, ax in g.axes_dict.items():
        # Filter data for the specific depth bin to calculate its mean
        bin_data = df_clean[df_clean[subset_col] == bin_name]["delta_difference_grid"]
        
        if not bin_data.empty:
            mean_val = bin_data.mean()
            
            # Draw the red dotted average line
            ax.axvline(x=mean_val, color="crimson", linestyle=":", linewidth=1.8, label=f"Mean: {mean_val:.2f}")
            
            # Add text label for the mean near the top of each plot
            # transform=ax.get_xaxis_transform() places the y-coordinate relative to the top of the axis (0.90 = 90% up)
            ax.text(
                mean_val + 0.1, 0.90, f"Mean: {mean_val:.2f} mg/L", 
                color="crimson", 
                fontweight="bold",
                fontsize=9,
                transform=ax.get_xaxis_transform(),
                ha="left" if mean_val >= 0 else "right" # Adjust alignment so text doesn't overlap the line drastically
            )
        
        # Clean title: Strip out "depth_bin =" entirely and just leave the clean name
        ax.set_title(bin_name, weight="bold", size=11)

    if 'OA' in csv_file: 
        # Adjust layout spacing to avoid overlaps
        plt.subplots_adjust(hspace=0.4)
        g.fig.suptitle(
            "Regional Distribution of Aragonite Delta Differences",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        g.set_axis_labels("Delta Difference (Observed - Model Dissolved Aragonite Ω)", "Count")
        plt.show()
    elif 'DO' or 'CONCAT' in csv_file: 
        plt.subplots_adjust(hspace=0.4)
        g.fig.suptitle(
            "Regional Distribution of Dissolved Oxygen Differences",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        g.set_axis_labels("Delta Difference (Observed - Model Dissolved Oxygen mg/L)", "Count")
        plt.show()
    else:
        plt.subplots_adjust(hspace=0.4)
        g.fig.suptitle(
            "Regional Distribution of Delta Differences",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        g.set_axis_labels("Delta Difference (Observed - Model)", "Count")
        plt.show()





def regress2(_x, _y, _method_type_1 = "ordinary least square",
             _method_type_2 = "reduced major axis",
             _weight_x = [], _weight_y = [], _need_intercept = True):
    # Regression Type II based on statsmodels
    # Type II regressions are recommended if there is variability on both x and y
    # It's computing the linear regression type I for (x,y) and (y,x)
    # and then average relationship with one of the type II methods
    #
    # INPUT:
    #   _x <np.array>
    #   _y <np.array>
    #   _method_type_1 <str> method to use for regression type I:
    #     ordinary least square or OLS <default>
    #     weighted least square or WLS
    #     robust linear model or RLM
    #   _method_type_2 <str> method to use for regression type II:
    #     major axis
    #     reduced major axis <default> (also known as geometric mean)
    #     arithmetic mean
    #   _need_intercept <bool>
    #     True <default> add a constant to relation (y = a x + b)
    #     False force relation by 0 (y = a x)
    #   _weight_x <np.array> containing the weigth of x
    #   _weigth_y <np.array> containing the weigth of y
    #
    # OUTPUT:
    #   slope
    #   intercept
    #   r
    #   std_slope
    #   std_intercept
    #   predict
    #
    # REQUIRE:
    #   numpy
    #   statsmodels
    #
    # The code is based on the matlab function of MBARI.
    # AUTHOR: Nils Haentjens
    # REFERENCE: https://www.mbari.org/products/research-software/matlab-scripts-linear-regressions/

    # Check input
    if _method_type_2 != "reduced major axis" and _method_type_1 != "ordinary least square":
        raise ValueError("'" + _method_type_2 + "' only supports '" + _method_type_1 + "' method as type 1.")

    # Set x, y depending on intercept requirement
    if _need_intercept:
        x_intercept = sm.add_constant(_x)
        y_intercept = sm.add_constant(_y)

    # Compute Regression Type I (if type II requires it)
    if (_method_type_2 == "reduced major axis" or
        _method_type_2 == "geometric mean"):
        if _method_type_1 == "OLS" or _method_type_1 == "ordinary least square":
            if _need_intercept:
                [intercept_a, slope_a] = sm.OLS(_y, x_intercept).fit().params
                [intercept_b, slope_b] = sm.OLS(_x, y_intercept).fit().params
            else:
                slope_a = sm.OLS(_y, _x).fit().params
                slope_b = sm.OLS(_x, _y).fit().params
        elif _method_type_1 == "WLS" or _method_type_1 == "weighted least square":
            if _need_intercept:
                [intercept_a, slope_a] = sm.WLS(
                    _y, x_intercept, weights=1. / _weight_y).fit().params
                [intercept_b, slope_b] = sm.WLS(
                    _x, y_intercept, weights=1. / _weight_x).fit().params
            else:
                slope_a = sm.WLS(_y, _x, weights=1. / _weight_y).fit().params
                slope_b = sm.WLS(_x, _y, weights=1. / _weight_x).fit().params
        elif _method_type_1 == "RLM" or _method_type_1 == "robust linear model":
            if _need_intercept:
                [intercept_a, slope_a] = sm.RLM(_y, x_intercept).fit().params
                [intercept_b, slope_b] = sm.RLM(_x, y_intercept).fit().params
            else:
                slope_a = sm.RLM(_y, _x).fit().params
                slope_b = sm.RLM(_x, _y).fit().params
        else:
            raise ValueError("Invalid literal for _method_type_1: " + _method_type_1)

    # Compute Regression Type II
    if (_method_type_2 == "reduced major axis" or
        _method_type_2 == "geometric mean"):
        # Transpose coefficients
        if _need_intercept:
            intercept_b = -intercept_b / slope_b
        slope_b = 1 / slope_b
        # Check if correlated in same direction
        if np.sign(slope_a) != np.sign(slope_b):
            raise RuntimeError('Type I regressions of opposite sign.')
        # Compute Reduced Major Axis Slope
        slope = np.sign(slope_a) * np.sqrt(slope_a * slope_b)
        if _need_intercept:
            # Compute Intercept (use mean for least square)
            if _method_type_1 == "OLS" or _method_type_1 == "ordinary least square":
                intercept = np.mean(_y) - slope * np.mean(_x)
            else:
                intercept = np.median(_y) - slope * np.median(_x)
        else:
            intercept = 0
        # Compute r
        r = np.sign(slope_a) * np.sqrt(slope_a / slope_b)
        # Compute predicted values
        predict = slope * _x + intercept
        # Compute standard deviation of the slope and the intercept
        n = len(_x)
        diff = _y - predict
        Sx2 = np.sum(np.multiply(_x, _x))
        den = n * Sx2 - np.sum(_x) ** 2
        s2 = np.sum(np.multiply(diff, diff)) / (n - 2)
        std_slope = np.sqrt(n * s2 / den)
        if _need_intercept:
            std_intercept = np.sqrt(Sx2 * s2 / den)
        else:
            std_intercept = 0
    elif (_method_type_2 == "Pearson's major axis" or
          _method_type_2 == "major axis"):
        if not _need_intercept:
            raise ValueError("Invalid value for _need_intercept: " + str(_need_intercept))
        xm = np.mean(_x)
        ym = np.mean(_y)
        xp = _x - xm
        yp = _y - ym
        sumx2 = np.sum(np.multiply(xp, xp))
        sumy2 = np.sum(np.multiply(yp, yp))
        sumxy = np.sum(np.multiply(xp, yp))
        slope = ((sumy2 - sumx2 + np.sqrt((sumy2 - sumx2)**2 + 4 * sumxy**2)) /
                 (2 * sumxy))
        intercept = ym - slope * xm
        # Compute r
        r = sumxy / np.sqrt(sumx2 * sumy2)
        # Compute standard deviation of the slope and the intercept
        n = len(_x)
        std_slope = (slope / r) * np.sqrt((1 - r ** 2) / n)
        sigx = np.sqrt(sumx2 / (n - 1))
        sigy = np.sqrt(sumy2 / (n - 1))
        std_i1 = (sigy - sigx * slope) ** 2
        std_i2 = (2 * sigx * sigy) + ((xm ** 2 * slope * (1 + r)) / r ** 2)
        std_intercept = np.sqrt((std_i1 + ((1 - r) * slope * std_i2)) / n)
        # Compute predicted values
        predict = slope * _x + intercept
    elif _method_type_2 == "arithmetic mean":
        if not _need_intercept:
            raise ValueError("Invalid value for _need_intercept: " + str(_need_intercept))
        n = len(_x)
        sg = np.floor(n / 2)
        # Sort x and y in order of x
        sorted_index = sorted(range(len(_x)), key=lambda i: _x[i])
        x_w = np.array([_x[i] for i in sorted_index])
        y_w = np.array([_y[i] for i in sorted_index])
        x1 = x_w[1:sg + 1]
        x2 = x_w[sg:n]
        y1 = y_w[1:sg + 1]
        y2 = y_w[sg:n]
        x1m = np.mean(x1)
        x2m = np.mean(x2)
        y1m = np.mean(y1)
        y2m = np.mean(y2)
        xm = (x1m + x2m) / 2
        ym = (y1m + y2m) / 2
        slope = (x2m - x1m) / (y2m - y1m)
        intercept = ym - xm * slope
        # r (to verify)
        r = []
        # Compute predicted values
        predict = slope * _x + intercept
        # Compute standard deviation of the slope and the intercept
        std_slope = []
        std_intercept = []

    # Return all that
    return {"slope": float(slope), "intercept": intercept, "r": r,
            "std_slope": std_slope, "std_intercept": std_intercept,
            "predict": predict}


if __name__ == '__main__':
    x = np.linspace(0, 10, 100)
    # Add random error on y
    e = np.random.normal(size=len(x))
    y = x + e
    results = regress2(x, y, _method_type_2="reduced major axis",
                       _need_intercept=False)
    # print(results)

def hexplot(df, metric_col):
    # Drop NaNs to ensure clean plotting and calculations
    import matplotlib.colors as colors
    plot_df = df
    
    n_samples=len(plot_df)
    if n_samples<3:
        return {'slope': np.nan , 'r_value': np.nan , 'N_samples': n_samples}
    x_data = plot_df[metric_col]
    y_data = plot_df['closest_model_val']
    r_val=float(x_data.corr(y_data))
    r2=r_val**2

    results_rma=regress2(x_data.values, y_data.values, 
                           _method_type_2="reduced major axis", 
                           _need_intercept=True)
    
    slope = results_rma['slope']
    intercept = results_rma['intercept']

    plt.figure(figsize=(8, 7))
    
    # 1. Core Scatter plot
   # plt.scatter(
       # x_data, 
        #y_data, 
       # s=20, 
       # color='#1f77b4', 
       # alpha=0.4, 
       # edgecolors='none', 
      #  label='Data Points'
   # )
    plt.hexbin(x_data, y_data, gridsize=30, cmap='Blues', mincnt=1, norm=colors.PowerNorm(gamma=0.5))
    if metric_col=='saturated_aragonite':
        plt.plot([0, 3], [0, 3], color='red', linestyle='--', linewidth=2, label='1:1 Line')
    else:
        plt.plot([0, 12], [0, 12], color='red', linestyle='--', linewidth=2, label='1:1 Line')
    # 2. Dynamic 1:1 Line of Perfect Agreement
    x_trend = np.linspace(0, 12, 100)
    y_trend = slope * x_trend + intercept
    plt.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label=f'RMA Trend (Slope = {slope:.2f})')
    
    # 3. Calculate and Plot the TYPE 2 (RMA) Linear Relationship Line
   # r_val = x_data.corr(y_data)
   # sign_r = 1 if r_val >= 0 else -1
    
    # RMA Slope = sign(R) * (StdDev(Y) / StdDev(X))
   # slope = sign_r * (y_data.std() / x_data.std())
    
    # Intercept forces the line through the mean of X and Y
    #intercept = y_data.mean() - (slope * x_data.mean())

    
    plt.plot(
        x_trend,            
        y_trend, 
        color='black', 
        linestyle='-', 
        linewidth=2, 
        label=f'N= {n_samples}'
    )       
    rmse=np.sqrt(np.mean((y_data-x_data)**2))
    obs_std=np.std(x_data)
    rsr=rmse/obs_std
    stats_text=(
        f"RMSE: {rmse:.3f} mg/L\n"
        f"Observed std: {obs_std:.3f} mg/L \n"
        f"RSR: {rsr:.3f}\n"
        f"Intercept: {intercept:+.2f} mg/L"
    )

    box_style = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)
    
    plt.gca().text(
        0.95, 0.05, 
        stats_text, 
        transform=plt.gca().transAxes,  # Locks coordinates to 0-1 screen space
        fontsize=14, 
        fontweight='bold',
        va='bottom', 
        ha='right', 
        bbox=box_style
    )

    # 5. Formatting
    if metric_col == 'dissolved_oxygen':
        plt.title(f"Bottom Dissolved oxygen(mg/L): Model vs. Observations\nR = {r_val:.2f} |R^2 ={r2:.2f}| Trend Slope = {slope:.2f}", fontsize=14)
        plt.xlabel(f"Observational {metric_col.replace('_', ' ').title()}", fontsize=11)
        plt.ylabel(f"Closest Model Grid {metric_col.replace('_', ' ').title()}", fontsize=11)
    else:
        plt.title(f"Bottom Saturated Aragonite Ω: Model vs. Observations\nR = {r_val:.2f} |R^2 ={r2:.2f}| Trend Slope = {slope:.2f}", fontsize=14)
        plt.xlabel(f"Observational {metric_col.replace('_', ' ').title()}", fontsize=11)
        plt.ylabel(f"Closest Model Grid {metric_col.replace('_', ' ').title()}", fontsize=11)
    if metric_col=='saturated_aragonite':
        plt.xlim(0, 3)
        plt.ylim(0, 3)
    else:
        plt.xlim(0, 12)
        plt.ylim(0, 12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    return {'slope':slope , 'r_value': r_val, 'N_samples': n_samples}   