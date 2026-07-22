# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:16:48 2026

@author: edmund.gable
"""
print("start")
import xarray as xr
import earthaccess as ea
import NumPy as npy
import dask as ds



print("success")
 

def get_bottom_values_at_time(file_path, time, nearest=True,
							  lat_names=None, lon_names=None,
							  depth_names=None, omega_names=None):
	"""Open a NetCDF file and return lat, lon, depth and bottom omega at a time.

	Parameters
	- file_path: str or Path to the .nc file
	- time: string or datetime-like object used to select the time coordinate
	- nearest: if True, select the nearest time value when exact match not found
	- *_names: optional lists of candidate variable/coord names to try

	Returns a dict with keys: 'lat', 'lon', 'depth', 'omega_bottom'. Values are
	xarray DataArray or None if the coordinate wasn't present.
	"""
	import numpy as np

	ds = xr.open_dataset(str(file_path), decode_times=True, chunks="auto")

	def _find_first(candidates):
		for c in candidates:
			if c in ds.coords or c in ds.data_vars:
				return c
		return None

	lat_candidates = lat_names or ["lat", "latitude", "y"]
	lon_candidates = lon_names or ["lon", "longitude", "x"]
	depth_candidates = depth_names or ["depth", "lev", "z", "deptht"]
	omega_candidates = omega_names or ["omega_bottom", "omega", "omega_btm"]

	lat_name = _find_first(lat_candidates)
	lon_name = _find_first(lon_candidates)
	depth_name = _find_first(depth_candidates)
	omega_name = _find_first(omega_candidates)

	# Try a fuzzy lookup for omega if not found
	if omega_name is None:
		for var in ds.data_vars:
			lname = var.lower()
			if "omega" in lname and ("bottom" in lname or "btm" in lname):
				omega_name = var
				break

	if "time" in ds.coords:
		try:
			if nearest:
				ds_sel = ds.sel(time=np.datetime64(time), method="nearest")
			else:
				ds_sel = ds.sel(time=time)
		except Exception:
			ds_sel = ds.sel(time=time)
	else:
		ds_sel = ds

	lat = ds.coords.get(lat_name) if lat_name else None
	lon = ds.coords.get(lon_name) if lon_name else None
	depth = ds.coords.get(depth_name) or ds.data_vars.get(depth_name)
	if omega_name:
		omega = ds_sel[omega_name]
	else:
		raise ValueError(f"Could not find an omega/bottom variable. Available variables: {list(ds.data_vars)}")

	return {"lat": lat, "lon": lon, "depth": depth, "omega_bottom": omega}


def classify_omega_arag(omega_arag, var_name="omega_arag", drop_above=1.4,
                        status_name="omega_arag_status"):
	"""Classify omega_arag values and drop values at or above the drop threshold.

	Parameters
	- omega_arag: xarray.DataArray, xarray.Dataset, or file path to a NetCDF file.
	- var_name: variable name to use when providing a Dataset or file path.
	- drop_above: values greater than or equal to this threshold are removed.
	- status_name: name for the returned classification variable.

	Returns an xarray.Dataset with the selected omega_arag values and a string
	status variable describing the threshold category.
	"""
	from pathlib import Path

	if isinstance(omega_arag, (str, Path)):
		ds = xr.open_dataset(str(omega_arag), decode_times=True, chunks="auto")
		if var_name not in ds:
			raise ValueError(f"Variable '{var_name}' not found in {omega_arag}")
		omega_arag = ds[var_name]
	elif isinstance(omega_arag, xr.Dataset):
		if var_name not in omega_arag:
			raise ValueError(f"Variable '{var_name}' not found in dataset")
		omega_arag = omega_arag[var_name]

	selected = omega_arag.where(omega_arag < drop_above, drop=True)

	status = xr.where(
		selected < 1.0,
		"critical threshold surpassed",
		xr.where(
			selected < 1.1,
			"Atlantic Sea Scallop threshold surpassed",
			xr.where(
				selected < 1.19,
				"Atlantic cod threshold surpassed",
				"suboptimal",
			),
		),
	)

	status = status.rename(status_name)

	return xr.Dataset({var_name: selected, status_name: status})


if __name__ == "__main__":
	# Example usage when run as a script (adjust the path as needed)
	try:
		example_path = r"C:\Users\edmund.gable\Documents\Coding1\GitHub\MOM6-COBALT_2026\data\bottom_OA\vessel_based_bottom_OA_data_2007_2024.nc"
		print("Opening example file (no heavy IO in import):", example_path)
	except Exception:
		pass