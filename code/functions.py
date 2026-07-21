
def open_file(file):
    ds = xr.open_dataset(file)
    df = ds.to_dataframe().reset_index()
    print("Available flattened columns:", df.columns.to_list())
    return df


def classify_stress(df, metric_column, status_col):
    if metric_column == 'saturated_aragonite':
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
    return classified_df
    
def prepare_OA_df(df, OA_column, id_col, obs_depth_col, bottom_depth_col, lat_col, lon_col, time_col, mom6_grid_df):
    df = df.dropna(subset=[OA_column])

    depth_mismatch = abs(df[obs_depth_col] - df[bottom_depth_col]).astype(float)
    max_mismatch = np.minimum(0.1 * df[bottom_depth_col], 20)
    df_filtered = df[depth_mismatch < max_mismatch].copy()

    df_filtered["matched_time"] = pd.to_datetime(df_filtered[time_col]).dt.normalize() + pd.Timedelta(hours=12)
   
    pt_lats = xr.DataArray(df_filtered[lat_col], dims='points')
    pt_lons = xr.DataArray(df_filtered[lon_col], dims='points')

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

def prepare_df(df, DO_col, id_col, obs_depth_col, bottom_depth_col, lat_col, lon_col, time_col, mom6_grid_df, battery_col=None):
    df = df.dropna(subset=[DO_col])

    depth_mismatch = abs(df[obs_depth_col] - df[bottom_depth_col]).astype(float)
    max_mismatch = np.minimum(0.1 * df[bottom_depth_col], 20)
    df_filtered = df[depth_mismatch < max_mismatch].copy()

    df_filtered["matched_time"] = pd.to_datetime(df_filtered[time_col]).dt.normalize() + pd.Timedelta(hours=12)
   
    pt_lats = xr.DataArray(df_filtered[lat_col], dims='points')
    pt_lons = xr.DataArray(df_filtered[lon_col], dims='points')

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
    df = classify_DO(df, 'dissolved_oxygen', 'DO_status', battery_col=battery_col)
    
    requested_columns = [
        "id", "lat", "lon", "obs_depth", "bottom_depth",
        "dissolved_oxygen", "DO_status", "grid_id",
        "matched_time", "matched_lat", "matched_lon"
    ]
    return df[requested_columns]


def prepare_df_no_depth(df, DO_col, id_col, lat_col, lon_col, time_col, mom6_grid_df, battery_col=None):
    df_filtered = df.dropna(subset=[DO_col]).copy()

    if battery_col is not None and battery_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[battery_col] >= 50]
        
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
    df_classified = classify_DO(df_agg, 'dissolved_oxygen', 'DO_status', battery_col=battery_col)
    
    requested_columns = [
        "id", "lat", "lon", "dissolved_oxygen", "DO_status",  
        "grid_id", "matched_time", "matched_lat", "matched_lon"
    ]
    return df_classified[requested_columns]

#Use this function to match mom6 data to observational data 
def load_matching_mom6_data(df, model_data):

    #load in a dataframe list to fill
    df_list = []

    # Get the list of unique datetimes from observational data
    unique_dates = pd.to_datetime(df['matched_time']).dt.normalize().unique()

    print(f"Found {len(unique_dates)} unique dates to query")

    #set the date structure
    for date in unique_dates:
        year = str(date.year)
        month = f"{date.month:02d}"
        day = f"{date.day:02d}"
        
        # Match the exact filename structure
        file_name = f"CEFI-BTM_O2-HINDCAST-{year}{month}{day}.nc"
        target_file_path = model_data / year / month / file_name
        
        # Filter the cruise dataframe down to just the points needed for this specific day
        current_date_str = f"{year}-{month}-{day}"
        df_day = df[pd.to_datetime(df['matched_time']).dt.normalize() == date]
        
        # Check if the file exists before attempting to open it
        if target_file_path.exists():
            print(f"Extracting data for {current_date_str}...")
            
            # Open the dataset
            ds = xr.open_dataset(target_file_path)
            
            # Extract ONLY the bottom oxygen variable
            o2_slice = ds['btm_o2'] 
            
            #  Target only the specific grid IDs present on this day ---
            for _, row in df_day.iterrows():
              try:
                    # Use coordinate matching (.sel) instead of positional indexing (.isel)
                    # This searches the grid using the target latitude and longitude directly
                    model_val = o2_slice.sel(
                        lat=row['matched_lat'], 
                        lon=row['matched_lon'], 
                        method='nearest'
                    ).values
                    
                    #  convert the resulting numpy array/scalar into a standard float
                    btm_o2_val = float(np.atleast_1d(model_val)[0])
              except Exception as e:
                    print(f" Lookup failed for Lat {row['matched_lat']}, Lon {row['matched_lon']}: {e}")
                    btm_o2_val = np.nan

                
                # Build a single clean row combining the cruise data with the model value
              matched_row = row.to_dict()
              matched_row['btm_o2'] = btm_o2_val
              df_list.append(matched_row)
            
            # Close the dataset file connection 
            ds.close()
        else:
            print(f"⚠️ Model file not found for {year}-{month}-{day} at: {target_file_path}")
            # If the file is missing, still retain the cruise data rows but fill model data with NaN
            for _, row in df_day.iterrows():
                matched_row = row.to_dict()
                matched_row['btm_o2'] = np.nan
                df_list.append(matched_row)

    # 4. Convert all individual rows into one single, master DataFrame
    if df_list:
        final_model_df = pd.DataFrame(df_list)
        return final_model_df
    else:
        print("No matching model files were found or loaded.")
        return pd.DataFrame() # Return empty dataframe as fallback
    
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

def hexplot_model_vs_obs_with_trend(df, metric_col):
    # Drop NaNs to ensure clean plotting and calculations
    import matplotlib.colors as colors
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