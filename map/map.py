#!/usr/bin/env python3
#Importing libraries
import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def create_map(csv_path:str,output_path:str) -> str:
    data = pd.read_csv(f"{csv_path}/stores.csv", header=0)
    #store_sales = pd.read_csv(f"{csv_path}/train.csv", header=0)
    # city = data['city']
    # state = data['state']
    df = data["state"].value_counts().rename_axis('state').reset_index(name='count')
    df1 = pd.DataFrame({
        'state': ['Pichincha', 'Guayas', 'Azuay', 'Santo Domingo de los Tsachilas', 'Manabi', 'Cotopaxi', 'Tungurahua',
                  'Los Rios', 'El Oro', 'Chimborazo', 'Imbabura', 'Bolivar', 'Pastaza', 'Santa Elena', 'Loja',
                  'Esmeraldas'],
        'Country': ['Ecuador', 'Ecuador', 'Ecuador', 'Ecuador', 'Ecuador', 'Ecuador', 'Ecuador', 'Ecuador', 'Ecuador',
                    'Ecuador', 'Ecuador', 'Ecuador', 'Ecuador', 'Ecuador', 'Ecuador', 'Ecuador'],
        'Latitude': [-00.14, -2.13, -3.08333, -0.15, -0.401792, -1.04, -2.55, -1.28, -3.5, -1.91667, 0.36667, -1.58333,
                     -1.91667, -2.08333, -4.16667, 0.83333],
        'Longitude': [-78.31, -79.54, -79.30, -79.10, -79.908538, -80.66, -78.9333, -78.26, -79.83, -78.75, -78.42,
                      -79.08, -77, -80.58, -79.5, -79.25]
    })
    df = df1.merge(df, how='inner', on='state')
    df["Coordinates"] = list(zip(df.Longitude, df.Latitude))
    df["Coordinates"] = df["Coordinates"].apply(Point)
    gdf = gpd.GeoDataFrame(df, geometry="Coordinates")
    ECU = gpd.read_file(f'{csv_path}/gadm36_ECU_shp/gadm36_ECU_2.dbf')

    fig, ax = plt.subplots(1, figsize=(10, 6), dpi=200)
    ECU.plot(color='lightblue', ax=ax)
    ax.axis('off')

    # This plot the cities. It's the same syntax, but we are plotting from a different GeoDataFrame. I want the
    # cities as pale red dots.
    gdf.plot(ax=ax, color='red', alpha=0.5)

    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_title('Ecuador')

    # Kill the spines...
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Label the cities
    for x, y, label in zip(gdf['Coordinates'].x, gdf['Coordinates'].y, gdf['state']):
        ax.annotate(label, xy=(x, y), xytext=(-3, 0.8), textcoords='offset points', fontsize=3)

    fig.savefig(f"{output_path}/map.png")
    plt.show()
    return output_path




if __name__ == "__main__":
    """
       Script is made specifically for a brane package,
       meaning that input parameters are read from the environment
       variables below. The name of the method has to be specified as
       the command line argument.
       """
    command = sys.argv[1]
    csv_path = os.environ["CSV_PATH"]
    output_path = os.environ["OUTPUT_PATH"]

    # ##########################################################################################
    # # For testing function (with 'brane --debug test visualization --data data' in CLI)
    # # NOTE: If you want to use the hardcoded values below instead, remove the first '/' in the file paths.
    # csv_path = "/data"
    # output_path = "/data/map.png"
    # ##########################################################################################

    functions = {
        "create_map": create_map,
    }
    output = functions[command](csv_path,output_path)
    # print("--> START CAPTURE")
    print(yaml.dump({"output": output}))
    # print("--> END CAPTURE")
