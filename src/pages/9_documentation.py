import streamlit as st

st.markdown("""

This document provides standard short codes for climate variables, following CF/WMO and commonly used conventions.

| Variable                                                                 | Standard Abbreviation | Standard Name (CF)                                               |
|--------------------------------------------------------------------------|------------------------|------------------------------------------------------------------|
| Precipitation                                                             | `pr`                   | precipitation                                                    |
| Snow depth                                                                | `sd`                   | snow_depth                                                       |
| Snowfall flux                                                             | `sf`                   | snowfall_flux                                                    |
| Surface snow amount (mass)                                               | `snw`                  | surface_snow_amount                                              |
| Evaporation (including sublimation & transpiration)                       | `evspsbl`              | evaporation_including_sublimation_and_transpiration              |
| Near-surface wind speed                                                  | `sfcWind`             | near_surface_wind_speed                                          |
| Northward near-surface wind component                                    | `u10`                  | northward_near_surface_wind                                     |
| Eastward near-surface wind component                                     | `v10`                  | eastward_near_surface_wind                                      |
| Near-surface air temperature                                             | `tas`                  | near_surface_air_temperature                                     |
| Surface downwelling shortwave radiation                                  | `ssrd`                 | surface_downwelling_shortwave_radiation                         |
| Total cloud cover percentage                                             | `tcc`                  | total_cloud_cover_percentage                                     |
| Near-surface specific humidity                                           | `huss`                 | near_surface_specific_humidity                                  |

""")
