# Ski Resort Recommendation System

## Overview
This is a collaborative filtering-based ski resort recommendation system built using Streamlit, python and R. It simulates user ratings and predicts how much a user might enjoy an unrated resort based on their preferences.

## Files
- app.py: Main Streamlit app file that runs the interface and prediction system.
- ski_resort_CF.py: Contains data processing, user rating simulation, and prediction logic.
- ski-resorts.csv: Dataset with features for various ski resorts.
- ski_resort_MLP.Rmd: Modeling using an MLP approach in R.

## Requirements
- Python 3.x
- Streamlit
- pandas, numpy, sklearn, plotly

## Running the App
```
streamlit run app.py
```

Make sure `ski-resorts.csv` is in a `data/` folder or adjust the path in `app.py`.
