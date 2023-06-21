# DC2-Group1-2023
Julia Dobladez Brisquet, 
## About the project
In order to help the Metropolitan Police allocate forces more effectively in the borough of Barnet in London to minimize the number of burglaries that happen, we developed two models; one predicted the number of crimes that would happen in the LSOA areas of Barnet in the next 12 months (Prophet model) and a second one which predicted the most likely LSOA areas to suffer a burglary in the forthcoming year (Forecaster with XGBoost model).
## Getting Started
### Prerequisites
* Python 3 (add it to path (system variables) in order to be able to access it from the command prompt)
* Git (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
### Instructions
1. Clone the code from github
2. Make sure to install all packages from requirement.txt
3. Data needed is provided here; hence there will be no need to run cleaning_data.py
4. Run the Prophet Model (Prophet.ipynb)
5. Run the XGBoost Model (forecaster-xgboost.ipynb)
## Acknowledgments
* Metropolitan Dataset: https://data.police.uk/data/archive/
* Additional Datasets:
  * Median house prices by wards: https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/medianpricepaidbywardhpssadataset37
  * Barnet Ward Profiles and Atlas: https://open.barnet.gov.uk/dataset/e6zwv/barnet-ward-profiles-and-atlas
* Amat Rodrigo, J., &amp; Escobar Ortiz, J. (2021a, February). Forecasting time series with gradient boosting: Skforecast, XGBoost, LightGBM, scikit-learn Y catboost. Forecasting time series with gradient boosting: Skforecast, XGBoost, LightGBM and CatBoost. https://www.cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html 
