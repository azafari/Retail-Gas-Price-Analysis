# Retail-Gasoline-Price-Analysis
## ECE143 Final Project - Team 1

## Problem:
Analyze the behaviors of gasoline price fluctuations in different regions of the US (from 1993 until now), and interpret the results to across population, geographic location, and availability of petroleum. 

## Dataset:
U.S. Energy Information Administration
(https://www.eia.gov/dnav/pet/pet_pri_gnd_a_epm0_pte_dpgal_w.htm)
The dataset contains the weekly gas price history in different cities and states across the US between 1993-2019. 

## Proposed Solution and Real World Application :
Our proposed solution is to use the Selector Gadget, an open source Chrome Extension, that could discover useful information on a complicated website to crawl the data on the website to the local. After that, we could apply Pyquery, a jquery API in Python, to preprocess the data building it as a dataset. We then use Pandas, a data analysis tool for the Python programming, to extract and filter this dataset. With the data properly processed, we hope to visualize the trends with graphs and a heat-map of the US, and make predictions through regression.
By analyzing gas prices in relation to population, geographic location, and petroleum access, we can gain insight on trends for future predictions, for example, what gas prices might be 10 years from now, and propose meaningful explanations for the behavior as well as what can be done to avoid inflation.
 
## Requirement
- Run ```pip install requirement.txt``` for downloading all the required package.

## Usage
1. ```python main.py```: Crawling down all the data from the website and ploting

## Guide
1. ```./images/```: Stored our final ploting.
2. ```./shape/```:  Some paramters for heatmap plotting.
3. ```./utils/```: Some scripts for preprocessing.
4. ```./vehicle/```: Data for vehicles.
5. ```./population/```: Data for population.