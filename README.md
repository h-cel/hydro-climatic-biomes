# Hydro-Climatic biomes: a multi-task learning approach.

This repository contains the source codes used in the following publication: 
* Papagiannopoulou, C., Miralles, D. G., Demuzere, M., Verhoest, N. E. C., and Waegeman, W.: Global hydro-climatic biomes identified via multi-task learning, Geosci. Model Dev. Discuss., https://doi.org/10.5194/gmd-2018-92, in review, 2018.
* [![DOI](https://zenodo.org/badge/127893450.svg)](https://zenodo.org/badge/latestdoi/127893450)

### Prerequisites

The script is mainly built using the following packages:

```
python 2.7.13 | Anaconda 5.0.0 (64-bit) (or higher)
scikit-learn
```

## Running the tests

In order to test the framework, a folder with 7 .csv files is provided (test). 
This folder contains the datasets for the locations {(-29.5,144.5), (-29.5,145.5), (-29.5,147.5), (-30.5,143.5), (29.5, 115.5), (29.5, 112.5), (29.5, 113.5)}. For different locations, the user should modify line 23 of the script main.py, providing the right coordinates. 
In general, the user can use the datasets provided on [the SAT-EX data page](http://www.sat-ex.ugent.be/data.php) in order to run the framework at global scale. In this case, line 23 of the script main.py should be modified into "coords = joblib.load('./coords.pkl')". The file coords.pkl is also provided and it contains the coordinates for all the land pixels (on a 1 x 1° resolution).

To execute the script, use the following commands:

```
python main.py value_for_parameter_h value_for_parameter_lambda folder_for_csv_files outpath txt_with_variable_names pkl_file_with_the_unused_feature_indices
```

Where:
* **value_for_parameter_h**: lower that the initial dimension of the dataset observations (< 3209), a value between 8-12 is recommended for the global dataset, while for the test (small) example of the 7 pixels, the value of 2 or 3 is a good option. This value indicates the number of clusters.

* **value_for_parameter_lambda**: a value around 10 is recommended

* **folder_for_csv_files**: e.g., path to the test folder "./test/"

* **outpath**: path to the output folder

* **txt_with_variable_names**: given, i.e., "./vars_all.txt"

* **pkl_file_with_the_unused_feature_indices**: given, i.e., "./rmvdatasets.pkl"

Example:
```
python main.py 2 10 ./test/  ./out/ ./vars_all.txt ./rmvdatasets.pkl
```

## Note

Files main.py and my_solver.py should be stored in the same folder.


## Output

The output of the method is stored in the folder outpath:

The method stores the matrices W, U, V and Theta. For more details see the paper.
The clustering algorithm runs on the V matrix. The result matrices can be loaded with the function joblib.load() as numpy arrays. 

For a simple visualization of the matrix V, use imshow(). Locations with coherent climate-vegetation interactions should have similar values to the components (columns) of this matrix (V matrix).
