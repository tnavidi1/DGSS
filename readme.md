# Requirements
Python 3.7 or 3.8

DSS Python:
pip install dss_python
https://github.com/dss-extensions/dss_python

Numpy,
Pandas,
Sklearn,
Argparse,
Statsmodels,
CVXPY,
Mosek,
Matplotlib,
Dask,
Fastparquet,
S3fs


Note Mosek optimization solver requires a license to run:
https://www.mosek.com/products/academic-licenses/


# Installation
installation via pip

# Datafiles
* Download OpenDSS network files from:  
For San Francisco area networks: https://egriddata.org/dataset/bay-area-synthetic-network  
For Iowa network: https://wzy.ece.iastate.edu/Testsystem.html  
For Vermont network: https://sourceforge.net/p/electricdss/code/HEAD/tree/trunk/Distrib/EPRITestCircuits/epri_dpv/J1/  
For Sacramento network: https://sourceforge.net/p/electricdss/code/HEAD/tree/trunk/Distrib/IEEETestCases/123Bus/  
For Arizona network: https://sourceforge.net/p/electricdss/code/HEAD/tree/trunk/Distrib/IEEETestCases/34Bus/  
The various network files can be reformatted to be made compatible with the rest of the code using the methods in standardize_network_files.py


* Download solar data profiles from:  
https://www.nrel.gov/grid/solar-power-data.html


* Download NREL projections from:  
https://data.nrel.gov/submissions/126
(We use high electrification with moderate technology growth in our study)    
For flexible load:
https://data.nrel.gov/submissions/127
(We use high electrification in our study)  
Projections can be extracted from downloaded files using 
read_energy_projections.py


* Download NREL load profiles using
dl_nrel_raw_data.py

* Note that the raw data can be >10GB per network


# Python files
Each python file contains code related to a different aspect of the simulation. A short description of each is as follows.
* DD_models.py: Training the data driven linear power flow models
* dl_nrel_raw_data.py: Extracting and processing the original load profile datasets
* read_energy_projections.py: Extracting the NREL projections from their datafile
* opendss_interface.py: Interfacing the data with the OpenDSS simulator
* EV_Profiles.py: Processing the data for the EV charging windows
* global_optimization.py: Code for running the centralized controller optimization
* local_controller.py: Code for running the local controller
* process_data.py: Code for generating the DER scenarios including PV, EV, and storage placement
* read_results.py: Code for reading the results of the power flow simulation
* standardize_network_files.py: Code for interpreting between different OpenDSS data types

# Procedure
1. Download datafiles as shown in Datafiles section
3. run process_data.py to generate the scenario including PV, EV, and storage placement
4. run local_controller.py to simulate the local controller on the scenario
5. run read_results.py to evaluate the results of the power flow simulation under local control
6. run DD_models.py to train the models used by the centralized controller
7. run global_optimization.py to simulate the centralized controller on the scenario
8. run read_results.py to evaluate the results of the power flow simulation under centralized control