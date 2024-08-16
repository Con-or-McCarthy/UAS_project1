This is the repository for UAS,HVA COP-Data2Activity course.

### Set up
To set up, clone this repository to your machine
In the terminal, change directory to UAS_project using the ```cd``` command
Create a virtual environment by running ``` python -m venv /path/to/project/UAS_project1/.venv ```, replacing /path/to/project with the path to UAS_project on your machine e.g. ``` /home/User/files/UAS_project1/.venv ```
Activate the venv with ``` source .venv/bin/activate ```
Load packages by running ``` pip install -r requirements.txt ```

### Download models
Download the saved_models.zip file which has been shared with you. Unzip the file and move the contents to the /saved_models/ folder.

### Upload files
Upload your experiment files to the mcroberts_files folder. They must have columns:
``` time,acc x rug,acc y rug,acc z rug,rot x rug,rot y rug,rot z rug,air pressure rug,acc x arm,acc y arm,acc z arm,rot x arm,rot y arm,rot z arm,air pressure arm ```
And file names must begin with the experiment id then an underscore before any other information e.g. DATA20231123_exp4_pp01_tuesday.csv
Filetype must be .csv

### Run inference
In the terminal, type the line:

 ``` python main.py -filename <experiment_ids> -air_pressure <is_there_pressure> -sensor_locations <desired_sensors> ```

* Replace ``` experiment_ids ``` with the code at the start of the filename extracted from the McRoberts devices which of the experiment(s) you want to perform inference on. For example, if you have the files DATA20231123_exp4_pp12_P.csv, you will write ``` -filename DATA20231123 ``` . If there are multiple experiments you can include all the experimnet IDs like so: ``` -filename DATA20231123 DATA20240123 DATA20231125 ```

* Replace ``` is_there_pressure ``` with either 'True' or 'False', depending on if you want the air pressure information included in inference. Default is True.

* Replace ``` <desired_sensors> ``` with 'arm', 'back', or 'both' depending on which sensors you want to be used in inference. Default is 'both'

An example command to analyse files DATA20231123_exp1_pp01.csv and DATA20240123_exp2_pp01.csv using air pressure and both sensors:

``` python main.py -filename DATA20231123 DATA20240123 -air_pressure True -sensor_locations both ```

To analyse only the second experiment, without air pressure, and only the arm sensor:

``` python main.py -filename DATA20240123 -air_pressure False -sensor_locations arm ```

### Read results
After processing, the inference results will be found in folder /model_outputs/. The file name will be of the form ``` <experiment_id>_<desired_sensors>_<is_there_pressure>_<date> ```, taken from your command line input. 
The files are in .csv format with three columns: timestamp,label,confidence. 
'timestamp' column has the associated timestamp for the prediction. Timestamps default to 10s in length with 5s overlap. 
'label' column has the label assigned by the model
'confidence' is the model's probability score for the label, between 0 and 1. 
