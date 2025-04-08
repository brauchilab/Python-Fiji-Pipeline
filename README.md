# Python-Fiji-Pipeline
This repository includes code related to Lysosome and Reticle pipeline.


----- Reticle Pipeline -----

This pipeline includes the following steps:
1. Difference of Gauss.
2. Calculate Binary mask of the step 1 (DoG).
3. Remove Outliers.
4. Normalize result of Binary Mask.
5. Calculate Regions of interest (ROIS), F0, log2(pixel/F0). In this step paralellism code is added. 

The pipeline is composed by one main file <main.py>, and a secondary file <utils.py> where the functions are defined.

<main.py> requires to run with the following seven parameters
- input_folder <path>
- output_path <path>
- sigma1 <number>
- sigma2 <number>
- remove_outliers <number>
- roi_size <width height>
- frame_range <start end>

This code will consider all the data inside subfolders too, from the main folder that is passing as input_folder.

Example on how to run the pipeline

Locally:

    python3 main.py --input_folder "/Users/alvaro/Desktop/pyimagej/macro_reticulo/data" --output_path "/Users/alvaro/Desktop/pyimagej/macro_reticulo/output" --sigma1 1.0 --sigma2 3.0 --remove_outliers_filter_size 4 --roi_size 20 20 --frame_range 30 34

Remote Patagon (outside of container): 

    Without sbatch:
        srun -p rtx -c 1 --mem=16G --container-workdir=${PWD} --container-name=cuda-11.2.2 /home/brauchilab/anaconda3/bin/python3 /home/brauchilab/Macros/Macro_Reticulo/pipeline_modular.py --input_folder "/home/brauchilab/Macros/Macro_Reticulo/data/" --output_path "/home/brauchilab/Macros/Macro_Reticulo/output" --sigma1 1.0 --sigma2 2.0 --remove_outliers_filter_size 4 --roi_size 20 20 --frame_range 30 34

    With sbatch: The file job.sh must be created and contain all the parameters required to run the job, once job.sh is set, execute:
        sbatch job.sh  
        



