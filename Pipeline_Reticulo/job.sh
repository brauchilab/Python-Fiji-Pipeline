#!/bin/bash
#SBATCH --job-name=process_images  # Nombre del trabajo
#SBATCH --output=process_images.out  # Archivo de salida
#SBATCH --error=process_images.err  # Archivo de error
#SBATCH --partition=rtx 
#SBATCH --time=24:00:00  # Tiempo máximo de ejecución (ajustar según sea necesario)
#SBATCH --mem=16GB  # Memoria asignada para el trabajo (ajustar según sea necesario)
#SBATCH --cpus-per-task=10  # Número de CPUs por tarea (ajustar según el código y recursos disponibles)
#SBATCH --ntasks=1  # Número de tareas


srun --container-workdir=${PWD} --container-name=cuda-11.2.2 
/home/brauchilab/anaconda3/bin/python3 /home/brauchilab/Macros/Macro_Reticulo/pipeline_sbatch.py 
--input_folder "/home/brauchilab/Macros/Macro_Reticulo/data/" --output_path 
"/home/brauchilab/Macros/Macro_Reticulo/output" --sigma1 1.0 
--sigma2 2.0 --remove_outliers_filter_size 4 --roi_size 20 20 --frame_range 30 34
