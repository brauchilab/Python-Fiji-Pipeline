import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter
import cv2
from skimage.filters import threshold_triangle
import os
import tifffile
import argparse
from utils import process
from scipy.ndimage import median_filter
from concurrent.futures import ProcessPoolExecutor
import time

def main(input_folder, output_path, sigma1, sigma2, remove_outliers_filter_size, roi_size, frame_range):
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            input_tiff = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0]

            specific_output_path = os.path.join(output_path, output_filename)
            os.makedirs(specific_output_path, exist_ok=True)
  
            # Procesos
            dog_stack_path = process.process_stack(input_tiff, specific_output_path, sigma1, sigma2, output_filename)
            mask_path = process.process_dog_stack(dog_stack_path, specific_output_path, output_filename)
            median_path = process.median_filter(mask_path, specific_output_path, output_filename,remove_outliers_filter_size)
            norm_path = process.normalizar_stack(median_path, input_tiff, specific_output_path, output_filename)
            process.process_stack_rois(norm_path, specific_output_path, roi_size=roi_size, frame_range=frame_range, output_filename=output_filename)

            print(f"Processed {filename} \n")

    print(f"""\033[36mSummary:\033[0m

     (1) Difference of gauss
     (2) BinaryMask
     (3) Remove Outliers using median filter
     (4) Normalize
     (5) Rois, log2(p/f0) \n""")

    print(f"All files processed. Results saved in {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image stack with Difference of Gaussians and ROIs.")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing .tif files')
    parser.add_argument('--output_folder', type=str, required=True, help='Directory where results will be saved')
    parser.add_argument('--sigma1', type=float, required=True, help='Sigma1 for Difference of Gaussians')
    parser.add_argument('--sigma2', type=float, required=True, help='Sigma2 for Difference of Gaussians')
    parser.add_argument('--remove_outliers_filter_size', type=int, required=True, help='Size for median filter to remove outliers')
    parser.add_argument('--roi_size', type=int, nargs=2, default=(20, 20), help='Size of the ROI as two integers (width height)')
    parser.add_argument('--frame_range', type=int, nargs=2, default=(0, 1), help='Size of the frame range as two integers (start,end)')

    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.sigma1, args.sigma2, args.remove_outliers_filter_size, tuple(args.roi_size), tuple(args.frame_range))