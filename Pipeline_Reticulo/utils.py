import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter
import cv2
from skimage.filters import threshold_triangle
import os
import tifffile
import argparse
from scipy.ndimage import median_filter
import time
import multiprocessing
from multiprocessing import Pool, cpu_count

class process:

        #-------------------------------- PASO 1 DoG --------------------------------------
    @staticmethod
    def difference_of_gaussians(image_stack, sigma1, sigma2):
        if sigma1 > 0:
            blur1 = gaussian_filter(image_stack, sigma=sigma1)
        else:
            blur1 = image_stack

        blur2 = gaussian_filter(image_stack, sigma=sigma2)
        dog = blur1 - blur2
        return dog

    @staticmethod
    def process_stack(stack_path, output_path, sigma1, sigma2, output_filename):
        image_stack = tiff.imread(stack_path)

        if image_stack.ndim == 3:
            frames, height, width = image_stack.shape
            channels = 1
            image_stack = image_stack[:, np.newaxis, :, :]
        elif image_stack.ndim == 4:
            frames, channels, height, width = image_stack.shape
        else:
            raise ValueError("Unexpected number of dimensions in the image stack")
        
        if channels > 1:
            channel_data = image_stack[:, 1, :, :].astype(np.float32)
        else:
            channel_data = image_stack[:, 0, :, :].astype(np.float32)

        dog_stack = process.difference_of_gaussians(channel_data, sigma1, sigma2)

        dog_output_path = os.path.join(output_path, f"{output_filename}_Paso_1_DoG_sigma{sigma1}_{sigma2}.tiff")
        tiff.imwrite(dog_output_path, dog_stack.astype(np.float32), imagej=True, metadata={'axes': 'TYX'})
        print(f"\033[92mDoG Completed....result saved in {dog_output_path}\033[0m")
        print()
        return dog_output_path

    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------


    #----------------------- PASO 2 Mascara Binaria del resultado de DoG --------------------------------------
    @staticmethod
    def process_dog_stack(dog_stack_path, output_path, output_filename):
        dog_stack = tiff.imread(dog_stack_path)
        frames, height, width = dog_stack.shape
        
        mask_stack = np.zeros_like(dog_stack)
        
        for frame in range(frames):
            image = dog_stack[frame, :, :]
            thresh_val = threshold_triangle(image)
            _, binary_mask = cv2.threshold(image, thresh_val, 65535, cv2.THRESH_BINARY)
            mask_stack[frame, :, :] = binary_mask

        mask_output_path = os.path.join(output_path, f"{output_filename}_Paso_2_DoG_Mask.tiff")
        tiff.imwrite(mask_output_path, mask_stack.astype(np.uint16), imagej=True)
        print(f"\033[92mMask Completed...result saved in {mask_output_path}\033[0m")
        print()
        return mask_output_path

    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------

    #----------------------- PASO 3 Remove Outliers ---------------------------

    @staticmethod
    def median_filter(stack, output_path, output_filename,remove_outliers_filter_size):
        stack_array = tiff.imread(stack)
        filtered_array = median_filter(stack_array, size=remove_outliers_filter_size)

        median_output_path = os.path.join(output_path, f"{output_filename}_Paso_3_Remove_outliers.tiff")
        tiff.imwrite(median_output_path, filtered_array.astype(np.uint16), imagej=True)
        print(f"\033[92mRemove Outliers Completed...result saved in {median_output_path}\033[0m")
        print()
        return median_output_path

    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------

    #----------------------------- PASO 4 Normalizar resultado Mascara Binaria ---------------------------
    @staticmethod
    def normalizar_stack(mask_path, original_stack_path, output_path, output_filename):
        #Abre el tiff resultante del paso previo (Normalizacion)
        mask_stack = tiff.imread(mask_path)

        max_value = np.iinfo(mask_stack.dtype).max
        #print(f"VALOR MASK_STACK: {mask_stack} \n")
        #print(f"VALOR MAX_VALUE: {max_value} \n")
        
        normalizado = mask_stack / max_value

        #print(f"VALOR NORMALIZADO: {normalizado} \n")

        #Abre el tiff original
        original_stack = tiff.imread(original_stack_path)

        if original_stack.ndim == 3:
            channel_0_original_stack = original_stack
            #print(f"Tiff con un canal \n")
        elif original_stack.ndim == 4:
            channel_0_original_stack = original_stack[:, 0, :, :]
        else:
            raise ValueError("Unexpected number of dimensions in the original stack")

        stack_result = normalizado * channel_0_original_stack
        
        #print(f"RESULTADO STACK_RESULT: {stack_result} \n")

        norm_output_path = os.path.join(output_path, f"{output_filename}_Paso_4_normalized.tif")
        tiff.imwrite(norm_output_path, stack_result.astype(np.float32), imagej=True, metadata={'axes': 'TYX'})
        print(f"\033[92mNormalize Completed...stack saved in {norm_output_path}\033[0m")
        print()
        return norm_output_path

    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------


    #----------------- PASO 5 Funciones necesarias para crear ROIS y aplicar log2(p/f0) ----------------
    @staticmethod
    def create_rectangle_roi(x, y, width, height, image_shape):
        img_height, img_width = image_shape
        # Ajustar el ancho y alto para que el ROI no se salga del área de la imagen
        width = min(width, img_width - x)
        height = min(height, img_height - y)
        return (x, y, width, height)

    @staticmethod
    def create_mask_from_roi(roi, image_shape):
        x, y, width, height = roi
        mask = np.zeros(image_shape, dtype=bool)
        mask[y:y + height, x:x + width] = True
        return mask

    @staticmethod
    def generate_rois(image_shape, roi_size):
        height, width = image_shape
        rois = []
        resto = width%roi_size[0]
        #print(resto)
        for y in range(0, height, roi_size[1]):
            for x in range(0, width, roi_size[0]):
                current_width = min(roi_size[0], width - x)  #width - x => 322 - 320 = 2
                current_height = min(roi_size[1], height - y) 
                roi = process.create_rectangle_roi(x, y, roi_size[0], current_height, (height, width))
                rois.append(roi) 
            if(resto != 0):
                roi = process.create_rectangle_roi(x+current_width,y,current_width,current_height,(height, width))
                rois.append(roi)

        #print(rois)
        return rois

    # Obtener los vecinos de un ROI dado
    @staticmethod
    def get_neighbor_indices(roi_index, rois_per_row, rois_per_col):
        neighbors = []
        row = roi_index // rois_per_row #por ejemplo index = rois_per_row
        col = roi_index % rois_per_row # por ejemplo roi_index
        # Vecinos posibles (arriba, abajo, izquierda, derecha, diagonal) 
        for d_row in [-1, 0, 1]:                                 #      *   *   *   *
            for d_col in [-1, 0, 1]:                             #      *   *   *   *
                if d_row == 0 and d_col == 0:                    #      *   *   *   *
                    continue  # Saltar el ROI actual
                neighbor_row = row + d_row
                neighbor_col = col + d_col
                if 0 <= neighbor_row < rois_per_col and 0 <= neighbor_col < rois_per_row:
                    neighbor_index = neighbor_row * rois_per_row + neighbor_col
                    neighbors.append(neighbor_index)

        return neighbors
    

    @staticmethod
    def calculate_f0_per_roi(avg_fluorescence_array, frame_range, rois_per_row, rois_per_col, num_rois, channel):
        f0_per_roi = np.zeros(num_rois)

        for roi_idx in range(num_rois):
            roi_fluorescence_values = []

            for frame in range(frame_range[0], frame_range[1] + 1):
                #print("FRAME:", frame)
                avg = avg_fluorescence_array[frame, channel, roi_idx]
                neighbors = process.get_neighbor_indices(roi_idx, rois_per_row, rois_per_col)
                neighbor_values = [avg_fluorescence_array[frame, channel, n] for n in neighbors]
                all_values = neighbor_values + [avg]

                all_values = np.where(np.array(all_values) == 0, np.nan, all_values)

                #Se usa nanmean ya que esta funcion realiza el promedio ignorando los nans.
                avg_with_neighbors = np.nanmean(all_values) 
                roi_fluorescence_values.append(avg_with_neighbors)

            f0_per_roi[roi_idx] = np.mean(roi_fluorescence_values)
        return f0_per_roi

    @staticmethod
    def process_frame_range(args):
        """Procesa un rango de frames."""
        frame_start, frame_end, rois, stack, num_channels, height, width = args
        local_fluorescence = []
        for frame in range(frame_start, frame_end):
            if num_channels == 1:
                image = stack[frame]
            else:
                image = stack[frame, 0]

            roi_avg = []
            for roi in rois:
                mask = process.create_mask_from_roi(roi, (height, width))
                roi_pixels = image[mask]
                roi_pixels = roi_pixels[roi_pixels > 0]

                if roi_pixels.size > 0:
                    roi_avg_value = np.mean(roi_pixels)
                    roi_avg.append(roi_avg_value)
                else:
                    roi_avg.append(0)

            for channel in range(num_channels):
                local_fluorescence.append((frame, channel, roi_avg))
        return local_fluorescence

    @staticmethod
    def process_chunk(args):
        """Procesa un rango de frames en paralelo."""
        start, end, rois, stack, f0_per_roi, num_channels, height, width, channel = args
        new_stack = stack.copy()
        for frame in range(start, end):
            for roi_index, roi in enumerate(rois):
                f0 = f0_per_roi[roi_index]
                mask = process.create_mask_from_roi(roi, (height, width))

                if new_stack.ndim == 3:
                    roi_pixels = new_stack[frame][mask]
                elif new_stack.ndim == 4:
                    roi_pixels = new_stack[frame, channel][mask]

                if f0 > 0:
                    safe_roi_pixels = np.where(roi_pixels != 0, roi_pixels / f0, np.nan)
                    if new_stack.ndim == 3:
                        new_stack[frame][mask] = np.log2(safe_roi_pixels)
                    elif new_stack.ndim == 4:
                        new_stack[frame, channel][mask] = np.log2(safe_roi_pixels)
                else:
                    safe_roi_pixels = np.nan
                    if new_stack.ndim == 3:
                        new_stack[frame][mask] = safe_roi_pixels
                    elif new_stack.ndim == 4:
                        new_stack[frame, channel][mask] = safe_roi_pixels
        return new_stack[start:end]




    @staticmethod
    def process_stack_rois(input_tiff, output_path, roi_size, frame_range, output_filename):
        with tifffile.TiffFile(input_tiff) as tif:
            stack = tif.asarray()

        if stack.ndim == 3:
            num_frames, height, width = stack.shape
            num_channels = 1
        elif stack.ndim == 4:
            num_frames, num_channels, height, width = stack.shape
        else:
            raise ValueError("Unexpected number of dimensions in the stack")
        print(f"STACK SHAPE: {stack.shape} \n")
        rois = process.generate_rois((height, width), roi_size)
        rois_per_row = (width + roi_size[0] - 1) // roi_size[0]
        rois_per_col = (height + roi_size[1] - 1) // roi_size[1]
        num_rois = len(rois)
        
        #num_workers = min(num_frames, multiprocessing.cpu_count())
        num_workers = 64 
        print(f"NUMERO DE WORKERS: {num_workers} \n") 
        chunk_size = (num_frames + num_workers - 1) // num_workers
        print(f"CHUNK SIZE: {chunk_size} \n")
        frame_ranges = [(i, min(i + chunk_size, num_frames)) for i in range(0, num_frames, chunk_size)]
        print(f"FRAME_RANGES: {frame_ranges} \n")

        # Argumentos para los procesos
        args = [(start, end, rois, stack, num_channels, height, width) for start, end in frame_ranges]
        #print(f"ARGS: {args} \n")

        # Ejecutar en paralelo
        inicio = time.time()
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(process.process_frame_range, args) #pool.map(func , iterable)
        fin = time.time()

        # Combinar resultados
        roi_fluorescence = [item for sublist in results for item in sublist]
        print(f"Tiempo de ejecución paralelo: {fin - inicio:.4f} segundos")


        avg_fluorescence_array = np.zeros((num_frames, num_channels, num_rois), dtype=np.float32)

        for frame, channel, averages in roi_fluorescence:
            avg_fluorescence_array[frame, channel, :len(averages)] = averages

        channel = 0
        
        inicio = time.time()
        f0_per_roi = process.calculate_f0_per_roi(avg_fluorescence_array, frame_range, rois_per_row, rois_per_col, num_rois, channel)
        fin = time.time()
        print(f"Tiempo de ejecución funcion F0: {fin - inicio:.4f} segundos")


        inicio = time.time()
        args = [
        (start, end, rois, stack, f0_per_roi, num_channels, height, width, channel)
        for start, end in frame_ranges
        ]


        # Ejecutar en paralelo
        with multiprocessing.Pool(processes=num_workers) as pool:
            processed_chunks = pool.map(process.process_chunk, args)

        new_stack = np.concatenate(processed_chunks, axis=0)
        fin = time.time()
        print(f"Tiempo de ejecución calculo despues de F0: {fin - inicio:.4f} segundos")

        rois_output_path = os.path.join(output_path, f"{output_filename}_Paso_5_rois.tif")
        tifffile.imwrite(rois_output_path, new_stack, imagej=True)
        print(f"\033[93mProcess Completed...stack saved in {rois_output_path}\033[0m")
        print()

        return rois_output_path

