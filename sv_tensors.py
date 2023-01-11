
import cv2
import math
import matplotlib
import matplotlib.image
import numpy as np
import os
import shutil
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy


#sv_data = os.path.join('/home', 'intern', 'sv_data')
sv_data = os.path.join('F:', 'data', 'physics_based')

folder_0090_0001 = os.path.join(sv_data, '0090_0001', '0090_0001')
folder_0091_0001 = os.path.join(sv_data, '0091_0001', '0091_0001')
folder_0092_0001 = os.path.join(sv_data, '0092_0001', '0092_0001')
folder_0093_0001 = os.path.join(sv_data, '0093_0001', '0093_0001')
folder_0094_0001 = os.path.join(sv_data, '0094_0001', '0094_0001')
folder_0095_0001 = os.path.join(sv_data, '0095_0001', '0095_0001')
folder_0140_2001 = os.path.join(sv_data, '0140_2001', '0140_2001')
folder_0142_1001 = os.path.join(sv_data, '0142_1001', '0142_1001')
folder_0149_1001 = os.path.join(sv_data, '0149_1001', '0149_1001')
folder_0154_0001 = os.path.join(sv_data, '0154_0001', '0154_0001')

models_0090_0001 = os.path.join(folder_0090_0001, 'Models')
models_0091_0001 = os.path.join(folder_0091_0001, 'Models')
models_0092_0001 = os.path.join(folder_0092_0001, 'Models')
models_0093_0001 = os.path.join(folder_0093_0001, 'Models')
models_0094_0001 = os.path.join(folder_0094_0001, 'Models')
models_0095_0001 = os.path.join(folder_0095_0001, 'Models')
models_0140_2001 = os.path.join(folder_0140_2001, 'Models')
models_0142_1001 = os.path.join(folder_0142_1001, 'Models')
models_0149_1001 = os.path.join(folder_0149_1001, 'Models')
models_0154_0001 = os.path.join(folder_0154_0001, 'Models')

simulations_0090_0001 = os.path.join(folder_0090_0001, 'Simulations', '0090_0001')
simulations_0091_0001 = os.path.join(folder_0091_0001, 'Simulations', '0091_0001')
simulations_0092_0001 = os.path.join(folder_0092_0001, 'Simulations', '0092_0001')
simulations_0093_0001 = os.path.join(folder_0093_0001, 'Simulations', '0093_0001')
simulations_0094_0001 = os.path.join(folder_0094_0001, 'Simulations', '0094_0001')
simulations_0095_0001 = os.path.join(folder_0095_0001, 'Simulations', '0095_0001')
simulations_0140_2001 = os.path.join(folder_0140_2001, 'Simulations', '0140_2001')
simulations_0142_1001 = os.path.join(folder_0142_1001, 'Simulations', '0142_1001')
simulations_0149_1001 = os.path.join(folder_0149_1001, 'Simulations', '0149_1001')
simulations_0154_0001 = os.path.join(folder_0154_0001, 'Simulations', '0154_0001')

extracted_data = os.path.join(sv_data, 'extracted_data')
extracted_models = os.path.join(extracted_data, 'Models')
extracted_masks = os.path.join(extracted_models, 'masks')
extracted_tensors = os.path.join(extracted_models, 'tensor_masks')
extracted_simulations = os.path.join(extracted_data, 'Simulations')

vtp_model_0090_0001 = os.path.join(models_0090_0001, '0090_0001.vtp')
vtp_model_0091_0001 = os.path.join(models_0091_0001, '0091_0001.vtp')
vtp_model_0092_0001 = os.path.join(models_0092_0001, '0092_0001.vtp')
vtp_model_0093_0001 = os.path.join(models_0093_0001, '0093_0001.vtp')
vtp_model_0094_0001 = os.path.join(models_0094_0001, '0094_0001.vtp')
vtp_model_0095_0001 = os.path.join(models_0095_0001, '0095_0001.vtp')
vtp_model_0140_2001 = os.path.join(models_0140_2001, '0140_2001.vtp')
vtp_model_0142_1001 = os.path.join(models_0142_1001, '0142_1001.vtp')
vtp_model_0149_1001 = os.path.join(models_0149_1001, '0149_1001.vtp')
vtp_model_0154_0001 = os.path.join(models_0154_0001, '0154_0001.vtp')

model_0090_0001_masks = os.path.join(extracted_masks, '0090_0001')
model_0091_0001_masks = os.path.join(extracted_masks, '0091_0001')
model_0092_0001_masks = os.path.join(extracted_masks, '0092_0001')
model_0093_0001_masks = os.path.join(extracted_masks, '0093_0001')
model_0094_0001_masks = os.path.join(extracted_masks, '0094_0001')
model_0095_0001_masks = os.path.join(extracted_masks, '0095_0001')
model_0140_2001_masks = os.path.join(extracted_masks, '0140_2001')
model_0142_1001_masks = os.path.join(extracted_masks, '0142_1001')
model_0149_1001_masks = os.path.join(extracted_masks, '0149_1001')
model_0154_0001_masks = os.path.join(extracted_masks, '0154_0001')

model_0090_0001_tensor = os.path.join(extracted_tensors, '0090_0001.pt')
model_0091_0001_tensor = os.path.join(extracted_tensors, '0091_0001.pt')
model_0092_0001_tensor = os.path.join(extracted_tensors, '0092_0001.pt')
model_0093_0001_tensor = os.path.join(extracted_tensors, '0093_0001.pt')
model_0094_0001_tensor = os.path.join(extracted_tensors, '0094_0001.pt')
model_0095_0001_tensor = os.path.join(extracted_tensors, '0095_0001.pt')
model_0140_2001_tensor = os.path.join(extracted_tensors, '0140_2001.pt')
model_0142_1001_tensor = os.path.join(extracted_tensors, '0142_1001.pt')
model_0149_1001_tensor = os.path.join(extracted_tensors, '0149_1001.pt')
model_0154_0001_tensor = os.path.join(extracted_tensors, '0154_0001.pt')

vtp_models = [vtp_model_0090_0001, vtp_model_0091_0001, vtp_model_0092_0001, vtp_model_0093_0001,
              vtp_model_0094_0001, vtp_model_0095_0001, vtp_model_0154_0001, vtp_model_0149_1001,
              vtp_model_0140_2001, vtp_model_0142_1001]

tensors = [model_0090_0001_tensor, model_0091_0001_tensor, model_0092_0001_tensor, model_0093_0001_tensor,
           model_0094_0001_tensor, model_0095_0001_tensor, model_0154_0001_tensor, model_0149_1001_tensor,
           model_0140_2001_tensor, model_0142_1001_tensor]

masks = [model_0090_0001_masks, model_0091_0001_masks, model_0092_0001_masks, model_0093_0001_masks,
           model_0094_0001_masks, model_0095_0001_masks, model_0154_0001_masks, model_0149_1001_masks,
           model_0140_2001_masks, model_0142_1001_masks]


def clean_dir(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def save_masks(np_array, dir_path):
    masks_count = 0
    os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(dir_path) and os.path.isdir(dir_path) :
        clean_dir(dir_path)
        for i in range(0, np_array.shape[0]):
            mask_path = os.path.join(dir_path, f'mask_{np_array.shape[0]-i-1}.png')
            matplotlib.image.imsave(mask_path, np_array[i], cmap="gray")
            masks_count += 1

    return masks_count


def vtp_to_pngs(vtp_path, pngs_path, spacing=0.05, inval=1, outval=0, cast_float32=True):
    masks_count = 0
    if os.path.exists(vtp_path) and os.path.isfile(vtp_path):
        #tensor_folder, file_name = os.path.split(pngs_path)
        os.makedirs(pngs_path, exist_ok=True)
        spacings = [spacing, spacing, spacing]

        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_path)
        reader.Update()
        data = reader.GetOutput()
        bounds = data.GetBounds()
        dim = [0] * 3
        for i in range(3):
            dim[i] = int(math.ceil((bounds[i * 2 + 1] - bounds[i * 2]) / spacings[i])) + 1
            if dim[i] < 1:
                dim[i] = 1

        origin = [0] * 3
        # NOTE: I am not sure whether or not we had to add some offset!
        origin[0] = bounds[0]  # + spacings[0] / 2
        origin[1] = bounds[2]  # + spacings[1] / 2
        origin[2] = bounds[4]  # + spacings[2] / 2

        # Convert the VTK array to vtkImageData
        whiteImage = vtk.vtkImageData()
        whiteImage.SetDimensions(dim)
        whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)
        whiteImage.SetSpacing(spacings)
        whiteImage.SetOrigin(origin)
        whiteImage.GetPointData()
        whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # fill the image with foreground voxels:
        count = whiteImage.GetNumberOfPoints()
        for i in range(count):
            whiteImage.GetPointData().GetScalars().SetTuple1(i, inval)

        # polygonal data -. image stencil:
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetTolerance(0)  # important if extruder.SetVector(0, 0, 1) !!!
        pol2stenc.SetInputData(data)
        pol2stenc.SetOutputOrigin(origin)
        pol2stenc.SetOutputSpacing(spacings)
        pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
        pol2stenc.Update()

        # cut the corresponding white image and set the background:
        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(whiteImage)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(outval)
        imgstenc.Update()

        np_array = vtk_to_numpy(imgstenc.GetOutput().GetPointData().GetScalars())
        np_array = np_array.reshape(dim[2], dim[1], dim[0])

        print(f'bounds : {bounds}')
        print(f'dim : {dim}')
        print(f'np_array.shape : {np_array.shape}')
        print(f'np_array.max() : {np_array.max()}')
        print(f'np_array.min() : {np_array.min()}')

        masks_count = save_masks(np_array, pngs_path)
        print(f'masks_count : {masks_count}')

    return masks_count


def get_np_tensor_from_pngs(png_masks_path):
    np_tensor = None
    if os.path.exists(png_masks_path) and os.path.isdir(png_masks_path) and len(os.listdir(png_masks_path)) > 0:
        for png_mask in sorted(os.listdir(png_masks_path)):
            png_mask_path = os.path.join(png_masks_path, png_mask)
            if os.path.exists(png_mask_path) and os.path.isfile(png_mask_path) and png_mask.endswith('.png'):
                mask = np.moveaxis(cv2.imread(png_mask_path, 0), 0, 1)
                if np_tensor is None:
                    np_tensor = np.empty((0, mask.shape[0], mask.shape[1]), dtype=np.uint8) # np.float32)
                np_tensor = np.append(np_tensor, np.expand_dims(mask, axis=0), axis=0)

    return np_tensor


def create_tensors_from_pngs(pngs_path, dims=[128, 128, 128]):
    tensors_paths = []
    if os.path.exists(pngs_path) and os.path.isdir(pngs_path) and len(os.listdir(pngs_path)) > 0:
        for tensor_folder in os.listdir(pngs_path):
            tensor_path = os.path.join(pngs_path, tensor_folder)
            if os.path.exists(tensor_path) and os.path.isdir(tensor_path):
                png_masks_path = os.path.join(tensor_path, 'png_masks')
                np_tensor = get_np_tensor_from_pngs(png_masks_path)



    return tensors_paths


def edit_png_masks(tensor_path):
    final_masks_count = 0
    if os.path.exists(tensor_path) and os.path.isdir(tensor_path):
        png_masks_path = os.path.join(tensor_path, 'png_masks')
        if os.path.exists(png_masks_path) and os.path.isdir(png_masks_path) and len(os.listdir(png_masks_path)) > 0:
            edited_masks_path = os.path.join(tensor_path, 'final_masks')
            os.makedirs(edited_masks_path, exist_ok=True)
            if os.path.exists(edited_masks_path) and os.path.isdir(edited_masks_path):
                for png_mask_file in os.listdir(png_masks_path):
                    png_mask_path = os.path.join(png_masks_path, png_mask_file)
                    if os.path.exists(png_mask_path) and os.path.isfile(png_mask_path) and png_mask_path.endswith('.png'):
                        png_mask = cv2.imread(png_mask_path)
                        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return final_masks_count


def crop_to_square(tensor_path, cropped_dir=None):
    cropped_npy_path = None
    if os.path.exists(tensor_path) and os.path.isfile(tensor_path):
        with open(tensor_path, 'rb') as f:
            np_tensor = np.load(f)
        print(f'np_tensor.shape: {np_tensor.shape}')
        padding_size = int((np_tensor.shape[1] - np_tensor.shape[2])/2)
        np_tensor = np_tensor[:, padding_size:padding_size+np_tensor.shape[2], :]
        file_folder, file_name = os.path.split(tensor_path)
        if cropped_dir is None:
            cropped_dir = file_folder
        else:
            os.makedirs(cropped_dir, exist_ok=True)
        cropped_npy_path = os.path.join(cropped_dir, f'cropped_{file_name}')
        if os.path.exists(cropped_npy_path) and os.path.isfile(cropped_npy_path):
            os.remove(cropped_npy_path)
        np.save(cropped_npy_path, np_tensor)

    return cropped_npy_path


def save_png_masks_as_np_tensors(masks_dirs_path):
    np_tensors_paths = []
    if os.path.exists(masks_dirs_path) and os.path.isdir(masks_dirs_path) and len(os.listdir(masks_dirs_path)) > 0:
        for npy_dir in sorted(os.listdir(masks_dirs_path)):
            pngs_path = os.path.join(masks_dirs_path, npy_dir, 'reordered_pngs')
            np_tensor = get_np_tensor_from_pngs(pngs_path)
            npy_path = os.path.join(masks_dirs_path, npy_dir, f'{npy_dir}.npy')
            if os.path.exists(npy_path) and os.path.isfile(npy_path):
                os.remove(npy_path)
            np.save(npy_path, np_tensor)
            np_tensors_paths.append(npy_path)

    return np_tensors_paths


def save_cropped_npy_as_pngs(cropped_npy_path, cropped_pngs_path=None):
    saved_pngs_count = 0
    if os.path.exists(cropped_npy_path) and os.path.isfile(cropped_npy_path):
        with open(cropped_npy_path, 'rb') as f:
            cropped_np_tensor = np.load(f)
        print(f'cropped_np_tensor.shape: {cropped_np_tensor.shape}')
        if cropped_pngs_path is None:
            npy_dir, npy_file = os.path.split(cropped_npy_path)
            cropped_pngs_path = os.path.join(npy_dir, 'cropped_png')
            if os.path.exists(cropped_pngs_path) and os.path.isdir(cropped_pngs_path) and len(os.listdir(cropped_pngs_path)) > 0:
                clean_dir(cropped_pngs_path)
            os.makedirs(cropped_pngs_path, exist_ok=True)

        for i in range(cropped_np_tensor.shape[0]):
            cropped_png = cropped_np_tensor[i, :, :]
            mask_number = str(i)
            while len(mask_number) < 3:
                mask_number = "0" + mask_number
            cropped_png_file = os.path.join(cropped_pngs_path, f'mask_{mask_number}.png')
            matplotlib.image.imsave(cropped_png_file, cropped_png, cmap="gray")
            saved_pngs_count += 1

    return saved_pngs_count


def reorder_masks(tensors_masks_dir, reordered_dir_name):
    reodered_dirs_paths = []
    prefix = 'mask_'
    postfix = '.png'
    if os.path.exists(tensors_masks_dir) and os.path.isdir(tensors_masks_dir) and len(os.listdir(tensors_masks_dir)) > 0:
        for tensor_mask_dir in os.listdir(tensors_masks_dir):
            png_masks_dir = os.path.join(tensors_masks_dir, tensor_mask_dir, 'png_masks')
            if os.path.exists(png_masks_dir) and os.path.isdir(png_masks_dir) and len(os.listdir(png_masks_dir)) > 0:
                reordered_masks_dir = os.path.join(tensors_masks_dir, tensor_mask_dir, reordered_dir_name)
                os.makedirs(reordered_masks_dir, exist_ok=True)
                for png_mask in os.listdir(png_masks_dir):
                    mask_path = os.path.join(png_masks_dir, png_mask)
                    new_mask_number = png_mask[5:-4]
                    while len(new_mask_number) < 3:
                        new_mask_number = "0" + new_mask_number
                    new_mask_name = f'{prefix}{new_mask_number}{postfix}'
                    new_mask_path = os.path.join(reordered_masks_dir, new_mask_name)
                    shutil.copyfile(mask_path, new_mask_path)

            reordered_dirs_paths.append(reordered_masks_dir)
    return reordered_dirs_paths


def get_cropped_npys(tensors_dir, prefix='cropped_', suffix = '.npy'):
    cropped_npys = []
    if os.path.exists(tensors_dir) and os.path.isdir(tensors_dir) and len(os.listdir(tensors_dir)) > 0:
        for tensor_dir in os.listdir(tensors_dir):
            tensor_dir_path = os.path.join(tensors_dir, tensor_dir)
            if os.path.exists(tensor_dir_path) and os.path.isdir(tensor_dir_path) and len(os.listdir(tensor_dir_path)) > 0:
                for path_ in os.listdir(tensor_dir_path):
                    if path_.startswith(prefix) and path_.endswith(suffix):
                        cropped_npy_path = os.path.join(tensor_dir_path, path_)
                        if os.path.exists(cropped_npy_path) and os.path.isfile(cropped_npy_path):
                            cropped_npys.append(cropped_npy_path)
    return cropped_npys


def save_npy_to_pngs(npy, pngs_dir):
    pngs_count = 0
    if type(npy) is np.ndarray and len(npy.shape) == 3 and npy.shape[0] > 0 and npy.shape[1] > 0 and npy.shape[2] > 0:
        if os.path.exists(pngs_dir) and os.path.isdir(pngs_dir) and len(os.listdir(pngs_dir)) > 0:
            clean_dir(pngs_dir)
        os.makedirs(pngs_dir, exist_ok=True)
        for i in range(npy.shape[0]):
            png = npy[i, :, :]
            mask_number = str(i)
            while len(mask_number) < 3:
                mask_number = "0" + mask_number
            png_file = os.path.join(pngs_dir, f'mask_{mask_number}.png')
            matplotlib.image.imsave(png_file, png, cmap="gray")
            pngs_count += 1
    else:
        print(f'type(npy) : {type(npy)}')
        print(f'npy.shape : {npy.shape}')
        print(f'len(npy.shape) : {len(npy.shape)}')

    return pngs_count


def modify_cropped_npys(cropped_npys, size=128, old_prefix = 'cropped_', new_prefix='modified_', pngs_dir_name='modified_pngs'):
    modified_npys = []
    size_wo_padding = size-1

    for cropped_npy_path in cropped_npys:
        path_, file_name = os.path.split(cropped_npy_path)
        cropped_npy_number = file_name[len(old_prefix):-4]
        with open(cropped_npy_path, 'rb') as f:
            modified_npy = np.load(f)

        if cropped_npy_number == '0090_0001':
            x_crop = y_crop = 38
            x_y_crop = 44
            modified_npy = modified_npy[:, modified_npy.shape[1]//6 : modified_npy.shape[1]-y_crop-modified_npy.shape[1]//6, x_crop+modified_npy.shape[2]//3:]
            for i in range(x_y_crop):
                for j in range(x_y_crop-i-1, -1, -1):
                    modified_npy[:, i, j] = 0

        elif cropped_npy_number == '0091_0001':
            top_crop = bottom_crop = 6
            left_crop = 12
            modified_npy = modified_npy[:, top_crop+modified_npy.shape[1]//4:modified_npy.shape[1]-bottom_crop-modified_npy.shape[1]//4, left_crop+modified_npy.shape[2]//2:]

        elif cropped_npy_number == '0092_0001':
            top_crop = 18 # 14
            bottom_crop = 14
            left_crop = 32 # 28
            modified_npy = modified_npy[:, top_crop+modified_npy.shape[1]//4:modified_npy.shape[1]-bottom_crop-modified_npy.shape[1]//4, left_crop+modified_npy.shape[2]//2:]
            dim = (size, size)
            new_modified_npy = np.empty((modified_npy.shape[0], size, size), dtype=np.uint8)
            for i in range(modified_npy.shape[0]):
                # resize image
                new_modified_npy[i, :, :] = cv2.resize(modified_npy[i, :, :], dim, interpolation=cv2.INTER_NEAREST)

            modified_npy = new_modified_npy

        elif cropped_npy_number == '0093_0001':
            top_crop = left_crop = 10
            modified_npy = modified_npy[:,top_crop+7*modified_npy.shape[1]//23:modified_npy.shape[1]-4*modified_npy.shape[1]//23, left_crop+11*modified_npy.shape[2]//23:]
            dim = (size, size)
            new_modified_npy = np.empty((modified_npy.shape[0], size, size), dtype=np.uint8)
            for i in range(modified_npy.shape[0]):
                # resize image
                new_modified_npy[i, :, :] = cv2.resize(modified_npy[i, :, :], dim, interpolation=cv2.INTER_NEAREST)

            modified_npy = new_modified_npy

        elif cropped_npy_number == '0140_2001':
            bottom_crop = right_crop = 65
            modified_npy = modified_npy[:, :modified_npy.shape[1]-bottom_crop, :modified_npy.shape[2]-right_crop]

        elif cropped_npy_number == '0142_1001':
            bottom_crop = right_crop = 8
            modified_npy = modified_npy[:, :modified_npy.shape[1] - bottom_crop, :modified_npy.shape[2] - right_crop]
            top_padding = 37
            modified_npy[:, :top_padding, :] = 0

        elif cropped_npy_number == '0149_1001':
            bottom_crop = right_crop = 7
            modified_npy = modified_npy[:, :modified_npy.shape[1] - bottom_crop, :modified_npy.shape[2] - right_crop]

        else:
            print(f'Wrong .npy: "{cropped_npy_number}" !')

        modified_pngs_dir = os.path.join(path_, pngs_dir_name)
        saved_pngs = save_npy_to_pngs(modified_npy, modified_pngs_dir)
        print(f'saved_pngs : {saved_pngs}')
        modified_npy_path = os.path.join(path_, f'{new_prefix}{cropped_npy_number}.npy')
        if os.path.exists(modified_npy_path) and os.path.isfile(modified_npy_path):
            os.remove(modified_npy_path)
        np.save(modified_npy_path, modified_npy)
        modified_npys.append(modified_npy_path)

    return modified_npys


def make_pt_tensors_from_npys(npys, pt_dir=None):
    prefix = 'modified_'
    suffix = '.npy'
    pt_tensors = []
    if len(npys) > 0:
        for npy in npys:
            dir_path, file_name = os.path.split(npy)
            if pt_dir is not None:
                dir_path = pt_dir
            pt_name = file_name[len(prefix):-len(suffix)]
            pt_path = os.path.join(dir_path, f'{pt_name}.pt')
            with open(npy, 'rb') as f:
                np_tensor = np.load(f)
            pt_tensor = torch.from_numpy(np_tensor)
            print(f'pt_path: {pt_path}')
            print(f'pt_tensor.size(): {pt_tensor.size()}')
            torch.save(pt_tensor, pt_path)
            pt_tensors.append(pt_path)

    return pt_tensors


if __name__ == '__main__':
    """
    vtp_to_pngs(vtp_model_0090_0001, model_0090_0001_masks, spacing=0.025)
    np_tensor = get_np_tensor_from_pngs('F:\\data\\physics_based\\extracted_data\\Models\\tensors_masks\\0090_0001\\png_masks')

    reordered_dirs_paths = reorder_masks(tensors_dir, "reordered_pngs")
    print(f'reordered_dirs_paths: {reordered_dirs_paths}')

    np_tensors_paths = save_png_masks_as_np_tensors(tensors_dir)
    print(f'np_tensors_paths: {np_tensors_paths}')
    for tensor_path in np_tensors_paths:
        saved_pngs_count = save_cropped_npy_as_pngs(crop_to_square(tensor_path, cropped_dir=None), cropped_pngs_path=None)
        print(f'tensor_path: {tensor_path}')
        print(f'saved_pngs_count: {saved_pngs_count}')
    """

    tensors_dir = 'F:\\data\\physics_based\\extracted_data\\Models\\tensors_masks'

    cropped_npys = get_cropped_npys(tensors_dir)
    print(f'cropped_npys : {cropped_npys}')
    modified_npys = modify_cropped_npys(cropped_npys)
    print(f'modified_npys : {modified_npys}')

    pt_tensors = make_pt_tensors_from_npys(modified_npys)
    print(f'pt_tensors : {pt_tensors}')