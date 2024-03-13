#Author: Yaojian Chen

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import math
import time
import copy

def load_image(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)  # Add batch dimension

def save_tensor_as_image(tensor, file_path, format='PNG'):
    """Saves a torch tensor as an image file.
        2D or 3D tensor without batch dimention
    
    Args:
        tensor (torch.Tensor): The image tensor to save.
        file_path (str): The file path to save the image to.
        format (str): The image format (e.g., 'JPEG', 'PNG'). Default is 'PNG'.
    """
    # Check if the tensor is in the range [0, 1]
    if torch.max(tensor) > 1.0:
        tensor = tensor / 255.0

    # Convert the tensor to PIL Image
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)

    # Save the image
    img.save(file_path, format=format)

def mirror_padded_convolution(input, kernel):
    # Ensure input is a 4D tensor: [batch_size, channels, height, width]
    if input.dim() != 4:
        raise ValueError("Input must be a 4D tensor")

    input_channels = input.size(1)
    kernel_height, kernel_width = kernel.size(2), kernel.size(3)
    pad_top = min(kernel_height // 2, input.size(2) - 1)
    pad_bottom = min(kernel_height // 2, input.size(2) - 1)
    pad_left = min(kernel_width // 2, input.size(3) - 1)
    pad_right = min(kernel_width // 2, input.size(3) - 1)

    padding = [pad_left, pad_right, pad_top, pad_bottom]

    # Apply mirror (reflective) padding
    input_padded = F.pad(input, pad=padding, mode='reflect')

    # Perform the convolution without additional padding (padding=0)
    return F.conv2d(input_padded, kernel, padding=0)

def hill_cost_function(X):
    # Define the high-pass filter (Kern-Bohme filter)
    while X.dim() < 4:
        X = torch.unsqueeze(X, 0)
    channels = X.size(1)
    H = torch.tensor([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    H = H.repeat(channels, channels, 1, 1)
    
    # Define the first low-pass filter (3x3 averaging filter)
    L1 = torch.ones(channels, channels, 3, 3, dtype=torch.float32) / 9

    # Define the second low-pass filter (15x15 averaging filter)
    L2 = torch.ones(channels, channels, 15, 15, dtype=torch.float32) / 225

    # Step 1: Convolve X with H
    M1 = mirror_padded_convolution(X, H)

    # Step 2: Absolute value of M1
    M2 = torch.abs(M1)

    # Step 3: Convolve M2 with L1
    M3 = mirror_padded_convolution(M2, L1)
    loc = torch.where(M3 == 0)
    M3[loc] = 1.0
    M4 = 1 / M3
    M4[loc] = 0

    # Step 4: Convolve 1/M3 with L2
    result = mirror_padded_convolution(M4, L2)
    #print("cost have inf ? ", torch.isinf(result).any())

    return result

def count_zeros(matrix, epsilon=1e-6):
    num_zeros = torch.sum(is_effectively_zero(matrix, epsilon)).item()
    return num_zeros

def is_effectively_zero(tensor, epsilon=1e-6):
    """Check if the elements of the tensor are effectively zero."""
    return torch.abs(tensor) < epsilon

def smallest_k_elements(tensor, k):
    # Flatten the tensor to 1D
    flat_tensor = tensor.flatten()
    # Use torch.topk to find the smallest k elements and their indices in the flattened tensor
    smallest_values, flat_indices = torch.topk(flat_tensor, k, largest=False)

    # Convert flat indices to original multi-dimensional indices
    original_indices = np.unravel_index(flat_indices.numpy(), tensor.shape)

    return smallest_values, original_indices

def additive_distortion(X, Y):
    diff = torch.abs(X - Y)
    cost = hill_cost_function(X)
    distortion_matrix = diff*cost
    additive_distortion = torch.sum(distortion_matrix)
    return additive_distortion, distortion_matrix

def remove_points(image, threshold=0.005):
    distortion = 0
    cost = hill_cost_function(image)
    whole_distortion, distortion_matrix = additive_distortion(image, torch.zeros_like(image)) 
    new_image = copy.deepcopy(image)
    #print(image.shape)
    #print("whole_distortion: ", whole_distortion)
    dis_threshold = whole_distortion * threshold
    #print("distortion threshold: ", dis_threshold)
    loc = torch.where(distortion_matrix * distortion_matrix.numel() / whole_distortion < threshold)
    new_image[loc] = 0
    numel_eliminated = distortion_matrix[loc].numel()
    numel_remained = distortion_matrix.numel() - numel_eliminated
    distortion = torch.sum(distortion_matrix[loc])
    distortion_matrix[loc] = math.inf
    dis_remained = dis_threshold - distortion

    #filter
    T0 = time.process_time()
    while numel_eliminated > 1000:
        dis_remained = dis_threshold - distortion
        loc = torch.where(distortion_matrix * numel_remained < dis_remained)
        distortion += torch.sum(distortion_matrix[loc])
        new_image[loc] = 0
        distortion_matrix[loc] = math.inf
        numel_eliminated = distortion_matrix[loc].numel()
        numel_remained = distortion_matrix.numel() - numel_eliminated
    #print("remained after filter: ", numel_remained)

    T1 = time.process_time()
    k = numel_remained // 2
    while distortion < dis_threshold and k > 20: 
        smallest_values, indices = smallest_k_elements(distortion_matrix, k)
        if torch.sum(smallest_values) < dis_remained:
            new_image[indices] = 0
            distortion_matrix[indices] = math.inf
            numel_remained -= distortion_matrix[indices].numel()
            distortion += torch.sum(smallest_values)
            dis_remained = dis_threshold - distortion
            if k >= numel_remained:
                k = numel_remained // 2
        else:
            k = k // 2
    #print("remained after topk: ", numel_remained)
            
    T2 = time.process_time()
    while distortion < dis_threshold: 
        min_element_index = torch.argmin(distortion_matrix).item()
        min_element_index = np.unravel_index(min_element_index, new_image.shape)
        #print(min_element_index)
        #print("distortion: ", distortion)
        #print(new_image[min_element_index] * cost[min_element_index])
        distortion += distortion_matrix[min_element_index]
        new_image[min_element_index] = 0
        distortion_matrix[min_element_index] = math.inf
    T3 = time.process_time()
    #print(f"filter time: {T1-T0}s, topk time: {T2-T1}s, finetune time: {T3-T2}s")

    return new_image
        
def set_zero(image):
    loc = torch.where(torch.isinf(image))
    image[loc] = 0
    print(image.shape)
    return image

# Example usage
# X = load_image('path_to_image.jpg')
# cost = hill_cost_function(X)
# To visualize or process the cost, convert it back to an appropriate format

