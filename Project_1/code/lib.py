import numpy as np
import skimage as sk
from skimage import img_as_ubyte
import skimage.io as skio
import matplotlib.pyplot as plt
import os


# part 1

def normalized_cross_correlation(img1, img2):
    img1_normalized = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
    img2_normalized = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
    return np.sum(img1_normalized * img2_normalized)

def align1(image, reference, search_window=90, crop_size=15):
    best_score = float('-inf')
    best_shift = (0, 0)
    
    height, width = image.shape
    
    for y_shift in range(-search_window, search_window + 1):
        for x_shift in range(-search_window, search_window + 1):
            shifted = np.roll(image, (y_shift, x_shift), axis=(0, 1))
            
            # Exclude borders (crop to internal pixels only)
            y_start = max(crop_size, -y_shift)
            y_end = min(height - crop_size, height - y_shift)
            x_start = max(crop_size, -x_shift)
            x_end = min(width - crop_size, width - x_shift)
            
            score = normalized_cross_correlation(
                shifted[y_start:y_end, x_start:x_end],
                reference[y_start:y_end, x_start:x_end]
            )
            
            if score > best_score:
                best_score = score
    
    return np.roll(image, best_shift, axis=(0, 1)), best_shift


#part 2

def compute_normalized_cross_correlation(im1, im2):
    im1_norm = (im1 - np.mean(im1)) / np.linalg.norm(im1)
    im2_norm = (im2 - np.mean(im2)) / np.linalg.norm(im2)
    return np.sum(im1_norm.flatten() * im2_norm.flatten())

def get_best_displacement(im1, im2, on_last_level, image_name):
    if image_name == 'emir.tif':
        max_displacement = 30
        cut_amount = max(int(len(im1) * 0.15), 1)
    else:
        max_displacement = 15
        cut_amount = max(int(len(im1) * 0.25), 1)
    
    skip_factor = 1
    
    if on_last_level:
        max_displacement = 3
    elif len(im1) > 100:
        skip_factor = 3
    
    best_alignment_displacement = 0, 0
    best_alignment_score = -np.inf

    cut_im2 = im2[cut_amount:-cut_amount, cut_amount:-cut_amount]

    for y_displacement in range(-max_displacement, max_displacement + 1, skip_factor):
        y_rolled_im1 = np.roll(im1, y_displacement, 0)
        for x_displacement in range(-max_displacement, max_displacement + 1, skip_factor):
            xy_rolled_im1 = np.roll(y_rolled_im1, x_displacement, 1)
            cut_im1 = xy_rolled_im1[cut_amount:-cut_amount, cut_amount:-cut_amount]
            
            new_alignment_score = compute_normalized_cross_correlation(cut_im1, cut_im2)
            if new_alignment_score > best_alignment_score:
                best_alignment_displacement = y_displacement, x_displacement
                best_alignment_score = new_alignment_score

    return best_alignment_displacement

def align(im1, im2, image_name):
    amount_of_images = 3
    low_resolution_images1 = []
    low_resolution_images2 = []
    
    for i in range(amount_of_images, 0, -1):
        low_resolution_images1.append(sk.transform.rescale(im1, 0.5**i))
        low_resolution_images2.append(sk.transform.rescale(im2, 0.5**i))
    
    low_resolution_images1.append(im1)
    low_resolution_images2.append(im2)

    total_displacement = [0, 0]
    displacement = [0, 0]

    for i in range(len(low_resolution_images1)):
        total_displacement = [2 * (total_displacement[0] + displacement[0]), 2 * (total_displacement[1] + displacement[1])]
        low_resolution_images1[i] = np.roll(low_resolution_images1[i], total_displacement[0], 0)
        low_resolution_images1[i] = np.roll(low_resolution_images1[i], total_displacement[1], 1)
        print(f"Set image {i} to position {total_displacement}")

        on_last_level = i == len(low_resolution_images1) - 1
        displacement = get_best_displacement(low_resolution_images1[i], low_resolution_images2[i], on_last_level, image_name)

    total_displacement = [total_displacement[0] + displacement[0], total_displacement[1] + displacement[1]]
    aligned_image = np.roll(np.roll(im1, total_displacement[0], 0), total_displacement[1], 1)
    print(f"Final image set to position {total_displacement}")

    return aligned_image, total_displacement

def process_image(imname):
    print(f"Processing {imname}")
    im = skio.imread(f'data/{imname}')
    im = sk.img_as_float(im)
    
    height = int(np.floor(im.shape[0] / 3.0))
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    ag, g_shift = align(g, b, imname)
    ar, r_shift = align(r, b, imname)

    print(f"Green channel shift: {g_shift}")
    print(f"Red channel shift: {r_shift}")

    cut_amount = max(int(height * 0.1), 1)
    b = b[cut_amount:-cut_amount, cut_amount:-cut_amount]
    ag = ag[cut_amount:-cut_amount, cut_amount:-cut_amount]
    ar = ar[cut_amount:-cut_amount, cut_amount:-cut_amount]

    im_out = np.dstack([ar, ag, b])
    im_out = sk.img_as_ubyte(im_out)

   
    os.makedirs('out_path', exist_ok=True)

    skio.imsave(f'out_path/aligned_{imname.split(".")[0]}.jpg', im_out)

