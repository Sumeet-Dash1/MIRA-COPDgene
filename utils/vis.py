import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib

def plot_images_nib(images, titles, cols, cmap='gray', fig_size=(15, 15), slice_idx=0):
    """
    Plot images with titles for nibabel images.

    Parameters:
        images (list): List of images (nibabel.Nifti1Image or file paths) to plot.
        titles (list): List of titles for each image.
        cols (int): Number of columns in the plot.
        cmap (str): Colormap to use.
        fig_size (tuple): Figure size.
        slice_idx (int): Slice index to plot.
    """
    # Determine the number of rows
    rows = len(images) // cols + (1 if len(images) % cols else 0)
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    axes = np.array(axes).reshape(-1)  # Flatten axes for easier 1D indexing

    for i, (image, title) in enumerate(zip(images, titles)):
        # Load image using nibabel if it's a file path
        if isinstance(image, str):
            image = nib.load(image)
        image_array = image.get_fdata()  # Convert to numpy array

        # Select the slice to display
        ax = axes[i]
        if image_array.ndim == 3:  # If 3D, plot the specified slice
            ax.imshow(image_array[:, :, slice_idx], cmap=cmap)
        elif image_array.ndim == 4:  # If 4D, plot the first volume
            ax.imshow(image_array[:, :, slice_idx, 0], cmap=cmap)
        else:
            raise ValueError("Unsupported image dimensions for plotting.")
        
        ax.set_title(title)
        ax.axis('off')

    # Turn off unused axes
    for ax in axes[len(images):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_images_sitk(images, titles, cols, cmap='gray', fig_size=(15, 15), slice_idx=0):
    """
    Plot images with titles.

    Parameters:
        images (list): List of images (SimpleITK.Image or file paths) to plot.
        titles (list): List of titles for each image.
        cols (int): Number of columns in the plot.
        cmap (str): Colormap to use.
        fig_size (tuple): Figure size.
        slice_idx (int): Slice index to plot.
    """
    # Determine the number of rows
    rows = len(images) // cols + (1 if len(images) % cols else 0)
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    axes = np.array(axes).reshape(-1)  # Flatten axes for easier 1D indexing

    for i, (image, title) in enumerate(zip(images, titles)):
        if isinstance(image, str):
            image = sitk.ReadImage(image)  # Read image if it's a file path
        image_array = sitk.GetArrayFromImage(image)  # Convert to numpy array
        
        ax = axes[i]
        ax.imshow(image_array[slice_idx, :, :], cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    # Turn off unused axes
    for ax in axes[len(images):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_images(images, titles, cols=2, cmap='gray', fig_size=(15, 15), slice_idx=0):
    """
    Plot images with titles for 3D or 4D image arrays.

    Parameters:
        images (list): List of 3D or 4D image arrays (e.g., numpy arrays or image data).
        titles (list): List of titles for each image.
        cols (int): Number of columns in the plot (default: 2).
        cmap (str): Colormap to use for plotting (default: 'gray').
        fig_size (tuple): Size of the figure (default: (15, 15)).
        slice_idx (int): Slice index to plot (default: 0).
    """
    if len(images) != len(titles):
        raise ValueError("The number of images and titles must be the same.")
    
    # Determine the number of rows
    rows = len(images) // cols + (1 if len(images) % cols else 0)
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    axes = np.array(axes).flatten()  # Flatten axes for 1D indexing

    for i, (image, title) in enumerate(zip(images, titles)):
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Image at index {i} is not a NumPy array.")
        
        # Select the slice to display
        ax = axes[i]
        if image.ndim == 3:  # If 3D, plot the specified slice
            ax.imshow(image[:, :, slice_idx], cmap=cmap)
        elif image.ndim == 4:  # If 4D, plot the first volume
            ax.imshow(image[:, :, slice_idx, 0], cmap=cmap)
        else:
            raise ValueError(f"Unsupported image dimensions at index {i}. Must be 3D or 4D.")
        
        ax.set_title(title)
        ax.axis('off')

    # Turn off unused axes
    for ax in axes[len(images):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def analyze_inhale_exhale(inhale_image, exhale_image, set_name='Dataset'):
    """
    Analyzes inhale and exhale images, computes statistics, and plots histograms and slices.
    
    Parameters:
        inhale_image (numpy.ndarray): 3D array of the inhale image.
        exhale_image (numpy.ndarray): 3D array of the exhale image.
        set_name (str): Name of the dataset (e.g., 'Train_Set', 'Validation_Set', or 'Test_Set').
    
    Returns:
        dict: Statistics of the inhale and exhale images.
    """
    # Compute statistics for inhale image
    inhale_stats = {
        "Min": np.min(inhale_image),
        "Min (Excluding Zeros)": np.min(inhale_image[inhale_image > 0]),
        "Max": np.max(inhale_image),
        "Image Type": str(inhale_image.dtype),
        "Intensity Range": (inhale_image.min(), inhale_image.max()),
        "Voxel Count (Total)": inhale_image.size,
        "Voxel Count (Non-Zero)": np.count_nonzero(inhale_image),
        "99.99th Percentile": np.percentile(inhale_image, 99.99)
    }

    # Compute statistics for exhale image
    exhale_stats = {
        "Min": np.min(exhale_image),
        "Min (Excluding Zeros)": np.min(exhale_image[exhale_image > 0]),
        "Max": np.max(exhale_image),
        "Image Type": str(exhale_image.dtype),
        "Intensity Range": (exhale_image.min(), exhale_image.max()),
        "Voxel Count (Total)": exhale_image.size,
        "Voxel Count (Non-Zero)": np.count_nonzero(exhale_image),
        "99.99th Percentile": np.percentile(exhale_image, 99.99)
    }

    # Plot histograms and slices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Histogram for inhale image
    # non_zero_inhale = inhale_image[inhale_image > 0]
    # axes[0, 0].hist(non_zero_inhale.ravel(), bins=50, color='blue', alpha=0.7)
    axes[0, 0].hist(inhale_image.ravel(), bins=50, color='blue', alpha=0.7)
    axes[0, 0].set_title("Histogram of Inhale Image (Excluding Zeros)")
    axes[0, 0].set_xlabel("Pixel Intensity")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(
        x=inhale_stats["99.99th Percentile"],
        color='red', linestyle='--', linewidth=2,
        label=f'99.99th Percentile = {inhale_stats["99.99th Percentile"]:.2f}'
    )
    axes[0, 0].legend()

    # Histogram for exhale image
    # non_zero_exhale = exhale_image[exhale_image > 0]
    # axes[0, 1].hist(non_zero_exhale.ravel(), bins=50, color='orange', alpha=0.7)
    axes[0, 1].hist(exhale_image.ravel(), bins=50, color='orange', alpha=0.7)
    axes[0, 1].set_title("Histogram of Exhale Image (Excluding Zeros)")
    axes[0, 1].set_xlabel("Pixel Intensity")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(
        x=exhale_stats["99.99th Percentile"],
        color='red', linestyle='--', linewidth=2,
        label=f'99.99th Percentile = {exhale_stats["99.99th Percentile"]:.2f}'
    )
    axes[0, 1].legend()

    # Slices of inhale image
    slice_index = inhale_image.shape[2] // 2  # Middle slice
    axes[1, 0].imshow(inhale_image[:, :, slice_index], cmap="gray")
    axes[1, 0].set_title("Inhale Image (Middle Slice)")
    axes[1, 0].axis("off")

    # Slices of exhale image
    axes[1, 1].imshow(exhale_image[:, :, slice_index], cmap="gray")
    axes[1, 1].set_title("Exhale Image (Middle Slice)")
    axes[1, 1].axis("off")

    # Difference view
    difference = inhale_image[:, :, slice_index] - exhale_image[:, :, slice_index]
    axes[1, 2].imshow(difference, cmap="bwr")
    axes[1, 2].set_title("Difference (Inhale - Exhale, Middle Slice)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"{set_name} - Inhale Image Statistics:")
    for key, value in inhale_stats.items():
        print(f"{key}: {value}")

    print(f"\n{set_name} - Exhale Image Statistics:")
    for key, value in exhale_stats.items():
        print(f"{key}: {value}")

    return {"Inhale": inhale_stats, "Exhale": exhale_stats}

def plot_slice(image):
    """
    Plot images with titles for 3D or 4D image arrays.

    Parameters:
        images (list): List of 3D or 4D image arrays (e.g., numpy arrays or image data).
        titles (list): List of titles for each image.
        cols (int): Number of columns in the plot (default: 2).
        cmap (str): Colormap to use for plotting (default: 'gray').
        fig_size (tuple): Size of the figure (default: (15, 15)).
        slice_idx (int): Slice index to plot (default: 0).
    """

    if image.ndim == 3:  # If 3D, plot the specified slice
        slice_idx = image.shape[2] // 2
        plt.imshow(image[:, :, slice_idx], cmap='gray')
        plt.title("Middle Slice")
        plt.axis('off')
        plt.show()

