import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

def plot_2d_slice(image, seg=None, slice_index=None, title="2D Slice", cmap="gray"):
    """
    Plot a 2D slice of the image with optional segmentation overlay.
    Parameters:
        image (np.ndarray): 3D image array.
        seg (np.ndarray, optional): 3D segmentation array (same dimensions as `image`).
        slice_index (int, optional): Index of the slice to plot. Defaults to the middle slice.
        title (str): Title for the plot.
        cmap (str): Colormap for the image.
    """
    if slice_index is None:
        slice_index = image.shape[2] // 2  # Default to the middle slice

    plt.figure(figsize=(8, 8))
    plt.imshow(image[:, :, slice_index], cmap=cmap, interpolation="none")
    if seg is not None:
        plt.contour(seg[:, :, slice_index], levels=[0.5], colors="red", linewidths=0.5)
    plt.title(title)
    plt.axis("off")
    plt.show()

def plot_3d_segmentation(seg, title="3D Segmentation", threshold=0.5):
    """
    Visualize a 3D segmentation using isosurfaces.
    Parameters:
        seg (np.ndarray): 3D segmentation array.
        title (str): Title for the plot.
        threshold (float): Threshold to create the isosurface.
    """
    verts, faces, _, _ = measure.marching_cubes(seg, level=threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    mesh = ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        cmap="Spectral", lw=1, alpha=0.6
    )
    ax.set_title(title)
    plt.show()

def visualize_overlay(image, seg, alpha=0.5, slice_index=None, cmap="gray", title="Overlay"):
    """
    Overlay segmentation on the image.
    Parameters:
        image (np.ndarray): 3D image array.
        seg (np.ndarray): 3D segmentation array.
        alpha (float): Transparency level of the overlay.
        slice_index (int, optional): Index of the slice to plot. Defaults to the middle slice.
        cmap (str): Colormap for the image.
        title (str): Title for the plot.
    """
    if slice_index is None:
        slice_index = image.shape[2] // 2  # Default to the middle slice

    plt.figure(figsize=(8, 8))
    plt.imshow(image[:, :, slice_index], cmap=cmap, interpolation="none")
    plt.imshow(seg[:, :, slice_index], cmap="Reds", alpha=alpha, interpolation="none")
    plt.title(title)
    plt.axis("off")
    plt.show()

def load_nifti_and_plot(filepath, slice_index=None):
    """
    Load a NIFTI file and plot a 2D slice.
    Parameters:
        filepath (str): Path to the NIFTI file.
        slice_index (int, optional): Index of the slice to plot. Defaults to the middle slice.
    """
    img = nib.load(filepath)
    data = img.get_fdata()
    plot_2d_slice(data, slice_index=slice_index, title=f"Slice from {filepath}")