import numpy as np
import nibabel as nib
from src.evaluation.metrics import compute_dice_coefficient


class LabelFusionPipeline:
    def __init__(self, fixed_image, transformed_labels = None, transformed_labels_dir = None, reference_image_path = None):
        """
        Initialize the label fusion pipeline.
        Parameters:
            fixed_image (str): Path to the fixed image (validation image).
            transformed_labels_dir (str): Directory containing transformed label maps.
            reference_image_path (str): Path to the reference image (for metadata).
        """
        self.fixed_image = fixed_image
        self.transformed_labels_dir = transformed_labels_dir
        self.reference_image_path = reference_image_path
        self.transformed_labels = transformed_labels
        self.fused_label = None

    def load_transformed_labels(self):
        """
        Load all transformed label maps from the directory.
        Returns:
            None: Populates self.transformed_labels with a list of label arrays.
        """
        import os
        labels = []
        for file in sorted(os.listdir(self.transformed_labels_dir)):
            if file.endswith(".hdr"):
                img = nib.load(os.path.join(self.transformed_labels_dir, file))
                labels.append(img.get_fdata())
        self.transformed_labels = labels
        print(f"Loaded {len(self.transformed_labels)} transformed labels.")

    def apply_fusion(self, fusion_strategy):
        """
        Apply the specified fusion strategy to the loaded labels.
        Parameters:
            fusion_strategy (callable): A function implementing the label fusion strategy.
        Returns:
            None: Populates self.fused_label with the fused label map.
        """
        if not self.transformed_labels:
            raise ValueError("Transformed labels not loaded. Call load_transformed_labels() first.")
        self.fused_label = fusion_strategy(self.transformed_labels)
        print("Applied fusion strategy.")

    def evaluate(self, ground_truth_path):
        """
        Evaluate the fused label map against the ground truth.
        Parameters:
            ground_truth_path (str): Path to the ground truth label map.
        Returns:
            dict: A dictionary containing evaluation metrics (e.g., Dice coefficient).
        """
        if self.fused_label is None:
            raise ValueError("Fused label not available. Call apply_fusion() first.")

        # Load ground truth
        ground_truth_img = nib.load(ground_truth_path)
        ground_truth = ground_truth_img.get_fdata()

        # Compute metrics
        dice = compute_dice_coefficient(ground_truth, self.fused_label)
        return {"dice": dice}

    def save_fused_label(self, output_path):
        """
        Save the fused label map to the specified output path.
        Parameters:
            output_path (str): Path to save the fused label map.
        Returns:
            None
        """
        if self.fused_label is None:
            raise ValueError("Fused label not available. Call apply_fusion() first.")

        # Load reference image for metadata
        reference_img = nib.load(self.fixed_image)

        # Save the fused label map
        fused_img = nib.Nifti1Image(self.fused_label.astype(np.int16), reference_img.affine)
        nib.save(fused_img, output_path)
        print(f"Fused label saved to {output_path}.")