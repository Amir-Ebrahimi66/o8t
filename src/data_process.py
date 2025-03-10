import os
import nibabel as nib
from nilearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import argparse
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_erosion
from nilearn.image import resample_to_img
from nilearn.image import resample_img
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric, AffineMap
from dipy.align.transforms import RigidTransform3D, AffineTransform3D
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    filename='app.log',  # Specify the log file name
                    filemode='w')
logger = logging.getLogger(__name__)


class MRIProcessor:
    def __init__(self, input_path, output_path, sigma, normalize="zscore"):
        self.input_path = input_path
        self.output_path = output_path
        self.sigma = sigma
        self.normalize_method = normalize
        self.img = None
        self.original_img = None
        self.transformed_img = None

    def load_image(self):
        """Load and validate the input volume (DICOM or NIFTI)."""
        try:
            if self.input_path.endswith(".nii") or self.input_path.endswith(".nii.gz"):
                self.img = nib.load(self.input_path)
                self.original_img = self.img  # Store original image
                logger.info(f"Loaded NIFTI file: {self.input_path}")
            elif os.path.isdir(self.input_path):  # Assume DICOM folder
                dicom_files = [f for f in os.listdir(self.input_path) if f.endswith(".dcm")]
                if not dicom_files:
                    raise ValueError("No DICOM files found in the directory.")
                dicom_data = pydicom.dcmread(os.path.join(self.input_path, dicom_files[0]))
                self.img = dicom_data.pixel_array.astype(np.float32)
                logger.info(f"Loaded DICOM scan from: {self.input_path}")
            else:
                raise ValueError("Unsupported file format. Use NIFTI (.nii/.nii.gz) or a DICOM folder.")
        except Exception as e:
            logger.error(f"Error loading image: {e}", exc_info=True)
            raise

    def normalize_image(self):
        """Normalize image intensities using Z-score or Min-Max scaling."""
        try:
            data = self.img.get_fdata()
            if self.normalize_method == "zscore":

                mean_val = np.mean(data)
                std_val = np.std(data)

                if std_val == 0:
                    std_val = 1  # Avoid division by zero

                norm_data = (data - mean_val) / std_val
                logger.info(f"Normalization applied. New min: {norm_data.min()}, max: {norm_data.max()}")

            elif self.normalize_method == "minmax":
                norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            else:
                raise ValueError("Invalid normalization method. Use 'zscore' or 'minmax'.")

            self.img = nib.Nifti1Image(norm_data.astype(np.float32), self.img.affine)
            self.original_img = self.img  # Store original image (normalized)
            logger.info("Image normalization applied.")
        except Exception as e:
            logger.error(f"Error normalizing image: {e}", exc_info=True)
            raise

    def apply_gaussian_smoothing(self):
        """Apply Gaussian smoothing."""
        try:
            data = self.img.get_fdata()
            smoothed_data = gaussian_filter(data, sigma=self.sigma)
            self.img = nib.Nifti1Image(smoothed_data, self.img.affine)
            self.original_img = self.img  # Store original image (smoothed)
            logger.info(f"Applied Gaussian smoothing with sigma={self.sigma}")
        except Exception as e:
            logger.error(f"Error applying Gaussian smoothing: {e}", exc_info=True)
            raise

    def skull_strip(self):
        """Perform atlas-based skull stripping."""
        try:
            # Load MNI152 template and mask from Nilearn
            atlas_img = datasets.load_mni152_template()
            mask_img = datasets.load_mni152_brain_mask()

            # Get the data and affine of the template and mask
            atlas_data = atlas_img.get_fdata()
            mask_data = mask_img.get_fdata()
            atlas_affine = atlas_img.affine
            mask_affine = mask_img.affine

            # Perform affine registration
            logger.info("Registering input image to atlas template...")
            affreg = AffineRegistration(metric=MutualInformationMetric(nbins=32))

            # Run registration
            affine_map = affreg.optimize(atlas_data, self.img.get_fdata(), transform=AffineTransform3D(), params0=None)
            self.transformed_img = nib.Nifti1Image(affine_map.transform(self.img.get_fdata()).astype(np.float32), atlas_affine)

            # Resample transformed mask to match the shape of the input image
            logger.info("Resampling transformed mask to match the input image shape...")

            resampled_mask_img = resample_img(
                mask_img,
                target_affine=self.transformed_img.affine,
                target_shape=self.transformed_img.shape,
                interpolation="nearest",  # Nearest-neighbor avoids soft mask values
                force_resample=True,
                # copy_header=True
            )

            resampled_mask_data = resampled_mask_img.get_fdata()
            binary_mask = resampled_mask_data > 0.5

            # Apply morphological erosion to remove boundary voxels (adjust iterations as needed)
            eroded_mask = binary_erosion(binary_mask, iterations=15)

            # Apply the mask to the transformed image data
            skull_stripped_data = self.transformed_img.get_fdata() * eroded_mask

            # Create new NIfTI image with the transformed brain data and the same affine as the original image
            self.img = nib.Nifti1Image(skull_stripped_data.astype(np.float32), self.img.affine)
            logger.info("Skull stripping completed using atlas mask.")
        except Exception as e:
            logger.error(f"Error during skull stripping: {e}", exc_info=True)
            raise

    def save_output(self):
        """Save the processed MRI scan to an output file."""
        try:
            nib.save(self.img, self.output_path)
            logger.info(f"Processed MRI saved to: {self.output_path}")
            # Generate visualization
            visualization_path = self.output_path.replace(".nii.gz", "_comparison.png")
            self.visualize_results(self.transformed_img, self.img, visualization_path) # TODO: Fix this
            logger.info(f"Visualization saved to: {visualization_path}")
        except Exception as e:
            logger.error(f"Error saving processed image: {e}", exc_info=True)
            raise

    @staticmethod
    def create_checkerboard(image1, image2, block_size=8):
        """
        Generate a checkerboard comparison of two MRI slices.

        Parameters:
            image1 (numpy.ndarray): First image.
            image2 (numpy.ndarray): Second image.
            block_size (int): Size of checkerboard blocks.

        Returns:
            numpy.ndarray: Checkerboard image.
        """
        h, w = image1.shape
        checkerboard = np.zeros((h, w))

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    checkerboard[i: i + block_size, j: j + block_size] = image1[i: i + block_size, j: j + block_size]
                else:
                    checkerboard[i: i + block_size, j: j + block_size] = image2[i: i + block_size, j: j + block_size]

        return checkerboard


    def visualize_results(self, original_img, processed_img, output_path):
        """
        Display and save a comparison between original and processed MRI images.

        Parameters:
            original_img (nib.Nifti1Image): Original MRI image.
            processed_img (nib.Nifti1Image): Processed MRI image.
            output_path (str): Path to save the visualization.
        """
        original_data = original_img.get_fdata()
        processed_data = processed_img.get_fdata()

        # Extract a middle slice
        slice_idx = original_data.shape[2] // 2
        original_slice = original_data[:, :, slice_idx]
        processed_slice = processed_data[:, :, slice_idx]

        # Create checkerboard visualization
        checkerboard_img = self.create_checkerboard(original_slice, processed_slice, block_size=8)

        # Plot images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_slice.T, cmap="gray", origin="lower")
        axes[0].set_title("After Registration")

        axes[1].imshow(processed_slice.T, cmap="gray", origin="lower")
        axes[1].set_title("After Skull Stripping")

        axes[2].imshow(checkerboard_img.T, cmap="gray", origin="lower")
        axes[2].set_title("Checkerboard Comparison")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.show(block=False)


    def run_pipeline(self):
        """Execute the full processing pipeline."""
        logger.info("Starting MRI processing pipeline...")
        self.load_image()
        self.normalize_image()
        self.apply_gaussian_smoothing()
        self.skull_strip()
        self.save_output()
        logger.info("Processing pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI Processing Pipeline")
    parser.add_argument("--input", required=True, help="Path to input MRI file (NIFTI or DICOM folder)")
    parser.add_argument("--output", required=True, help="Path to save processed output")
    parser.add_argument("--sigma", type=float, required=True, help="Sigma for Gaussian smoothing")
    parser.add_argument("--normalize", choices=["zscore", "minmax"], default="zscore", help="Normalization method")

    args = parser.parse_args()

    # Initialize processor and run pipeline
    processor = MRIProcessor(args.input, args.output, args.sigma, args.normalize)
    processor.run_pipeline()
