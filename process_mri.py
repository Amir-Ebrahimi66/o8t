import argparse
from src.data_process import MRIProcessor

# Example usage: python process_mri.py --input sub-0002_ses-01_T1w.nii.gz --output output.nii.gz --sigma 1.5
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process an MRI scan with Gaussian smoothing.")

    parser.add_argument("--input", required=True, help="Path to the input MRI file (.nii.gz)")
    parser.add_argument("--output", required=True, help="Path to save the processed output")
    parser.add_argument("--sigma", type=float, required=True, help="Sigma value for Gaussian smoothing")

    # Parse arguments
    args = parser.parse_args()

    # Initialize processor and run pipeline
    normalize = "minmax"
    processor = MRIProcessor(args.input, args.output, args.sigma, normalize)
    processor.run_pipeline()
