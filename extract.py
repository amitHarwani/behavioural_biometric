import os
import zipfile


def extract_data(zip_file_path, output_dir):
    """
        Extracts HMOG dataset file
        IMP: Remove data_description.pdf and __MACOSX__ post extraction
    """

    # As the zip file contains a public_dataset folder
    complete_extracted_path_dir = os.path.join(output_dir, "public_dataset")

    # Checking if the file has already been extracted
    if os.path.exists(complete_extracted_path_dir) and os.path.getsize(complete_extracted_path_dir) > 0:
        print("Dataset Has Already Been Extracted")
        return

    with zipfile.ZipFile(zip_file_path, 'r') as zipped_file:
        # Extracting the zipped file into the output directory
        zipped_file.extractall(path=output_dir)

        # For all the files in the zipped file
        for file_name in zipped_file.namelist():
            # If the file is a zip file
            if file_name.endswith('.zip'):
                # Its path is output_dir/file_name
                next_zipfile_path = os.path.join(output_dir, file_name)
                # Extract the file
                extract_data(next_zipfile_path, os.path.dirname(next_zipfile_path))
                # Remove the .zip file
                os.remove(next_zipfile_path)
        

HMOG_DATASET_ZIP_PATH = './datasets/hmog_dataset.zip'
OUTPUT_DIR = './datasets/hmog'
EXTRACT_DATA = True

if EXTRACT_DATA:
    extract_data(HMOG_DATASET_ZIP_PATH, OUTPUT_DIR)






