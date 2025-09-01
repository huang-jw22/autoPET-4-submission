import os
import shutil
from nnunetv2.training.dataloading.utils import _convert_to_npy, unpack_dataset
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.utilities.file_path_utilities import maybe_convert_to_dataset_name
from tqdm import tqdm

def merge_lesions_and_organ_dataset(lesion_dataset_id, organ_dataset_id):
    lesion_dataset_name = maybe_convert_to_dataset_name(lesion_dataset_id)
    organ_dataset_name = maybe_convert_to_dataset_name(organ_dataset_id)
    
    # Updated print statement to reflect the new logic
    print(f"Merging lesion dataset {lesion_dataset_name} into organ dataset {organ_dataset_name}")
    print(f"The final merged dataset will be located in the preprocessed folder for: {organ_dataset_name}")
    print(f"The lesion dataset '{lesion_dataset_name}' will be left untouched.")

    # Define paths to the preprocessed 3d_fullres folders
    lesion_preprocessed_folder = join(nnUNet_preprocessed, lesion_dataset_name, "nnUNetPlans_3d_fullres")
    organ_preprocessed_folder = join(nnUNet_preprocessed, organ_dataset_name, "nnUNetPlans_3d_fullres")

    # print("\nUnpacking both datasets. This can take some time...")
    # Unpacking is still necessary for both to ensure all files are available as .npy
    unpack_dataset(lesion_preprocessed_folder, unpack_segmentation=True, overwrite_existing=False,
                   num_processes=max(1, round(get_allowed_n_proc_DA() // 2)), verify_npy=True)
    unpack_dataset(organ_preprocessed_folder, unpack_segmentation=True, overwrite_existing=False,
                   num_processes=max(1, round(get_allowed_n_proc_DA() // 2)), verify_npy=True)
    
    # --- Start of Modified Logic ---

    # Step 1: Rename the original organ segmentations in the organ dataset folder
    organ_seg_files = [f for f in os.listdir(organ_preprocessed_folder) if f.endswith("_seg.npy")]

    for f in tqdm(organ_seg_files, desc="Renaming organ files"):
        original_seg_path = join(organ_preprocessed_folder, f)
        new_organ_seg_path = join(organ_preprocessed_folder, f.replace("_seg.npy", "_seg_org.npy"))
        
        if not os.path.isfile(new_organ_seg_path):
            shutil.move(original_seg_path, new_organ_seg_path)

    print(f"Found {len(organ_seg_files)} organ segmentation files in '{organ_dataset_name}'.")

    # Step 2: Copy the lesion segmentations from the lesion dataset into the organ dataset folder
    print(f"\nStep 2: Copying lesion segmentations from '{lesion_dataset_name}' into '{organ_dataset_name}'...")
    lesion_seg_files = [f for f in os.listdir(lesion_preprocessed_folder) if f.endswith("_seg.npy")]
    print(f"Found {len(lesion_seg_files)} lesion segmentation files in '{lesion_dataset_name}'.")
    # First delete any existing lesion segmentation files in the organ dataset folder
    for f in lesion_seg_files:
        destination_lesion_path = join(organ_preprocessed_folder, f)
        if os.path.isfile(destination_lesion_path):
            os.remove(destination_lesion_path)
    print(f"Total files in '{organ_dataset_name}': {len(os.listdir(organ_preprocessed_folder))}")
    if len(os.listdir(organ_preprocessed_folder)) == 2400:
        for f in tqdm(lesion_seg_files, desc="Copying lesion files"):
            # Path to the source file in the lesion dataset
            source_lesion_path = join(lesion_preprocessed_folder, f)
            # Path to the destination in the organ dataset (name remains _seg.npy)
            destination_lesion_path = join(organ_preprocessed_folder, f)
            
            if not os.path.isfile(destination_lesion_path):
                shutil.copy(source_lesion_path, destination_lesion_path)
    else:
        npy_files = [f for f in os.listdir(organ_preprocessed_folder) if f.endswith(".npy")]
        num_data_npy_files = len([f for f in npy_files if not f.endswith("_seg_org.npy")])
        print(f"Warning: The number of data files in '{organ_dataset_name}' is not as expected. Found {num_data_npy_files} data files instead of 1614.")
        #Now we now three are missing, so we need to unpack the three data from .npz/.pkl files
        print("Unpacking the missing data files from .npz files...")
        # Only unpack the three missing data files, do not unpack the whole dataset because that would take too long
        missing_data_files = [f for f in os.listdir(organ_preprocessed_folder) if f.endswith(".npz") and f[:-4] + ".npy" not in npy_files]
        print(f"Found {len(missing_data_files)} missing data files to unpack. Their names are: {missing_data_files}")
        for npz_file in tqdm(missing_data_files, desc="Unpacking missing data files"):
            npz_file_path = join(organ_preprocessed_folder, npz_file)
            # Unpack the npz file to .npy
            _convert_to_npy(npz_file_path, unpack_segmentation=True, overwrite_existing=True, verify_npy=True)
    # --- End of Modified Logic ---

    print(f"\nDone merging datasets. The combined dataset is now in:\n{organ_preprocessed_folder}")
    # Number check to ensure the merge was successful
    print(f"Total files in '{organ_dataset_name}': {len(os.listdir(organ_preprocessed_folder))}")
    lesion_seg_files = [f for f in os.listdir(organ_preprocessed_folder) if f.endswith("_seg.npy")]
    organ_seg_files = [f for f in os.listdir(organ_preprocessed_folder) if f.endswith("_seg_org.npy")]
    print(f"There are {len(lesion_seg_files)} lesion segmentation files and {len(organ_seg_files)} organ segmentation files in the merged dataset.")


def merge_datasets_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lesion_dataset_id', type=int, required=True, help='dataset id of the lesion dataset')
    parser.add_argument('-o', '--organ_dataset_id', type=int, required=True, help='dataset id of the organ dataset')
    args = parser.parse_args()
    merge_lesions_and_organ_dataset(args.lesion_dataset_id, args.organ_dataset_id)


if __name__ == "__main__":
    merge_lesions_and_organ_dataset(722, 723)