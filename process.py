import json
import os
import shutil
import subprocess
from pathlib import Path
import SimpleITK
import torch
import nibabel as nib
import numpy as np
from classify_pet import classify_pet

from utils import save_click_heatmaps

class Autopet_final_model:

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        # according to the specified grand-challenge interfaces
        self.input_path = "/input/"
        # according to the specified grand-challenge interfaces
        self.output_path = "/output/images/tumor-lesion-segmentation/"
        self.nii_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs"
        )
        self.lesion_click_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/clicksTs"
        )
        self.result_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        )
        self.nii_seg_file = "TCIA_001.nii.gz"
        self.ckpt_path = "/opt/algorithm/tracer_classifier.pt"
        self.num_clicks = 10
        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)
    
    def gc_to_swfastedit_format(self, gc_json_path, swfast_json_path):
        with open(gc_json_path, 'r') as f:
            gc_dict = json.load(f)
        swfast_dict = {
            "tumor": [],
            "background": []
        }
        background_clicks_count = 0
        for point in gc_dict.get("points", []):
            if point["name"] == "tumor":
                swfast_dict["tumor"].append(point["point"])
            elif point["name"] == "background":
                swfast_dict["background"].append(point["point"])
                background_clicks_count += 1
        self.num_clicks = background_clicks_count
        print(f"Detected {self.num_clicks} background clicks. This will determine model selection.")
        with open(swfast_json_path, 'w') as f:
            json.dump(swfast_dict, f)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print(
                "Device memory: "
                + str(torch.cuda.get_device_properties(0).total_memory)
            )

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/ct/", ct_mha),
            os.path.join(self.nii_path, "TCIA_001_0000.nii.gz"),
        )
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/pet/", pet_mha),
            os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
        )
        
        json_file = os.path.join(self.input_path, "lesion-clicks.json")
        print(f"json_file: {json_file}")

        self.gc_to_swfastedit_format(json_file, os.path.join(self.lesion_click_path, "TCIA_001_clicks.json"))

        click_file = os.listdir(self.lesion_click_path)[0]
        if click_file:
            with open(os.path.join(self.lesion_click_path, click_file), 'r') as f:
                clicks = json.load(f)
            save_click_heatmaps(clicks, self.nii_path, 
                                os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
                                )
        print(os.listdir(self.nii_path))

        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(
            os.path.join(self.result_path, self.nii_seg_file),
            os.path.join(self.output_path, uuid + ".mha"),
        )
        print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))
    
    def postprocess(self, tracer):
        seg = nib.load(os.path.join(self.result_path, 'TCIA_001.nii.gz'))
        seg_arr = seg.get_fdata()

        pet = nib.load(os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz')).get_fdata()
        suv_threshold = 1.5 if tracer == 'fdg' else 1

        # Apply SUV threshold to filter out non-significant regions
        seg_arr[pet < suv_threshold] = 0
        seg_arr = np.where(seg_arr == 1, 1, 0)

        final_seg = nib.Nifti1Image(seg_arr.astype(np.int16), seg.affine, seg.header)
        nib.save(final_seg, os.path.join(self.result_path, self.nii_seg_file))

    def predict(self):
        """
        Your algorithm goes here
        """
        print("Tracer classification starting now!")
        pet_path = os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz')
        tracer = classify_pet(pet_path, self.ckpt_path)
        dataset_nr = 723 if tracer == 'fdg' else 820
        
        print("nnUNet segmentation starting!")
        if tracer == 'fdg':
            if self.num_clicks >= 5:
                dataset_nr = 723
                cproc = subprocess.run(
                    f"nnUNetv2_predict -i {self.nii_path} -o {self.result_path} -d {dataset_nr} -c 3d_fullres_patch192_bs2 -tr autoPET3_Trainer -p ResEncL_Plan -f 0 1 2",
                    shell=True,
                    check=True,
                )
            else:
                dataset_nr = 826
                cproc = subprocess.run(
                    f"nnUNetv2_predict -i {self.nii_path} -o {self.result_path} -d {dataset_nr} -c 3d_fullres_patch192_bs2 -tr autoPET3_Trainer -p ResEncL_Plan -f 0 1",
                    shell=True,
                    check=True,
                )   
        else:
            cproc = subprocess.run(
                f"nnUNetv2_predict -i {self.nii_path} -o {self.result_path} -d {dataset_nr} -c 3d_fullres_patch192_bs3 -tr nnUNetTrainer_finetune -p ResEncL_Plan",
                shell=True,
                check=True,
            )    
        print(cproc)
        # since nnUNet_predict call is split into prediction and postprocess, a pre-mature exit code is received but
        # segmentation file not yet written. This hack ensures that all spawned subprocesses are finished before being
        # printed.
        print("Prediction finished")
        return tracer

   
    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        tracer = self.predict()
        # Add Postprocessing
        print("Start Postprocessing")
        self.postprocess(tracer)

        print("Start output writing")
        self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_final_model().process()
