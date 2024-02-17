import cv2
import os
import shutil

def detect_and_copy_assets_in_folders(root_folder, output_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        
        if not os.path.isdir(folder_path):
            continue

        preview_path = os.path.join(folder_path, f"{folder_name}_preview.png")
        if not os.path.isfile(preview_path):
            continue

        preview_img = cv2.imread(preview_path)

        assets_to_copy = [preview_path]

        for image_name in os.listdir(folder_path):
            if image_name.endswith(".png") and image_name != f"{folder_name}_preview.png":
                image_path = os.path.join(folder_path, image_name)
                template = cv2.imread(image_path)

                result = cv2.matchTemplate(preview_img, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                similarity_threshold = 0.8

                if max_val >= similarity_threshold:
                    assets_to_copy.append(image_path)

        if len(assets_to_copy) > 1:
            output_subfolder = os.path.join(output_folder, folder_name)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            for asset in assets_to_copy:
                shutil.copy2(asset, output_subfolder)

            print(f"Copied folder {folder_name} with assets to {output_subfolder}")

# Specify the root folder containing subfolders with images
root_folder = "Data/Challenge_Data/Assets"

# Specify the output folder to copy the folders with detected assets
output_folder = "output"

detect_and_copy_assets_in_folders(root_folder, output_folder)
