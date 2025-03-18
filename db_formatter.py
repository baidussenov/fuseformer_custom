import os
import shutil

# Define the source and destination base directories
source_dir = "dataset"  # Original dataset folder
dest_dir = "dataset_copy"  # New dataset folder

# Create the destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Step 1: Get the list of scene folders (assuming there are 40 scenes)
scene_folders = sorted(os.listdir(source_dir))

# Step 2: Identify the 4 lighting condition folder names from the first scene
# (Assuming they are consistent across all scenes)
first_scene_path = os.path.join(source_dir, scene_folders[0])
lighting_conditions = sorted(os.listdir(first_scene_path))

# Step 3: Create the new structure
# For each lighting condition, create a folder in dataset_copy
for lighting in lighting_conditions:
    lighting_dest_path = os.path.join(dest_dir, lighting)
    os.makedirs(lighting_dest_path, exist_ok=True)

    # For each scene, create a subfolder under the lighting condition
    for scene in scene_folders:
        scene_dest_path = os.path.join(lighting_dest_path, scene)
        os.makedirs(scene_dest_path, exist_ok=True)

        # Source path for the images (original structure: scene -> lighting -> images)
        source_images_path = os.path.join(source_dir, scene, lighting)

        # Copy all .png images from the source to the destination
        for image_file in os.listdir(source_images_path):
            if image_file.endswith(".png"):
                source_image = os.path.join(source_images_path, image_file)
                dest_image = os.path.join(scene_dest_path, image_file)
                shutil.copy2(source_image, dest_image)  # Use copy2 to preserve metadata

print("Dataset reorganization complete!")