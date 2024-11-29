import os

def get_subfolder_structure(base_path, category, sample_ratio=0.5):
    """
    Recursively get all subfolders and image paths for a category,
    selecting the first half of images from each subfolder.
    
    Args:
        base_path (str): Base directory path
        category (str): Main category name (e.g., "1_change_object_80")
        sample_ratio (float): Ratio of first images to select (0.5 means first half)
    
    Returns:
        dict: Dictionary with folder paths as keys and list of selected image names as values
    """
    category_path = os.path.join(base_path, category)
    folder_images = {}
    
    def is_image_file(filename):
        """Check if file is an image"""
        return filename.lower().endswith(('.jpg', '.jpeg', '.png'))
    
    def get_first_half_images(image_list, ratio=0.5):
        """Get first n images from the list based on ratio"""
        # Sort images to ensure consistent ordering
        sorted_images = sorted(image_list)
        # Calculate how many images to take
        n = max(1, int(len(sorted_images) * ratio))
        # Take the first n images
        selected = sorted_images[:n]
        print(f"Selected {len(selected)}/{len(image_list)} images from folder")
        return selected
    
    def explore_folder(folder_path, relative_path):
        """Recursively explore folder structure"""
        items = os.listdir(folder_path)
        
        # Get all image files in current folder
        images = [f for f in items if is_image_file(f)]
        if images:
            print(f"\nProcessing folder: {relative_path}")
            print(f"Total images found: {len(images)}")
            selected_images = get_first_half_images(images, sample_ratio)
            if selected_images:
                folder_images[relative_path] = selected_images
        
        # Process subfolders
        for item in sorted(items):  # Sort to ensure consistent processing order
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                explore_folder(item_path, os.path.join(relative_path, item))
    
    explore_folder(category_path, category)
    
    # Print summary of selected images
    print("\nSelection Summary:")
    for folder, images in folder_images.items():
        print(f"\n{folder}:")
        print(f"Selected images: {len(images)}")
        print("First few images:")
        for img in sorted(images)[:5]:
            print(f"  - {img}")
    
    return folder_images

def get_selected_image_paths(folder_images):
    """Convert folder_images dictionary to a list of full relative paths."""
    selected_paths = []
    for folder, images in sorted(folder_images.items()):
        for image in sorted(images):
            selected_paths.append(os.path.join(folder, image))
    return selected_paths