from pathlib import Path

from image_gridifier import image_gridifier
import main_params as mp

# ---

# create local copy of main_gridifier.py and 

train_images = (Path       # C:\\path\\to\\folder\\of\\large_satellite_images (Training Set)
train_labelses = ''     # C:\\path\\to\\folder\\of\\colormaps (Training Set)

test_images = ''        # C:\\path\\to\\folder\\of\\large_satellite_images (Test Set)
test_labelses = ''      # C:\\path\\to\\folder\\of\\colormaps (Test Set)

image_gridifier.generate_sorted_grid_image_files_by_directory(
    images_directory=test_images,
    labelses_directory=test_labelses,
    subimage_height=mp.img_HEIGHT,
    subimage_width=mp.img_WIDTH,
    lookup_colors=mp.lookup_rgb_to_index_full,
    label_names=mp.label_names_full
)
