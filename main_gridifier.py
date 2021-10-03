from pathlib import Path

from image_gridifier import image_gridifier
import main_params as mp

# ---

# create local copy of main_gridifier.py and 

train_images = str(
    Path.cwd()
    / 'datasets'
    / 'train_images'
    / 'satellite_4800x4800_unsorted'
)
train_labelses = str(
    Path.cwd()
    / 'datasets'
    / 'train_images'
    / 'colormap_20x20_unsorted'
)

test_images = str(
    Path.cwd()
    / 'datasets'
    / 'test_images'
    / 'satellite_4800x4800_unsorted'
)
test_labelses = str(
    Path.cwd()
    / 'datasets'
    / 'test_images'
    / 'colormap_20x20_unsorted'
)

image_gridifier.generate_sorted_grid_image_files_by_directory(
    images_directory = test_images,
    labelses_directory = test_labelses,
    subimage_height = mp.img_HEIGHT,
    subimage_width = mp.img_WIDTH,
    lookup_colors = mp.lookup_rgb_to_index_full,
    label_names = mp.label_names_full
)
