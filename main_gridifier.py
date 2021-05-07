from image_gridifier import image_gridifier
import main_params as mp

# ---

train_images = 'D:\\Users\\lwrnc\\Desktop\\LWRNC\\11 - A\\Research 2\\R2X1-03_landuseproj\\train_imgdata\\satellite_4800x4800_unsorted'
train_labelses = 'D:\\Users\\lwrnc\\Desktop\\LWRNC\\11 - A\\Research 2\\R2X1-03_landuseproj\\train_imgdata\\colormap_20x20_unsorted'

test_images = 'D:\\Users\\lwrnc\\Desktop\\LWRNC\\11 - A\\Research 2\\R2X1-03_landuseproj\\test_imgdata\\satellite_4800x4800_unsorted'
test_labelses = 'D:\\Users\\lwrnc\\Desktop\\LWRNC\\11 - A\\Research 2\\R2X1-03_landuseproj\\test_imgdata\\colormap_20x20_unsorted'

image_gridifier.generate_sorted_grid_image_files_by_directory(
    images_directory=test_images,
    labelses_directory=test_labelses,
    subimage_height=mp.img_HEIGHT,
    subimage_width=mp.img_WIDTH,
    lookup_colors=mp.lookup_rgb_to_index_full,
    label_names=mp.label_names_full
)
