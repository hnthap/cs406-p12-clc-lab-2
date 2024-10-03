from glob import glob
import json
import os
import random
import shutil
import sys

from PIL import Image
import cv2
import numpy as np
import streamlit as st
from tqdm import tqdm


def main():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR = os.path.join(CURRENT_DIR, 'data')

    seed_everything(2024)

    # Uncompress data
    prepare_images(IMAGE_DIR, CURRENT_DIR)

    # Generate histograms from images
    image_files = list(glob(
        os.path.join(IMAGE_DIR, 'seg/**/*.jpg'), 
        recursive=True
    ))
    histogram_list = get_histograms(
        image_files,
        os.path.join(CURRENT_DIR, 'histograms.json'),
    )

    # Prepare for the Streamlit app
    prepare_streamlit(image_files, histogram_list)


def prepare_images(images_dir: str, current_dir: str):
    '''
    This function uncompresses all the tarballs in the images_dir directory.
    If the images_dir directory already exists, it does nothing.

    Args:
        images_dir: The directory where the images will be uncompressed
        current_dir: The directory where the script is running
    '''
    if os.path.isdir(images_dir):
        return
    for file in glob(os.path.join(current_dir, 'compressed/**/*.tar'), 
                     recursive=True):
        print("unpacking '{}'".format(file))
        shutil.unpack_archive(file, current_dir)
    assert os.path.exists(images_dir)


def get_histograms(image_files: list[str], 
                   histograms_file: str) -> list[np.ndarray]:
    '''
    This function generates histograms for a list of images and saves them to a JSON file.
    If the histograms file already exists, it loads the histograms from the file instead of recalculating them.

    Args:
        image_files (list[str]): A list of paths to the image files for which histograms need to be generated.
        histograms_file (str): The path to the JSON file where the histograms will be saved or loaded from.
    '''
    # If histograms file exists, load it and return the results
    if os.path.exists(histograms_file):
        with open(histograms_file, 'r') as f:
            raw = json.load(f)
            histogram_list = list(map(np.array, raw['histograms']))
    else:
        # Otherwise, generate the histograms
        histogram_list = []
        for image_file in tqdm(image_files, ncols=78):
            gray_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            assert gray_image is not None, 'Failed to load image'
            assert len(gray_image.shape) == 2, 'Image is not grayscale'
            histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            histogram = histogram.flatten() / (histogram.sum() + 1e-5)
            histogram_list.append(histogram)
        
        # Save the histograms to a JSON file and also return them
        with open(histograms_file, 'w') as f:
            json.dump({ 
                'histograms': list(map(lambda x: x.tolist(), histogram_list))
            }, f)

    return histogram_list


def seed_everything(seed):
    '''
    This function sets the seed for both random and numpy for reproducibility.
    '''
    random.seed(seed)
    np.random.seed(seed)


def prepare_streamlit(image_files: list[str], 
                      histogram_list: list[np.ndarray]):
    title = 'Image Retrieval using Histogram Matching'
    st.set_page_config(title, '🥇', 'wide')
    st.header(title)
    [left, right] = st.columns([0.4, 0.6], gap='small',
                               vertical_alignment='top')
    with left:
        image = prepare_left_side()
    if image is not None:
        with right:
            prepare_right_side(image, image_files, histogram_list)
    

def prepare_left_side():
    with st.container(height=500):
        image = st.file_uploader('Upload an image', ['jpg', 'png', 'jpeg'])
        if image is None:
            st.error('Please provide an image to start')
            return image
        image = Image.open(image)
        width, height = image.size
        if width < height: # Portrait
            columns_spec = [0.3, 0.4, 0.3]
        else: # Landscape
            columns_spec = [0.2, 0.6, 0.2]
        [_, col_2, _] = st.columns(columns_spec, vertical_alignment='center',
                                   gap='medium')
        with col_2:
            st.image(image, use_column_width=True, caption='Uploaded Image')
    return image


def prepare_right_side(image_arg, image_files: list[str],
                       histogram_list: list[np.ndarray]):
    image = np.asarray(image_arg)
    closest_indices, distances = get_k_closest_images(
        image, 10, image_files, histogram_list
    )
    with st.container(height=500, border=True):
        [col_1, col_2, col_3, col_4] = new_st_columns()
        with col_1:
            st.subheader('Rank')
        with col_2:
            st.subheader('Image')
        with col_3:
            st.subheader('Distance')
        with col_4:
            st.subheader('Category')
        for rank, index in enumerate(closest_indices, start=1):
            [col_1, col_2, col_3, col_4] = new_st_columns()
            image_file = image_files[index]
            distance = distances[index]
            image = Image.open(image_files[index])
            with col_1:
                st.markdown(f'''
<p style="text-align: right; font-size: xx-large;">
    {get_rank_str(rank)}
</p>
''', unsafe_allow_html=True)
            with col_2:
                st.image(image, use_column_width=True)
            with col_3:
                st.text(f'{distance:.4f}')
            with col_4:
                st.text(image_file.split(os.path.sep)[-2].capitalize())


@st.cache_data
def get_k_closest_images(image, k, image_lists: list[str], 
                         histogram_list: list[np.ndarray]):
    assert image is not None, 'Image must be provided'
    assert type(image) == np.ndarray, 'Image must be a numpy array'
    assert k > 0, 'k must be a positive integer'
    assert k <= 100, 'k must be less than 100'
    assert k < len(image_lists), 'Not enough images in the dataset'
    if image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        assert image.ndim == 1, 'Image must be grayscale or RGB'
        gray_image = image
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    histogram = histogram.flatten() / (histogram.sum() + 1e-5)
    distances = [
        np.linalg.norm(histogram - hist)
        for hist in histogram_list
    ]
    sorted_indices = sorted(range(len(distances)), 
                            key=lambda i: distances[i])
    return sorted_indices[:k], distances


def get_rank_str(rank: int) -> str:
    if  rank == 1:
        return '🥇'
    elif rank == 2:
        return '🥈' 
    elif rank == 3:
        return '🥉'
    return str(rank)
    

def new_st_columns():
    return st.columns(
        [0.2, 0.3, 0.25, 0.25], gap='medium', vertical_alignment='center'
    )


if __name__ == '__main__':
    sys.exit(main())
