# Instructions to Run the Code

check out website, it has small animations!

## 1. Ensure the Images Are Available
- Add a folder named `data` in the same directory as `main.ipynb`.
- The `data/` folder should contain the images (provided in the zip file) that will be processed by the code.

## 2. Open the Jupyter Notebook
- Open `main.ipynb`  This notebook is the main entry point and contains all the code to process the images.

## 3. Run the Notebook
- Ensure that all dependencies are installed (see below).
- Simply run all cells in `main.ipynb`. 
- It will process each image in the list, align the channels, and save the final aligned images in the `out_path/` directory.

## 4. View the Output
- After running the notebook, aligned images will be saved as `.jpg` files in the `out_path/` folder (e.g., `aligned_cathedral.jpg`, `aligned_emir.jpg`, etc.).

## Dependencies
To run this notebook, you need the following Python libraries:
- numpy
- skimage
- matplotlib

You can install these dependencies using pip: