# ToyDataset Generator for Reprojection
### *Nicolas GAUTTIER*

## Description

Many datasets with sets of observations are available for **Structure-from-Motion (DfM)** tasks but they often lack ground truths to allow for thorough, precise and controlled testing. This project provides a unified code base to create simple synthetic datasets as well as very realistic dataset.

A modular class `ToyModel` is used to instantiate, and then visualize prepare custom datasets from chosen landmarks and cameras or from 3D point cloud scenes

A custom adaptation of the **Z-buffering** is derived to recreate natural occlusions and select realistic observations.

---

## Create your own dataset

The `examples_notebook.ipynb` will walk you through a detailed explanation to create datasets in a controlled environment. The project's modularity allows easy integration of new functions to create specific datasets of observations and carry out precise testing.

To recreate a dataset from point clouds, one can use the models from the Tanks and Temples benchmark and store them in `\3d_models\`.

---

## Project structure

- `ToyDataset/`
    - `\src\`
        - `toymodel.py`: Contains the `ToyModel` and its subclasses `ToyModel_From_PointCloud` or `ToyModel_From_SavedNPZ` to instantiate the toy-model by different means.
        - `projection.py` : contains the class `ProjectionMixin` with methods to project and compute Z-buffer.
        - `visualization.py` : contains the class `VisualizationMixin` with methods to visualize landmarks, and cameras and projections in 2D and 3D space.
        - `exportation.py` : contains the class `ExportationMixin` with methods to export to save the model, to export the observations and to randomly initialize the problem before SfM and bundle adjustment.
    - `\utils\`
        - `camera_utils.py` : auxiliary functions useful for camera positioning and manipulation.
        - `landmarks_utils.py` : auxiliary functions useful for scene loading and generation.
    - `\exports\` : contains the exported .txt dataset to be used by the bundle adjustment pipeline.
    - `\import_results\` : contains the .txt output of bundle adjustment for one to visualize and compare.
    - `\saved_toy_dataset\` : contains the .npz files to recover an instance of `ToyModel` to avoid recomputing the observations through Z-buffering.
    - `\3d_models\` : contains 3d point clouds .ply models used as landmarks.
    - `example_dataset.ipynb`: a detailed walk-through on how to create, process, visualize and export a custom dataset.


On wayland you need to do first:
export XDG_SESSION_TYPE=x11
