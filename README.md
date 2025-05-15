# S2O: Static to Openable Enhancement for Articulated 3D Objects

[Denys Iliash](https://scholar.google.com/citations?user=guExFlYAAAAJ&hl=en&oi=ao)<sup>1</sup>,
[Hanxiao Jiang](https://jianghanxiao.github.io)<sup>2</sup>,
[Yiming Zhang](https://scholar.google.ca/citations?user=scUaE38AAAAJ&hl=en)<sup>1</sup>,
[Manolis Savva](https://msavva.github.io)<sup>1</sup>,
[Angel X. Chang](https://angelxuanchang.github.io/)<sup>1,3</sup>

<sup>1</sup>Simon Fraser University, <sup>2</sup>Columbia University, <sup>3</sup>Canada-CIFAR AI Chair, Amii

### [Project Page](https://3dlg-hcvc.github.io/s2o/)

This repo contains the code for S2O paper. Data can be found on [HuggingFace](https://huggingface.co/datasets/3dlg-hcvc/s2o).

In the Static to Openable (S2O) task, we aim to convert static meshes of container objects to articulated openable objects.

We develop a three stage pipeline consisting of 1) part segmentation, 2) motion prediction, and 3) interior completion.

<img src='docs/static/images/teaser.png'/>

## Installation
    conda create -n s2o python=3.10 pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    conda activate s2o
    conda install gxx_linux-64=9.5.0 -c anaconda
    conda install openblas-devel -c anaconda
    git clone git@github.com:NVIDIA/MinkowskiEngine.git
    cd MinkowskiEngine
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
    pip install -r requirements_minimal.txt

See `environment.yml` for the versions of the missing packages.

Additionally, follow instructions in the submodules you would like to use in order to install required libraries and build some dependencies from source.

## Data

Data and checkpoints can be found on [HuggingFace](https://huggingface.co/datasets/3dlg-hcvc/s2o).

Please request access. Remember, to authenticate with huggingface-cli. After you are approved, you can download the data with git lfs.
```
git lfs install
git clone git@hf.co:datasets/3dlg-hcvc/s2o
```

# Running the Static to Openable Pipeline

## scene-toolkit preliminaries

Some of the scipts used for S2O are integrated in [scene-toolkit](https://github.com/smartscenes/sstk). For instance, if you need to obtain connectivity segmentation (i.e. for running the demo), you need to run `node ssc/segment-mesh.js --input <path> --input_type path --format <format> --segmentator_method connectivity --segmentator_format trimesh`.

## Demo

See `pipeline_exp/pipeline_connectivity.py` for the code that runs segmentation and heuristic-based motion prediction, given a mesh as an input.

## Part Segmentation

We explore different methods (point cloud based, image based, and mesh based) for identifying openable parts and segmenting out the parts from the mesh.  

We provide checkpoints for the different models at https://huggingface.co/datasets/3dlg-hcvc/s2o.  Below we provide a summary of the different models, code directory, and their part segmentation performance.  We recommend using the FPNGroup mode, as well as variants of it trained on different data that can be found on our HuggingFace.

| Type | Method | code  | weights |  F1 on PM-Openable | F1 on ACD | 
|--------|------|-------|-------|-----|----|
| PC | [PointGroup](https://github.com/dvlab-research/PointGroup) | minsu3d |  [pg_unet.ckpt](https://huggingface.co/datasets/3dlg-hcvc/s2o/blob/main/ckpts/pg_unet.ckpt) | 42.1 | 4.1 |      
| PC | [Mask3D](https://github.com/JonasSchult/Mask3D) | Mask3D |  [mask3d.ckpt](https://huggingface.co/datasets/3dlg-hcvc/s2o/blob/main/ckpts/mask3d.ckpt) | 42.9 | 4.8 |
| PC | FPNGroup | internal_pg |  [pg_px_fpn.ckpt](https://huggingface.co/datasets/3dlg-hcvc/s2o/blob/main/ckpts/pg_px_fpn.ckpt)  | **81.5** | **11.9** |
| Mesh | [MeshWalker](https://github.com/AlonLahav/MeshWalker) | MeshWalker | [meshwalker.keras](https://huggingface.co/datasets/3dlg-hcvc/s2o/tree/main/ckpts/meshwalker) | 1.0 | 0.9 |
| Image | [OPDFormer](https://github.com/3dlg-hcvc/OPDMulti) | OPDMulti |  [opdformer_p.pth](https://huggingface.co/datasets/3dlg-hcvc/s2o/blob/main/ckpts/opdformer_p.pth) | 0.9 |  1.3 |


For PC-based methods run:

    # Pre-processing
    python scripts/preprocess/create_subset_points.py --data_path {path/to/pcd/downsample.h5} --data_json {path/to/split/json}
    
    # For all PointGroup methods convert to minsu3d format
    python scripts/preprocess/prepare_for_minsu3d.py --data_path {path/to/pcd-subset/downsample.h5} --data_json {path/to/split/json}. For training of FPNGroupMot run scripts/preprocess/prepare_for_minsu3d_w_mot.py instead
    
Follow submodule instructions for inference. Then, for post-processing and mapping run:

    # Post-processing
    
    # Map predictions from subset to full point clouds
    python scripts/postprocess/map_predictions_from_subset_points.py --exp_dir {path/to/predictions} --data_path {path/to/pcd/downsample.h5} --subset_path {path/to/pcd-subset/downsample.h5} --output_path {path/to/full/predictions}
    
    # Map full predictions to mesh, use --gt flag with this script to generate gt for evaluation
    python scripts/postprocess/map_pc_to_mesh.py --{path/to/full/predictions} --data_path {path/to/processed_mesh} --data_json {path/to/split/json} --sampled_data {path/to/pcd/downsample.h5} --output_dir {path/to/mapped/meshes/output}

    # If you want to vote for oversegments rather than triangles during propagation, use scripts/postprocess/map_pc_to_mesh_connectivity.py instead (recommended).

## Motion prediction

To run heuristic motion prediction:
    python motion_inference.py --pred_path {path/to/mapped/meshes/output} --output_path {path/to/mapped/meshes/output/motion} --export

If obtained the motion predictions from FPNGroupMot, run post-processing with `scripts/postprocess/postprocess_fpngroupmot_motion.py`.


# Reproducing Experiments

## Evaluation
PC metrics are obtained from minsu3d eval.py and OC-cost demo.py, follow the instructions from the submodules. For mesh segmentation and motion prediction evaluation:

    # GT is obtained from running map_pc_to_mesh with --gt flag
    python mesh_eval.py --predict_dir {path/to/mapped/meshes/output} --gt_path {path/to/preprocessed/gt} --output_dir {dir/for/logged/metrics} --data_json {path/to/split/json} --glb_path {path/to/processed_mesh}

    # For metrics from the supplement
    python mesh_eval_seg.py --predict_dir {path/to/mapped/meshes/output} --gt_path {path/to/preprocessed/gt} --output_dir {dir/for/logged/metrics} --data_json {path/to/split/json} --glb_path {path/to/processed_mesh}

    # For motion evaluation
    python motion_eval.py --predict_dir {path/to/mapped/meshes/output} --output_dir {dir/for/logged/metrics} --data_json {path/to/split/json} --glb_path {path/to/processed_mesh}

    # For interior evaluation
    python eval_interior_completion.py --predict_dir {path/to/meshes/with/completed/interiors} --gt_path {path/to/meshes/with/gt/interiors} --data_json {path/to/split/json} --output_dir {dir/for/logged/metrics}


## Training 

See documentation of the corresponding submodules desired to train.
    
## Citation
Please cite our work if you use S2O results/code or ACD dataset.
```
@article{iliash2024s2o,
  title={{S2O}: Static to openable enhancement for articulated {3D} objects},
  author={Iliash, Denys and Jiang, Hanxiao and Zhang, Yiming and Savva, Manolis and Chang, Angel X},
  journal={arXiv preprint arXiv:2409.18896},
  year={2024}
}
```
