# cardiac-scn

This project adapts the [SpatialConfiguration-Net](https://www.sciencedirect.com/science/article/pii/S1361841518305784) and the [Adaptive Wing Loss](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Adaptive_Wing_Loss_for_Robust_Face_Alignment_via_Heatmap_Regression_ICCV_2019_paper.pdf) to detect 32 cardiac landmarks in 3D echocardiography images and then reconstructs 7 cross-section views of the heart based on the predicted landmark locations. 

## Usages
### 1. Install dependencies

```
conda env create -f env.yaml
conda activate comp4801
```

### 2. Place raw data

- Place raw `json` annotations inro `data/raw/json/`.
- Place raw `nrrd` images into `data/raw/nrrd/`.

The directory tree should be like this:

```
cardiac-scn
├─ data
│  └─ raw
│     ├─ json
│     │  ├─ 2021-09-17-annotations
│     │  │  ├─ PWHOR191529000T_17Sep2021_BPCZ8ERS_3DQ.json
│     │  │  ├─ PWHOR191529000T_17Sep2021_BPCZ8ESW_3DQ.json
│     │  │  └─ ...
│     │  ├─ 2021-09-20-annotations
│     │  │  └─ ...
│     │  └─ ...
│     └─ nrrd
│        ├─ 2021-09-17-3d-nrrd
│        │  ├─ PWHOR191529000T_17Sep2021_BPCZ8ERS_3DQ.seq.nrrd
│        │  ├─ PWHOR191529000T_17Sep2021_BPCZ8ESW_3DQ.seq.nrrd
│        │  └─ ...
│        ├─ 2021-09-20-3d-nrrd
│        │  └─ ...
│        └─ ...
└─ ...
```

### 3. Preprocess data

1. Parse all raw `json` annotations into `csv` files.

    ```
    python _parse_raw.py
    ```

    - The results will be saved in `data/meta/4d_ijk/$VIEW/`.

2. Preprocess data (extract 3D from 4D and resize); generate groundtruth heatmaps (place Gaussian excitations); split the dataset into training, validation, and testing sets.

    ```
    python _preprocess.py --views $VIEW
    python _generate_truth.py --views $VIEW
    python _split.py --views $VIEW --test "TEST.txt"
    ```
    where `$VIEW` is one of is a list of cross-section view abbreviations separated by `,`, *e.g.*, `VIEW="A2C,A4C,SAXA,SAXB,ALAX,SAXMV,SAXM"`.

    - `TEST.txt` is a list of fixed test filenames. 
    - This process may take a while. Recommended to process one view at a time.   
    - The processed data will be saved in `data/nrrd/$VIEW/`; the groundtruth heatmaps will be saved in `data/truth/$VIEW/`; the dataset meta will be saved in `data/meta/train/$VIEW/`, `data/meta/val/$VIEW/`, `data/meta/train_val/$VIEW/`, and `data/meta/test/$VIEW/`.

### 4. Train the model

Train with `gpu-interactive`.

```
python train.py --config $CFG
```

- `$CFG` is the config file. Example config files are provided in `configs/`.
- Checkpoints will be saved in `pths/$VIEW/`; logs will be saved in `logs/$VIEW/`.

### 5. Evaluate the model

Calculate the Euclidean distance between the predicted and groundtruth landmark locations.

```
python evaluate.py --view $VIEW --pth_path $CKPT
```

- `$VIEW` is the view abbreviation.
- `$CKPT` is the path to the `pth` file.
- The results will be saved in `evaluation/$VIEW/`.

### 6. Reconstruct and visualize the cross-section views

- For A2C, A4C, ALAX, use the general SVD method.
    ```
    python recover.py --view $VIEW --pth_path $CKPT
    ```
- For SAXA, jointly use SAXA's, SAXM's and SAXMV's landmarks to predict the normal. 
    ```
    python recover_saxa.py --view "SAXA" --pth_path $SAXA_CKPT --saxm_pth_path $SAXM_CKPT --saxmv_pth_path $SAXMV_CKPT
    ```
- For SAXB, ignore PV-tip when fitting the plane.
    ```
    python recover_saxb.py --view "SAXB" --pth_path $SAXB_CKPT
    ```
- For SAXM, SAXMV, either use the general SVD method.
    ```
    python recover.py --view $VIEW --pth_path $CKPT
    ```
    or jointly use each other's landmarks to predict the normal. 
    ```
    python recover_saxa.py --view $VIEW --pth_path $CKPT --saxm_pth_path $SAXM_CKPT --saxmv_pth_path $SAXMV_CKPT
    ```
    - The results will be saved in `results/$VIEW/`.
    - `fit.csv` records the predicted centroid and normal vector of the cross-section plane.
    - `err.csv` records the distances from predicted landmarks to the predicted cross-section plane.
    - `images/` contains visualizations of the cross-section views.
