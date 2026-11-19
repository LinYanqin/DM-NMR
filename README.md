# DM-NMR

### Environment

This project uses Python 3.6.3 and torch 1.10.1. We recommend creating a dedicated Conda environment. Run the following commands from the project root:

```
conda create -n dm-nmr python=3.6.3 -y
conda activate dm-nmr

python -m pip install pip==21.3.1 setuptools==59.6.0 wheel==0.37.1
python -m pip install -r requirement.txt --find-links https://download.pytorch.org/whl/cu102/torch_stable.html
```

GPU execution requires a compatible NVIDIA driver. NMRPipe, csh, and MATLAB are not included in the Python dependencies and must be installed separately for the workflows that use them.

### Download pre-trained model weights

pre-trained model can be download in 'https://drive.google.com/drive/folders/1fIRTAn99VXY0hhibAW9lRcz-Ilsb9ug9?usp=sharing', and put them into `Diff-1D/model/` and `Diff-2D/model/`, respectively.

### For 1D model

Set `image_size: ` and `direct_size: ` in the `config/nmr.yml` file to satisfy the data size.
Execute the following commands to reconstruct data:
    `main.py --target_path the_fully_sampling_data_path --mask_path the_sampling_mask_path`
Then the reconstruction data will be set in `Process_data`.
`Matlab_process/test_BMRB.m` is used to show reconstructed results.

If you want to retrain the model, run `Matlab_process/generate2D.m` to create a simulated dataset, and run `scripts/image_train.py` to train the model.

### For 2D model

Set `row_size: ` and `col_size: ` in the `config/nmr.yml` file to satisfy the data size.
Execute the following commands to reconstruct data:
    `main.py --target_path the_fully_sampling_data_path --mask_path the_sampling_mask_path --direct_dim the_direct_dim_of_data`
Set `data_path_idx`, `nmrpipe_dat_path`, `com_path` and `test_path` in the `postprocess.py` file to satisfy the data path.
Then run the `postprocess.py` to get the reconstruction frequency domain data. The reconstruction frequency domain data will be set in `Res_data`.
`Matlab_process/test_BMRB.m` is used to show reconstructed results.

If you want to retrain the model, run `Matlab_process/generate3DFID.m` to create a simulated dataset, and run `scripts/image_train.py` to train the model.

## Example Data and Reproduction of Experimental Results

To facilitate reproducibility, we provide experimental NMR datasets and sampling masks for reproducing the DM-NMR reconstruction results reported in Figures 3, 4, and 6 of our paper.

The example datasets include a 2D ¹⁵N–¹H HSQC spectrum (BMRB 15430) and a 3D HNCO spectrum (BMRB 16101). The pretrained diffusion models, reconstruction scripts, and evaluation programs are also provided.

### 1. Example Datasets

| Dataset    | Experiment     | Spectral size | Corresponding figures     | Model              |
| ---------- | -------------- | ------------- | ------------------------- | ------------------ |
| BMRB 15430 | 2D ¹⁵N–¹H HSQC | 370 × 128     | Figures 3, 6(a), and 6(c) | 1D diffusion model |
| BMRB 16101 | 3D HNCO        | 280 × 32 × 64 | Figures 4, 6(b), and 6(d) | 2D diffusion model |

**2D NMR example (BMRB 15430)**

The fully sampled reference spectrum is provided at:

`Diff-1D/nmrpipe_data/fully_sampled/test.ft2`

The associated NMRPipe preprocessing script is:

`Diff-1D/nmrpipe_data/fully_sampled/ft.com`

The sampling mask used for Figure 3 is:

`Diff-1D/mask/15/mask1.txt`

**3D NMR example (BMRB 16101)**

The fully sampled time-domain reference data are provided at:

`Diff-2D/nmrpipe_data/fully_sampled/fid_temp.dat`

The associated preprocessing script is:

`Diff-2D/nmrpipe_data/fully_sampled/process.com`

The sampling mask used for Figure 4 is:

`Diff-2D/mask/3/mask1.txt`

The provided example data have already been prepared for reconstruction. No additional data preprocessing is required before running the reconstruction scripts.

### 2. Pretrained Models

The pretrained diffusion models used in the paper can be downloaded through the Google Drive links provided in the following directories:

| Model              | Directory       | Training checkpoint | Application           |
| ------------------ | --------------- | ------------------- | --------------------- |
| 1D diffusion model | `Diff-1D/model` | 100,000 steps       | 2D NMR reconstruction |
| 2D diffusion model | `Diff-2D/model` | 80,000 steps        | 3D NMR reconstruction |

Each pretrained model is reused across the evaluated sampling rates and sampling schemes for its corresponding reconstruction task, without retraining.

### 3. Reproducing Figures 3 and 4

The reconstruction scripts contain the default configurations for the provided example datasets.

**Figure 3: 2D HSQC reconstruction**

Figure 3 presents the reconstruction of the BMRB 15430 HSQC spectrum at a 15% Poisson gap sampling rate.

Run the reconstruction script:

```bash
cd Diff-1D
python main.py
```

The reconstructed spectra are saved as MATLAB `.mat` files.

To calculate the relative L2-norm error (RLNE) and visualize the reconstructed spectrum, run the MATLAB script:

`Diff-1D/Matlab_process/test_BMRB.m`

The reported RLNE for DM-NMR in Figure 3 is **0.2687**.

**Figure 4: 3D HNCO reconstruction**

Figure 4 presents the reconstruction of the BMRB 16101 HNCO spectrum at a 3% Poisson gap sampling rate.

Run the reconstruction script:

```bash
cd Diff-2D
python main.py
```

**Note:** The 3D NMR postprocessing procedure requires NMRPipe, which must be available in the C shell (`csh`) environment. Before running the postprocessing script, ensure that NMRPipe is installed and its environment is properly configured.

After reconstruction, execute the postprocessing script:

```bash
cd Diff-2D
python postprocess.py
```

The postprocessed spectra are saved as MATLAB `.mat` files.

To calculate RLNE and visualize the reconstructed spectrum, run:

`Diff-2D/Matlab_process/complute_RLNE.m`

The reported RLNE for DM-NMR in Figure 4 is **0.2345**.

### 4. Reproducing Figure 6

Figure 6 evaluates DM-NMR reconstruction performance across different sampling rates and sampling schemes using the same experimental datasets as Figures 3 and 4.

- Figures 6(a) and 6(b) evaluate different sampling rates using Poisson gap sampling.
- Figures 6(c) and 6(d) compare Poisson gap, T2 weighted, and random sampling at fixed sampling rates of 25% for the 2D dataset and 8% for the 3D dataset.

For each sampling condition, ten independently generated sampling masks were used. The corresponding masks are provided in:

`Diff-1D/mask/`

`Diff-2D/mask/`

The supported sampling scheme identifiers are:

| Identifier | Sampling scheme       |
| ---------- | --------------------- |
| `Pois`     | Poisson gap sampling  |
| `Exp`      | T2 weighted sampling  |
| `Rand`     | Fully Random sampling |

**Figures 6(a) and 6(c): 2D NMR reconstruction**

Use `Diff-1D/main_SR.py` to reconstruct the BMRB 15430 dataset under a specified sampling rate and sampling scheme:

```bash
cd Diff-1D
python main_SR.py --mask_type Pois --sample_rate RATE
```

For the comparison of sampling rates in Figure 6(a), use `--mask_type Pois` with `--sample_rate` set to one of the 12 tested rates (10%, 15%, 20%, 25%, 30%, 35%, 40%, 50%, 60%, 70%, 80%, or 90%). For the comparison of sampling schemes in Figure 6(c), use `--sample_rate 25` with each of `Pois`, `Exp`, and `Rand`.

The program performs reconstruction using the ten masks corresponding to the selected sampling condition and saves the results as `.mat` files.

To calculate the RLNE values for the ten reconstructions, run:

`Diff-1D/Matlab_process/test_SR.m`

**Figures 6(b) and 6(d): 3D NMR reconstruction**

Use `Diff-2D/main_SR.py` to reconstruct the BMRB 16101 dataset:

```bash
cd Diff-2D
python main_SR.py --mask_type Pois --sample_rate RATE
```

For the comparison of sampling rates in Figure 6(b), use `--mask_type Pois` with `--sample_rate` set to one of the 8 tested rates (4%, 6%, 8%, 10%, 12%, 16%, 20%, or 30%). For the comparison of sampling schemes in Figure 6(d), use `--sample_rate 8` with each of `Pois`, `Exp`, and `Rand`.

After reconstruction, run the corresponding postprocessing script with NMRPipe configured, as described in Section 3:

```bash
cd Diff-2D
python postprocess_SR.py --mask_type Pois --sample_rate RATE
```

The sampling scheme and sampling rate can be specified using the same argument names as for the 2D diffusion model.

The postprocessed results are saved as `.mat` files.

To calculate the RLNE values for the ten reconstructions, run:

`Diff-2D/Matlab_process/complute_RLNE_SR.m`

For each sampling condition, Figure 6 reports the mean RLNE across the ten masks, with error bars representing ± one standard deviation.

The reconstruction parameters and pretrained model weights remain unchanged across the evaluated sampling conditions. Random seeds are fixed in the provided scripts.

### 5. Reconstruction Settings

The following parameters were used for the experimental reconstructions:

| Parameter                          | 2D NMR reconstruction   | 3D NMR reconstruction   |
| ---------------------------------- | ----------------------- | ----------------------- |
| Diffusion model                    | 1D diffusion model      | 2D diffusion model      |
| Reverse diffusion iterations       | 1000                    | 1000                    |
| DDIM noise mixing parameter zeta   | 0.1                     | 0.1                     |
| Low-rank projection interval       | Every 100 reverse steps | Every 100 reverse steps |
| Hankel patch size                  | 32                      | 8 × 8                   |
| Retained rank                      | 6                       | 25                      |
| Data consistency                   | Every reverse iteration | Every reverse iteration |

At each reverse iteration, the trained diffusion model first estimates the clean spectrum from the predicted noise. A time-domain low-rank projection is selectively applied, followed by data consistency using the acquired measurements. The corrected estimate is then used for the DDIM-based reverse update. In the final two reverse iterations, the data-consistent clean-spectrum estimate is directly used without an additional DDIM update.

### 6. Evaluation and Reproducibility

Reconstruction accuracy is evaluated using the relative L2-norm error (RLNE) against the fully sampled reference spectrum.

Before RLNE calculation, the real components of the reconstructed and reference spectra are independently normalized by their respective maximum real-part intensities.

The provided MATLAB scripts perform the corresponding RLNE calculations and reconstruction visualizations.
**MATLAB working directory:** Before running the evaluation scripts, set the MATLAB current folder to `Diff-1D\Matlab_process` for 2D NMR reconstruction or `Diff-2D\Matlab_process` for 3D NMR reconstruction. This ensures that relative paths to the reconstructed spectra and reference data are resolved correctly.

The reconstruction scripts use the configurations provided for the experimental datasets, and users do not need to preprocess the example data.
