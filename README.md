# DM-NMR

## Reproduce The Results In The Paper

### Environment

Navigate to the data directory：`Diff-1D/` or `Diff-2D/`
Execute the commands `uv sync`
The environment is following:
    `Diff-1D/pyproject.toml`
    `Diff-2D/pyptoject.toml`

### Download pre-trained model weights

pre-trained model can be download in 'https://drive.google.com/drive/folders/1fIRTAn99VXY0hhibAW9lRcz-Ilsb9ug9?usp=sharing', and put them into `Diff-1D/model/` or `Diff-2D/model/`.

### For 1D model

Set `image_size: ` and `direct_size: ` in the `config/nmr_20250325.yml` file to satisfy the data size.
Execute the following commands to reconstruct data:
    `main.py --target_path the_fully_sampling_data_path --mask_path the_sampling_mask_path`
Then the reconstruction data will be set in `Process_data`.

If you want to retrain the model, run `Matlab_process/generate1D.m` to create a simulated dataset, and run `scripts/image_train` to train the model.

### For 2D model

Set `row_size: ` and `col_size: ` in the `config/nmr_20250102.yml` file to satisfy the data size.
Execute the following commands to reconstruct data:
    `main.py --target_path the_fully_sampling_data_path --mask_path the_sampling_mask_path --direct_dim the_direct_dim_of_data`
Set `data_path_idx`, `nmrpipe_dat_path`, `com_path` and `test_path` in the `postprocess.py` file to satisfy the data path.
Then run the `postprocess.py` to get the reconstruction frequency domain data. The reconstruction frequency domain data will be set in `Res_data`.

If you want to retrain the model, run `Matlab_process/generate3DFID_SP.m` to create a simulated dataset, and run `scripts/image_train` to train the model.
