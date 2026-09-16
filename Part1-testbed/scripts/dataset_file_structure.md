# Dataset Structure

- Folder: `<datetime>_<random seed>`
    - `params.json`
        - `random_seed`
        - `num_waypoints`
        - `num_frames`
        - `camera_params`
            - `set to CAMERA_CONFIGS from util.py`
    - Folder: `videos`
        - `camera_XX.mp4`
        - `...`
    - Folder: `y`
        - `camera_XX_truth.csv`
            `x,y,z,thetas,speed,car_visible`
        - `...`
    