# Consistent Video NST

The goal of this project is to analyze different neural style transfer techniques, implement them, and tune them in order to produce visually pleasant content, especially consistent videos, which is a big challenge considering the instability of such process.

The project is pretty much composed of two parts. The first one concerns the standard technique as presented by Gatys et al. (2015), and the second one concerns a trained-network-based stylization process (instead of performing the whole optimization process for each image), inspired by Johnson et al. (2016). We tuned and modified both methods in order to try and produce best results, as detailed in the report `report.pdf`.

## Gatys et al. Part

The first part of the project is implemented in the files `vnst.py`, `losses.py`, `style_content_extractor.py` and `utils.py`. It allows for various options to run the different implementations we did in order to easily compare hyper-parameters.

###  <ins>Usage</ins>:

This can be run using the `vnst.py` file with Python. The arguments are the following:

| Parameter   | Short name  | Description | Values |
| ----------- | ----------- | ----------- | ----------- |
| `--content` | `-c` | Content file (name of file in folder `content`) | String |
| `--style` | `-s` | Style image (name of file in folder `style`) | String |
| `--output` | `-o` | Output file (name of file in folder `result`) | String |
| `--content_weight` | `-w` | Content weight | Float (def. `1`) |
| `--style_weight` | `-y` | Style weight | Float (def. `1`) |
| `--resize` | `-z` | Maximum dimension of an image | Integer (def. `512`) |
| `--temporal_loss` | `-T` | Use temporal loss? | Boolean (def. `true`) |
| `--temporal_weight` | `-W` | Temporal loss weight | Float (def. `1`) |
| `--target_start` | `-I` | Start from content or white noise | `content` (def.) or `random` |
| `--start_from_prev` | `-p` | Start from previous stylized frame warped | Boolean (def. `true`) |
| `--style_from_prev` | `-Y` | Style current frame using previous stylized frame | Boolean (def. `false`) |
| `--iters` | `-i` | Number of iterations for optimizing one frame | Integer (def. `300`) |
| `--optimizer` | `-a` | Optimizer algorithm | `lbfgs` (def) or `adam` |
| `--early-stopping` | `-e` | Stop early when good convergence | Boolean (def. `false`) |

One can them for example stylize an GIF `example.gif` with style `example.jpg` with Adam in 500 iterations using:

`python vnst.py -c example.gif -s example -a adam -i 500`

## Johnson et al. Part

The code for this part was done in a notebook: `fast_style_network.ipynb`. It contains instructions to run each part (training or applying).

Training as we did with about 65k images resized to 512x512 takes about 2 hours and 15 minutes on an RTX 3080. If you only want to apply one of the models stored in `models`, run all cells except the training one! Stylization can be done in the last cell, for an image or a GIF/mp4.

Frames are stored temporarily in the `frames_video` and `frames_video_stylized` folders. These should not really be considered.

For both parts of the project, content is stored in `content`, style images in `style`, and the results are stored in `result`.