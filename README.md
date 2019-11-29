# Gaussian Prototypical Networks for Few-Shot Learning
## Development version - use with care

#### Disclaimer
This repository contains code in its developmental form.

#### Referencing
If you find this code useful, consider giving it credit by citing: Fort, S. (2018). Gaussian Prototypical Networks for Few-Shot Learning on Omniglot. arXiv preprint arXiv:1708.02735. The link is https://arxiv.org/abs/1708.02735.

### Intro
This repository contains the original TensorFlow implementation of a Gaussian Prototypical Network from <a href = https://arxiv.org/abs/1708.02735>Gaussian Prototypical Networks for Few-Shot Learning on Omniglot</a>. The code is set to work with the Omniglot dataset (accessible at https://github.com/brendenlake/omniglot, citation: Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction. Science, 350(6266), 1332-1338.) I cleaned the code from historical baggage, but a lot of things does not have convenient switches, and needs to be done in the code directly. This repository contains shell scripts that will download and prepare Omniglot to be used by the code directly.

### First run
<b>To get the code running for the first time:</b> 
1. Make all scripts executable with chmod +x and run `./make_folders.sh` to generate necessary empty folders for images and checkpoints
2. Run `get_omniglot.sh` in the `data/` folder to download and unzip the Omniglot dataset for you.
3. Run `get_lists.sh` in `data/` folder to generate lists of available alphabets and characters.
4. You're good to go! Run `classifier7.py` in the root folder. The training should start automatically on a small subset of the data and with a small encoder.

There are several choice you can make and I detailed them in the code. The architecture and run wrapper are in `utils/cnn6.py`. The majority of hyperparams are set in `classifier7.py` which you should run for both train and test.

Without training, you should expect the training accuracy of `1/N_classes`, as the decision is random at first. This should quickly improve on the small subset of Omniglot that is set as default in this repo. Even on a CPU, you should be able to overtrain in minutes. 

### Full-scale run

#### Data settings
In `classifier7.py` use `loadOmniglot(....,train = 0, limit = None,...)` to get the full Omniglot dataset loaded and used.

#### Hyperparameters
In `classifier7.py` you can choose hyperparameters of the training and test. The whole thing is described in https://arxiv.org/abs/1708.02735 but in general, the training is done in a regime with `N_classes` in each batch. `N_classes = 60` seems to work well -- it's better to have more demanding training. `N_classes_val` is the same but for validation/test, and on Omniglot people look at 5 and 20. `N_support` is the number of points that define a class during training, i.e. the <math>k</math> of the <math>k</math>-shot classification. `N_query` is the number of images per class to classify during training. Since Omniglot has 20 images per class, set it to `N_query = 20 - N_support`. `embed_dim` is the dimensionality of the embedding space.

#### Sigma estimates
Error bound estimates around embedding points can be realized in 3 ways:
+ not at all, `sigma_mode = "constant"`
+ one real number per image, `sigma_mode = "radius"`
+ `embed_dim` real numbers per image, `sigma_mode = "diagonal"`

The more free parameters in the sigma, the worse the training, but the more resistant the system is to currupted and inhomogeneous data. To play with partially corrupted data, you can set `damage = True` in `classifier7.py` and set your own downsampling specification.

#### Encoder settings
In `utils/cnn6.py` the encoder hyperparams are set by `num_filters`. The small version corresponds to `[64,64,64,embed_dim + sigma_dim]` and the large one to `[128,256,512,embed_dim + sigma_dim]`.
