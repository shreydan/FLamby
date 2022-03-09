## Camelyon16


Camelyon16 as Camelyon17 are open access (CC0), the original dataset is accessible [here](https://camelyon17.grand-challenge.org/Data/).
We will use the [Google-Drive-API-v3](https://developers.google.com/drive/api/v3/quickstart/python) in order to fetch the slides from the public Google Drive and will then tile the matter using a feature extractor producing a bag of features for each slide.

## Dataset description

|                   | Dataset description 
| ----------------- | -----------------------------------------------
| Description       | This is the dataset from Camelyon16
| Dataset           | 399 slides with labels (170 (Train) + 89 (Test) slides in Center0 (RUMC), 100 (Train)+ 50 (Test) slides in Center1 (UMCU))
| Centers           | Original Institutions from which WSIs originate RUMC and UMCU (2)
| Task              | Weakly Supervised Classification


## Download instructions

In order to use the Google Drive API we need to have a gmail account and to access the [google developpers console](https://console.cloud.google.com/apis/credentials/consent?authuser=1) in order to get a json containing an OAuth2.0 secret.  

All steps necessary to obtain the JSON are described in numerous places in the internet such as in pydrive's [quickstart](https://pythonhosted.org/PyDrive/quickstart.html), or in this [very nice tutorial's first 5 minutes](https://www.youtube.com/watch?v=1y0-IfRW114) on Youtube.
It takes 5 minutes and the important steps are listed below:

Step 1: Create a project in [Google console](https://console.cloud.google.com/apis/credentials/consent?authuser=1) you can call it flamby for instance   
Step 2: Go to Oauth2 consent screen (on the left of the webpage), choose a name for your app and publish it for external use   
Step 3: Go to Credentials, create id, then client oauth id  
Step 4: Choose Web app, go through the steps and **allow URI redirect** towards http://localhost:6006 and http://localhost:6006/ (notice the last backslash)  
Step 5: retrieve the secrets in JSON by clicking on Download icon at the end of the process  
Then copy-paste your secrets to the directory you want:
```
cp ~/Downloads/code_secret_client_bignumber.apps.googleusercontent.com.json client_secrets.json
````
**If you have ssh to a distant server make sure you do ssh forwarding of the port 6006 onto the port 6006 of your laptop.**
Make sure you have enough space to hold the dataset (900G).
Then run:
```
python download.py --output-folder ./camelyon16_dataset --path-to-secret /path/towards/client_secrets.json --port 6006
```
This will download all of Camelyon's slides in `./camelyon16_dataset`. As there are multiple
slides that are quite big, this script can take a few hours to complete. It can be stopped and
resumed anytime however if you are ssh into a server better use detached mode (screenrc/tmux/etc.).

**IMPORTANT :** If you choose to relocate the dataset after downloading it, it is
imperative that you run the following script otherwise all subsequent scripts will not find it:
```
python update_config.py --new-path /new/path/towards/dataset
```


The next step is to tile the matter on each slide with a feature extractor pretrained on IMAGENET.  

We will use the [histolab package](https://github.com/histolab/histolab) to segment the matter on each slide and torchvision to download a pretrained ResNet50 that will be applied on each tile to convert each slide to a numpy feature.
One can chose to remove or not the original slides that take up quite some space to keep only the features (therefore using only approximatively 50G instead of 800).
Again as extracting the matter on all the slides is a lengthy process this script might take a few days. 
It can also be stopped and resumed anytime and should be preferably run in detached mode.
This process should be run on an environment with GPU otherwise it might be prohibitively slow.

```
python tiling_sides.py --batch-size 64
```
or
```
python tiling_sides.py --batch-size 64 --remove-big-tiff
```

You can check the dataset is ready for use by running:

## Using the dataset

Now that the dataset is ready for use you can load it using the low or high-level API
by doing:
```python
from flamby.datasets.fed_camelyon16 import FedCamelyon16, Camelyon16Raw

# To load the first center
center1 = FedCamelyon16(center=0, train=True)
# To load the second center
center1 = FedCamelyon16(center=1, train=True)
```

# TODO benchmarking the baselines WP2


