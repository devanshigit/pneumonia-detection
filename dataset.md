##   Dataset <a name="dataset"></a>

This project uses the Chest X-ray Pneumonia dataset from Kaggle: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Choose ONE of the following options to proceed:**

**Option A: Dataset Included in Repository (If feasible and license allows)**

The dataset is included in this repository under the `data/` directory.

The dataset files are organized as follows:

* `data/train/` (training images)
* `data/test/` (testing images)
* `data/val/` (validation images)

**Option B: Manual Download Instructions (Recommended)**

To use this project, you need to download the dataset manually from Kaggle:

1.  Go to the Kaggle dataset page: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2.  If required, sign up for a Kaggle account and agree to the dataset's terms and conditions.
3.  Download the dataset (usually as a ZIP file).
4.  Place the downloaded dataset in the `data/` directory of this project.
5.  The code in the Colab notebook will extract the necessary files.

**Option C: Kaggle API Download (Requires `kaggle.json` - Use with caution)**

This project can also download the dataset using the Kaggle API. To do this:

1.  Create a Kaggle account at [https://www.kaggle.com/](https://www.kaggle.com/).
2.  Go to your account settings and generate a new API token. This will download a file named `kaggle.json`.
3.  Place the `kaggle.json` file in the root directory of this project.
4.  The Colab notebook will use this file to download the dataset.
