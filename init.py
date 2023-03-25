import warnings
warnings.filterwarnings("ignore")

import nltk

nltk.download('stopwords')
nltk.download('punkt')

import utils
from constants import models_dir, dataset_dir

utils.check_dir(models_dir)
utils.check_dir(dataset_dir)
