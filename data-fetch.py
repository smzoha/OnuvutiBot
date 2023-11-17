import os
import kaggle

DATASET_NAME = 'raseluddin/bengali-empathetic-conversations-corpus'
DATASET_PATH = './data'

if not os.path.exists(DATASET_PATH):
    os.mkdir(DATASET_PATH)

# Kaggle API requires API keys. Please download your own and
# place it in the suitable directory (i.e. ~/.kaggle/kaggle.json)
print('Authenticating')
kaggle.api.authenticate()
print('Authentication Successful!')

kaggle.api.dataset_download_files(DATASET_NAME, DATASET_PATH, quiet=False, unzip=True)
print('Dataset can be found in the ./data directory')
