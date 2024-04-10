from yacs.config import CfgNode as CN


which = 'train'  # val, test
_C = CN()

_C.AZURE_COMPUTER_VISION = CN()
_C.AZURE_COMPUTER_VISION.ENDPOINT_DOMAIN = 'TODO'
_C.AZURE_COMPUTER_VISION.KEY = 'TODO'

_C.AZURE_OPENAI = CN()
_C.AZURE_OPENAI.ENDPOINT_DOMAIN = 'TODO'
_C.AZURE_OPENAI.KEY = 'TODO'
_C.AZURE_OPENAI.DEPLOYMENT = 'TODO'

_C.AZURE_BLOB_STORAGE = CN()
_C.AZURE_BLOB_STORAGE.ACCOUNT_NAME = 'TODO'
_C.AZURE_BLOB_STORAGE.KEY = 'TODO'
_C.AZURE_BLOB_STORAGE.CONTAINER_NAME = 'TODO'
_C.AZURE_BLOB_STORAGE.SAS_TOKEN = 'TODO'

_C.LOCAL_FILES = CN()
_C.LOCAL_FILES.DIRPATH_VIDEOS = './data/siq2/video'
_C.LOCAL_FILES.FPATH_QA_JSON = f'./data/siq2/qa/qa_{which}.json' # NOTE: need to clean up JSON formatting: add "," to the end of each line (except for the last line) and close all with "[]"
_C.LOCAL_FILES.DIRNAME_RESULT = f'./gpt4v/result_{which}'

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
