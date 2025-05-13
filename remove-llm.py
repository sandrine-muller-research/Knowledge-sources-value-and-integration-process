# from huggingface_hub import snapshot_download
# import shutil
# import os

# cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
# for folder in os.listdir(cache_dir):
#     if "nlpie--Llama2-MedTuned-13b" in folder:
#         shutil.rmtree(os.path.join(cache_dir, folder))

import shutil
import os

cache_dir = os.path.expanduser("~/.cache/huggingface")
shutil.rmtree(cache_dir)