import os
import shutil
import random
    
train_dir = "/home/ubuntu/data/face/train_20"
dst_root = "/home/ubuntu/data/face/train_20_1/"
    
if os.path.exists(dst_root) == False:
  os.mkdir(dst_root)
      
for _cls in os.listdir(train_dir):
  dst_dir = os.path.join(dst_root, _cls)
  _cls = os.path.join(train_dir, _cls)
  if os.path.exists(dst_dir) == False:
    os.mkdir(dst_dir)
  
  if ".ipynb" in _cls:
    continue
  files = os.listdir(_cls)
  files = files[:100]

  for file in files:
    if ".ipynb" in file:
      continue
    src_file = os.path.join(_cls, file)
    dst_file = os.path.join(dst_dir, file)
    shutil.copy(src_file, dst_file)