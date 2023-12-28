import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
for i in range(5):
    os.system('python stage_train.py')