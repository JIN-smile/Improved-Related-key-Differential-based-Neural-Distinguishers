import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
for i in range(5):
    os.system('python stage_train.py')