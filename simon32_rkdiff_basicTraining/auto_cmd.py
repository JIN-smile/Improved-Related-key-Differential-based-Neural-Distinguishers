import os
for i in range(5):
  for gpu_num in [0]:
      for round_value in [10,11,12,13]:
        os.system('python train_rounds.py --GPU_device={0} --batch_size=30000 --s_groups=8 --epochs=120 --num_rounds={1} --depth_value=5 --file_name=test --lr_choice=gohr --d0=64 --d1=128 --d2=128 --ks_value_1=1 --ks_value_2=3 --ks_value_3=3 --dropout_value=0.5 --num_filters_1=64 --num_filters_2=64 --num_filters_3=64 --machine_name=vgc --reg_param=0.00001 --high_lr=0.003 --low_lr=0.0001 --lr_epoch=30 --se_ratio=16'.format(gpu_num,round_value))
