import simon32_rkdiff as sp
import numpy as np
import  math
from pickle import dump
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Input, Reshape, Add, Flatten, BatchNormalization, Activation, Permute,GlobalAveragePooling1D,multiply
from tensorflow.keras.regularizers import l2
from tensorflow.nn import dropout
import tensorflow as tf
#from non_local import non_local_block
#bs = 1000
wdir = './freshly_trained_nets/'



if(not os.path.exists(wdir)):
  os.makedirs(wdir)
            
def scheduler14(epoch):
  return 0.001 * math.exp(0.04 * (0 - epoch))

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);


def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True)
  return(res)


def make_resnet(num_blocks=2, num_filters_1=32, num_filters_2=32, num_filters_3=32, num_outputs=1, d0=512,d1=64, d2=64, word_size=16, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid',s_groups = 1,ks_value_1=1,ks_value_2=3,ks_value_3=3,dropout_value=0.8,se_ratio=16):
  inp = Input(shape=(int(s_groups * num_blocks*word_size * 4),))
  rs = Reshape((s_groups, int(num_blocks*word_size * 4)))(inp)
  #rs = Reshape((1, 80))(inp)
  #perm = Permute((2,1))(rs);

  conv0 = Conv1D(num_filters_1, kernel_size=ks_value_1, padding='same', kernel_regularizer=l2(reg_param))(rs)
  conv0 = BatchNormalization()(conv0)
  conv0 = Activation('relu')(conv0)
  

  dense0 = Dense(num_filters_1, kernel_regularizer=l2(reg_param))(conv0);
  dense0 = BatchNormalization()(dense0);
  conv0 = Activation('relu')(dense0);
  dense0 = Dense(num_filters_1, kernel_regularizer=l2(reg_param))(conv0);
  dense0 = BatchNormalization()(dense0);
  conv0 = Activation('relu')(dense0);
  
  '''
  #nonlocal
  conv_non = non_local_block(conv0, mode='embedded', compression=2)
  conv0 = Add()([conv0,conv_non])
  
  shortcut = conv0;

  se_ratio = 2
  se_input = shortcut
  conv2 = GlobalAveragePooling1D()(se_input)
  conv2 = Reshape((1, num_filters_1))(conv2)
  conv2 = Dense(num_filters_1 // se_ratio, activation='relu', use_bias=False)(conv2)
  conv2 = Dense(num_filters_1, activation='sigmoid', use_bias=False)(conv2)
  shortcut = multiply([conv2, se_input])

  shortcut = Add()([conv0,shortcut])
  '''
  
  shortcut = conv0
  
  for i in range(depth):
    conv1 = Conv1D(num_filters_2, kernel_size=ks_value_2, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv1D(num_filters_3, kernel_size=ks_value_3, padding='same',kernel_regularizer=l2(reg_param))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    
    # SENet_module
    #se_ratio = 16
    se_input = conv2
    conv2 = GlobalAveragePooling1D()(se_input)
    conv2 = Reshape((1, num_filters_1))(conv2)
    conv2 = Dense(num_filters_1 // se_ratio, activation='relu', use_bias=False)(conv2)
    conv2 = Dense(num_filters_1, activation='sigmoid', use_bias=False)(conv2)
    conv2 = multiply([conv2, se_input])
    
    
    
    shortcut = Add()([shortcut, conv2])
  flat1 = Flatten()(shortcut)
  
  dense0 = dropout(flat1, dropout_value)

  '''
  dense0 = Dense(d0, kernel_regularizer=l2(reg_param))(dense0)
  dense0 = BatchNormalization()(dense0)
  dense0 = Activation('relu')(dense0)
  '''

  dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1)
  dense1 = BatchNormalization()(dense1)
  dense1 = Activation('relu')(dense1)
  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
  dense2 = BatchNormalization()(dense2)
  dense2 = Activation('relu')(dense2)
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
  model = Model(inputs=inp, outputs=out)
  model.summary()
  return(model)


def train_distinguisher(num_epochs, num_rounds=7, depth=1,filename="default",s_groups=1,bs=5000,lr_choice="14",d0=512,d1=64,d2=64,ks_value_1=1,ks_value_2=3,ks_value_3=3,machine_name='vgc',dropout_value=0.8,num_filters_1=32,num_filters_2=32,num_filters_3=32,reg_param=10**-4,high_lr=0.002,low_lr=0.0001,lr_epoch=10,se_ratio=16):
    net = make_resnet(depth=depth, reg_param=reg_param,s_groups=s_groups,d0=d0,d1=d1,d2=d2,ks_value_1=ks_value_1,ks_value_2=ks_value_2,ks_value_3=ks_value_3,dropout_value=dropout_value,num_filters_1=num_filters_1,num_filters_2=num_filters_2,num_filters_3=num_filters_3,se_ratio=se_ratio)
    #net.compile(optimizer='Adam',loss='mse',metrics=['acc'])
    net.compile(optimizer='Adam',loss='mse',metrics=['acc',tf.keras.metrics.TruePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives()])
    trainableParams = np.sum([np.prod(v.get_shape()) for v in net.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in net.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams

    
    '''
    tf.keras.metrics.TruePositives(),
    tf.keras.metrics.TrueNegatives(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.FalseNegatives()])
    '''
    X, Y = sp.make_train_data(10**7*2,num_rounds,s_groups=s_groups)
    X_eval, Y_eval = sp.make_train_data(10**6*2, num_rounds,s_groups=s_groups)
    

    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5')
    if(lr_choice=="14"):
      lr = LearningRateScheduler(scheduler14)
    elif(lr_choice=='gohr'):
      lr = LearningRateScheduler(cyclic_lr(lr_epoch,high_lr, low_lr))
      #lr = LearningRateScheduler(schedule=)
    h = net.fit(X,Y,epochs=num_epochs,batch_size=bs, verbose=1, validation_data=(X_eval, Y_eval), callbacks=[lr,check])
    
    import datetime
    curr_time = datetime.datetime.now()
    filename = "ep{0}_round{1}_depth{2}_group{3}_bs{4}_lr{5}_d1{6}_d2{7}_ks{8}_dp{9}_filter{10}_reg{11}_lr{12}_{13}_lrepoch{14}_{15}".format(str(num_epochs),str(num_rounds),str(depth),str(s_groups),str(bs),str(lr_choice),str(d1),str(d2),str(ks_value_1)+str(ks_value_2)+str(ks_value_3),str(dropout_value)[2:],str(num_filters_1)+"-"+str(num_filters_2)+"-"+str(num_filters_3),str(reg_param),str(high_lr)+"-"+str(low_lr),machine_name,str(lr_epoch),str(curr_time.year)+'-'+str(curr_time.month)+'-'+str(curr_time.day)+'-'+str(curr_time.hour)+'-'+str(curr_time.minute)+'-'+str(curr_time.second))
    import os
    import pandas as pd

    repeat_filename = "ep{0}_round{1}_depth{2}_group{3}_bs{4}_lr{5}_d1{6}_d2{7}_ks{8}_dp{9}_filter{10}_reg{11}_lr{12}_{13}_lrepoch{14}".format(str(num_epochs),str(num_rounds),str(depth),str(s_groups),str(bs),str(lr_choice),str(d1),str(d2),str(ks_value_1)+str(ks_value_2)+str(ks_value_3),str(dropout_value)[2:],str(num_filters_1)+"-"+str(num_filters_2)+"-"+str(num_filters_3),str(reg_param),str(high_lr)+"-"+str(low_lr),machine_name,str(lr_epoch))
    repeat_dir = './repeat_data/' + repeat_filename + '/'
    if(not os.path.exists(repeat_dir)):
      os.makedirs(repeat_dir)



    excel_name = repeat_dir+filename+"_record.xlsx"
    if(not os.path.exists(excel_name)):
        dfData = {
        'epoch':[],
        'acc': [],
        'val_acc':[],
        'loss':[],
        'val_loss':[]
        }
        df = pd.DataFrame(dfData)
        df.to_excel(excel_name, index=False)
    for epoch_cnt in range(0,len(h.history['acc'])):
      
      old_df = pd.read_excel(excel_name)
      new_df = {
          'epoch':[epoch_cnt],
          'acc': [h.history['acc'][epoch_cnt]],
          'val_acc':[h.history['val_acc'][epoch_cnt]],
          'loss':[h.history['loss'][epoch_cnt]],
          'val_loss':[h.history['val_loss'][epoch_cnt]]
      }
      new_df = pd.DataFrame(new_df)
      df = pd.concat([old_df, new_df])
      df.to_excel(excel_name, index=False)
    
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig_name = repeat_dir+filename+"_train_history_acc.jpg"
    plt.savefig(fig_name)
    plt.cla()
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig_name = repeat_dir+filename + "_train_history_loss.jpg"
    plt.savefig(fig_name)
    #np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_acc'])
    #np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_loss'])
    #dump(h.history,open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'))
    
    
    data_index = np.argmax(h.history['val_acc'])
    
    vaL_tp = h.history['val_true_positives'][data_index]
    vaL_tn = h.history['val_true_negatives'][data_index]
    vaL_fp = h.history['val_false_positives'][data_index]
    vaL_fn = h.history['val_false_negatives'][data_index]
    print("true positive rate: ", vaL_tp/(vaL_tp+vaL_fn))
    print("true negative rate: ", vaL_tn/(vaL_tn+vaL_fp))
    
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    
    
    repeat_excel_name = repeat_dir+"record.xlsx"
    if(not os.path.exists(repeat_excel_name)):
        dfData = {
        'epoch':[],
        'round': [],
        'depth':[],
        'group':[],
        'bs':[],
        'lr':[],
        'd1': [],
        'd2':[],
        'ks':[],
        'acc':[],
        'dropout':[],
        'num_filters':[],
        'machine':[],
        'reg_param':[],
        'high_lr':[],
        'low_lr':[],
        'TPR':[],
        'TNR':[],
        'totalParams':[],
        'train_acc':[],
        'd0':[],
        'curr_time':[],
        'lr_epoch':[],
        'max_acc_epoch':[],
        'se_ratio':[]
        }
        df = pd.DataFrame(dfData)
        df.to_excel(repeat_excel_name, index=False)
    
    old_df = pd.read_excel(repeat_excel_name)
    new_df = {
        'epoch':[num_epochs],
        'round': [num_rounds],
        'depth':[depth],
        'group':[s_groups],
        'bs':[bs],
        'lr':[lr_choice],
        'd1': [d1],
        'd2':[d2],
        'ks':[str(ks_value_1)+str(ks_value_2)+str(ks_value_3)],
        'acc':[np.max(h.history['val_acc'])],
        'dropout':[dropout_value],
        'num_filters':[str(num_filters_1)+"-"+str(num_filters_2)+"-"+str(num_filters_3)],
        'machine':[machine_name],
        'reg_param':[reg_param],
        'high_lr':[high_lr],
        'low_lr':[low_lr],
        'totalParams':[totalParams],
        'train_acc':[h.history['acc'][data_index]],
        'd0':[d0],
        'curr_time':[str(curr_time.year)+'-'+str(curr_time.month)+'-'+str(curr_time.day)+'-'+str(curr_time.hour)+'-'+str(curr_time.minute)+'-'+str(curr_time.second)],
        'TPR':[vaL_tp/(vaL_tp+vaL_fn)],
        'TNR':[vaL_tn/(vaL_tn+vaL_fp)],
        'lr_epoch':[lr_epoch],
        'max_acc_epoch':[data_index+1],
        'se_ratio':[se_ratio]
    }
    new_df = pd.DataFrame(new_df)
    df = pd.concat([old_df, new_df])
    df.to_excel(repeat_excel_name, index=False)
    
    excel_name = "all_record.xlsx"
    if(not os.path.exists(excel_name)):
        dfData = {
        'epoch':[],
        'round': [],
        'depth':[],
        'group':[],
        'bs':[],
        'lr':[],
        'd1': [],
        'd2':[],
        'ks':[],
        'acc':[],
        'dropout':[],
        'num_filters':[],
        'machine':[],
        'reg_param':[],
        'high_lr':[],
        'low_lr':[],
        'TPR':[],
        'TNR':[],
        'totalParams':[],
        'train_acc':[],
        'd0':[],
        'curr_time':[],
        'lr_epoch':[],
        'max_acc_epoch':[],
        'se_ratio':[]
        }
        df = pd.DataFrame(dfData)
        df.to_excel(excel_name, index=False)
    
    old_df = pd.read_excel(excel_name)
    new_df = {
        'epoch':[num_epochs],
        'round': [num_rounds],
        'depth':[depth],
        'group':[s_groups],
        'bs':[bs],
        'lr':[lr_choice],
        'd1': [d1],
        'd2':[d2],
        'ks':[str(ks_value_1)+str(ks_value_2)+str(ks_value_3)],
        'acc':[np.max(h.history['val_acc'])],
        'dropout':[dropout_value],
        'num_filters':[str(num_filters_1)+"-"+str(num_filters_2)+"-"+str(num_filters_3)],
        'machine':[machine_name],
        'reg_param':[reg_param],
        'high_lr':[high_lr],
        'low_lr':[low_lr],
        'totalParams':[totalParams],
        'train_acc':[h.history['acc'][data_index]],
        'd0':[d0],
        'curr_time':[str(curr_time.year)+'-'+str(curr_time.month)+'-'+str(curr_time.day)+'-'+str(curr_time.hour)+'-'+str(curr_time.minute)+'-'+str(curr_time.second)],
        'TPR':[vaL_tp/(vaL_tp+vaL_fn)],
        'TNR':[vaL_tn/(vaL_tn+vaL_fp)],
        'lr_epoch':[lr_epoch],
        'max_acc_epoch':[data_index+1],
        'se_ratio':[se_ratio]
    }
    new_df = pd.DataFrame(new_df)
    df = pd.concat([old_df, new_df])
    df.to_excel(excel_name, index=False)
    
    
    return(net, h)


