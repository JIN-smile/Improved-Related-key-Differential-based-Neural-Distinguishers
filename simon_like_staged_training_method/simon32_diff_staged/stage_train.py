import simon32_diff as sp
import os
from tensorflow.keras.models import load_model,model_from_json

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import numpy as np
wdir = './freshly_trained_nets/'
if(not os.path.exists(wdir)):
  os.makedirs(wdir)

def cyclic_lr(num_epochs, high_lr=0.002, low_lr=0.0001):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_acc', save_best_only=True)
    return(res)

def first_stage(num_rounds=9,s_groups=8,high_lr=0.005, low_lr=0.0001,bs=5000,data_num=28,epochs=10,adam_lr=0.0001,data_diff=(0x0440,0x0100),lr_epoch=10,curr_time="default"):

    X, Y = sp.make_train_data(int(2**data_num),num_rounds-1,s_groups=s_groups,diff=data_diff)
    X_eval, Y_eval = sp.make_train_data(int(2**(data_num-2)), num_rounds-1,s_groups=s_groups,diff=data_diff)
    print("\n")
    print("train: {} test: {}".format(str(int(2**data_num)),str(int(2**(data_num-2)))))
    print("\n")
    
    net = load_model(wdir+"best_simon_model_r{}_group{}.h5".format(num_rounds,s_groups))
    net_json = net.to_json()

    net_first = model_from_json(net_json)
    #net_first.compile(optimizer='Adam',loss='mse',metrics=['acc'])
    net_first.compile(optimizer=Adam(learning_rate = adam_lr), loss='mse', metrics=['acc',tf.keras.metrics.TruePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives()])
    # net_first.compile(optimizer='adam', loss='mse', metrics=['acc'])
    net_first.load_weights(wdir+"best_simon_model_r{}_group{}.h5".format(num_rounds,s_groups))
    
    check = make_checkpoint(wdir+'best_stage_1_{}.h5'.format(curr_time))
    
    lr = LearningRateScheduler(cyclic_lr(num_epochs=lr_epoch,high_lr=high_lr, low_lr=low_lr))
    h = net_first.fit(X,Y,epochs=epochs,batch_size=bs, verbose=1, validation_data=(X_eval, Y_eval), callbacks=[lr,check])
    
    
    
    del X
    del Y
    del X_eval
    del Y_eval
    
    import matplotlib.pyplot as plt
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig_name = "first_stage_train_history_acc_{}.jpg".format(curr_time)
    plt.savefig(fig_name)
    plt.cla()
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig_name = "first_stage_train_history_loss_{}.jpg".format(curr_time)
    plt.savefig(fig_name)

    net_first.save(wdir+"net_first_{}.h5".format(curr_time))
    import pandas as pd
    excel_name = "first_stage_single_record_{}.xlsx".format(curr_time)
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
    
    return h
    
def second_stage(num_rounds=11,s_groups=8,high_lr=0.005, low_lr=0.0001,bs=5000,data_num=30,epochs=1,adam_lr=0.0001,data_diff=(0x0000,0x0040),lr_epoch=10,curr_time="default"):

    X, Y = sp.make_train_data(int(2**data_num ),num_rounds,s_groups=s_groups,diff=data_diff)
    X_eval, Y_eval = sp.make_train_data(int(2**(data_num-2)), num_rounds,s_groups=s_groups,diff=data_diff)
    print("\n")
    print("train: {} test: {}".format(str(int(2**data_num )),str(int(2**(data_num-2)))))
    print("\n")
    
    #net = load_model(wdir+'net_first_{}.h5'.format(curr_time))
    net = load_model(wdir+'best_stage_1_{}.h5'.format(curr_time))
    
    #net = load_model(wdir+'best_stage_1_2022-10-21-22-56-37.h5')
    
    net_json = net.to_json()

    net_second = model_from_json(net_json)
    
    net_second.compile(optimizer=Adam(learning_rate = adam_lr), loss='mse', metrics=['acc',tf.keras.metrics.TruePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives()])
    
    net_second.load_weights(wdir+'best_stage_1_{}.h5'.format(curr_time))
    #net_second.load_weights(wdir+'net_first_{}.h5'.format(curr_time))
    #net_second.load_weights(wdir+'best_stage_1_2022-10-21-22-56-37.h5')
    
    check = make_checkpoint(wdir+'best_stage_2_{}.h5'.format(curr_time))
    
    lr = LearningRateScheduler(cyclic_lr(num_epochs=lr_epoch,high_lr=high_lr, low_lr=low_lr))
    h = net_second.fit(X,Y,epochs=epochs,batch_size=bs, verbose=1, validation_data=(X_eval, Y_eval), callbacks=[lr,check])
    
    del X
    del Y
    del X_eval
    del Y_eval
    
    import matplotlib.pyplot as plt
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig_name = "second_stage_train_history_acc_{}.jpg".format(curr_time)
    plt.savefig(fig_name)
    plt.cla()
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig_name = "second_stage_train_history_loss_{}.jpg".format(curr_time)
    plt.savefig(fig_name)

    net_second.save(wdir+"net_second_{}.h5".format(curr_time))
    import pandas as pd
    excel_name = "second_stage_single_record_{}.xlsx".format(curr_time)
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
    
    return h
    
def last_stage(num_rounds=11,s_groups=8,high_lr=0.005, low_lr=0.0001,bs=5000,data_num=30,epochs=2,adam_lr=0.00001,data_diff=(0x0000,0x0040),lr_epoch=10,curr_time="default"):

    X, Y = sp.make_train_data(int(2**data_num),num_rounds,s_groups=s_groups,diff=data_diff)
    X_eval, Y_eval = sp.make_train_data(int(2**(data_num-2)), num_rounds,s_groups=s_groups,diff=data_diff)
    
    print("\n")
    print("train: {} test: {}".format(str(int(2**data_num)),str(int(2**(data_num-2)))))
    print("\n")
    
    net = load_model(wdir+'best_stage_2_{}.h5'.format(curr_time))
    #net = load_model(wdir+'net_second_{}.h5'.format(curr_time))
    #net = load_model(wdir+'best_stage_2_2022-10-17-9-18-3.h5')
    
    net_json = net.to_json()

    net_final = model_from_json(net_json)
    #net_first.compile(optimizer='Adam',loss='mse',metrics=['acc'])
    net_final.compile(optimizer=Adam(learning_rate = adam_lr), loss='mse', metrics=['acc',tf.keras.metrics.TruePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives()])
    # net_first.compile(optimizer='adam', loss='mse', metrics=['acc'])
    
    net_final.load_weights(wdir+'best_stage_2_{}.h5'.format(curr_time))
    #net_final.load_weights(wdir+'net_second_{}.h5'.format(curr_time))
    #net_final.load_weights(wdir+'best_stage_2_2022-10-17-9-18-3.h5')
    
    check = make_checkpoint(wdir+'best_stage_3_{}.h5'.format(curr_time))
    
    lr = LearningRateScheduler(cyclic_lr(num_epochs=lr_epoch,high_lr=high_lr, low_lr=low_lr))
    h = net_final.fit(X,Y,epochs=epochs,batch_size=bs, verbose=1, validation_data=(X_eval, Y_eval), callbacks=[lr,check])
    
    del X
    del Y
    del X_eval
    del Y_eval
    
    import matplotlib.pyplot as plt
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig_name = "last_stage_train_history_acc_{}.jpg".format(curr_time)
    plt.savefig(fig_name)
    plt.cla()
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig_name = "last_stage_train_history_loss_{}.jpg".format(curr_time)
    plt.savefig(fig_name)

    net_final.save(wdir+"net_final_{}.h5".format(curr_time))
    
    import pandas as pd
    excel_name = "last_stage_single_record_{}.xlsx".format(curr_time)
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
    
    
    
    
    
    return h
    
    
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    
    
    # input file location
    
    parser.add_argument("--first_stage_epochs",type=int,default=30)
    parser.add_argument("--second_stage_epochs",type=int,default=30)
    parser.add_argument("--last_stage_epochs",type=int,default=30)
    
    parser.add_argument("--first_stage_num_round",type=int,default=10)
    parser.add_argument("--second_stage_num_round",type=int,default=12)
    parser.add_argument("--last_stage_num_round",type=int,default=12)
    
    parser.add_argument("--first_stage_bs",type=int,default=30000)
    parser.add_argument("--second_stage_bs",type=int,default=30000)
    parser.add_argument("--last_stage_bs",type=int,default=30000)
    
    parser.add_argument("--first_stage_data_num",type=int,default=25)
    parser.add_argument("--second_stage_data_num",type=int,default=25)
    parser.add_argument("--last_stage_data_num",type=int,default=25)
    
    
    parser.add_argument("--first_stage_high_lr",type=float,default=0.001)
    parser.add_argument("--second_stage_high_lr",type=float,default=0.001)
    parser.add_argument("--last_stage_high_lr",type=float,default=0.001)
    
    parser.add_argument("--first_stage_low_lr",type=float,default=0.0001)
    parser.add_argument("--second_stage_low_lr",type=float,default=0.0001)
    parser.add_argument("--last_stage_low_lr",type=float,default=0.0001)
    
    parser.add_argument("--first_stage_adam_lr",type=float,default=0.0001)
    parser.add_argument("--second_stage_adam_lr",type=float,default=0.0001)
    parser.add_argument("--last_stage_adam_lr",type=float,default=0.00001)
    
    parser.add_argument("--first_stage_lr_epoch",type=int,default=30)
    parser.add_argument("--second_stage_lr_epoch",type=int,default=30)
    parser.add_argument("--last_stage_lr_epoch",type=int,default=30)
    
    args = parser.parse_args()
    
    import datetime
    curr_time = datetime.datetime.now()
    time_record = str(curr_time.year)+'-'+str(curr_time.month)+'-'+str(curr_time.day)+'-'+str(curr_time.hour)+'-'+str(curr_time.minute)+'-'+str(curr_time.second)
    import time
    h_first = first_stage(num_rounds=args.first_stage_num_round,s_groups=8,high_lr=args.first_stage_high_lr, low_lr=args.first_stage_low_lr,bs=args.first_stage_bs,data_num=args.first_stage_data_num,epochs=args.first_stage_epochs,adam_lr=args.first_stage_adam_lr,lr_epoch=args.first_stage_lr_epoch,data_diff=(0x0440,0x0100),curr_time=time_record)
    h_second = second_stage(num_rounds=args.second_stage_num_round,s_groups=8,high_lr=args.second_stage_high_lr, low_lr=args.second_stage_low_lr,bs=args.second_stage_bs,data_num=args.second_stage_data_num,epochs=args.second_stage_epochs,adam_lr=args.second_stage_adam_lr,lr_epoch=args.second_stage_lr_epoch,data_diff=(0x0000,0x0040),curr_time=time_record)
   
    #h_last = last_stage(num_rounds=args.last_stage_num_round,s_groups=8,high_lr=args.last_stage_high_lr, low_lr=args.last_stage_low_lr,bs=args.last_stage_bs,data_num=args.last_stage_data_num,epochs=args.last_stage_epochs,adam_lr=args.last_stage_adam_lr,lr_epoch=args.last_stage_lr_epoch,data_diff=(0x0000,0x0040),curr_time=time_record)
    
    import pandas as pd
    import numpy as np
    excel_name = "all_record.xlsx"
    if(not os.path.exists(excel_name)):
        dfData = {
        'first_stage_acc': [],
        'second_stage_acc':[],
        'last_stage_acc':[],
        'first_stage_TPR':[],
        'first_stage_TNR':[],
        'second_stage_TPR':[],
        'second_stage_TNR':[],
        'last_stage_TPR':[],
        'last_stage_TNR':[],
        'first_stage_train_acc': [],
        'second_stage_train_acc':[],
        'last_stage_train_acc':[],
        'first_stage_epochs':[],
        'second_stage_epochs':[],
        'last_stage_epochs':[],
        'first_stage_adam_lr':[],
        'second_stage_adam_lr':[],
        'last_stage_adam_lr':[],
        'first_stage_high_lr':[],
        'second_stage_high_lr':[],
        'last_stage_high_lr':[],
        'first_stage_low_lr':[],
        'second_stage_low_lr':[],
        'last_stage_low_lr':[],
        'first_stage_num_round':[],
        'second_stage_num_round':[],
        'last_stage_num_round':[],
        'first_stage_bs':[],
        'second_stage_bs':[],
        'last_stage_bs':[],
        'first_stage_data_num':[],
        'second_stage_data_num':[],
        'last_stage_data_num':[],
        'first_stage_lr_epoch':[],
        'second_stage_lr_epoch':[],
        'last_stage_lr_epoch':[],
        'current_time':[]
        }
        df = pd.DataFrame(dfData)
        df.to_excel(excel_name, index=False)
    old_df = pd.read_excel(excel_name)
    
    
    data_index = np.argmax(h_first.history['val_acc'])
    
    first_vaL_tp = h_first.history['val_true_positives'][data_index]
    first_vaL_tn = h_first.history['val_true_negatives'][data_index]
    first_vaL_fp = h_first.history['val_false_positives'][data_index]
    first_vaL_fn = h_first.history['val_false_negatives'][data_index]
    
    data_index = np.argmax(h_second.history['val_acc'])
    
    second_vaL_tp = h_second.history['val_true_positives_1'][data_index]
    second_vaL_tn = h_second.history['val_true_negatives_1'][data_index]
    second_vaL_fp = h_second.history['val_false_positives_1'][data_index]
    second_vaL_fn = h_second.history['val_false_negatives_1'][data_index]
    '''
    data_index = np.argmax(h_last.history['val_acc'])
    
    last_vaL_tp = h_last.history['val_true_positives_2'][data_index]
    last_vaL_tn = h_last.history['val_true_negatives_2'][data_index]
    last_vaL_fp = h_last.history['val_false_positives_2'][data_index]
    last_vaL_fn = h_last.history['val_false_negatives_2'][data_index]
    '''
    new_df = {
        'first_stage_acc': [np.max(h_first.history['val_acc'])],
        'second_stage_acc':[np.max(h_second.history['val_acc'])],
        #'last_stage_acc':[np.max(h_last.history['val_acc'])],
        'first_stage_TPR':[first_vaL_tp/(first_vaL_tp+first_vaL_fn)],
        'first_stage_TNR':[first_vaL_tn/(first_vaL_tn+first_vaL_fp)],
        'second_stage_TPR':[second_vaL_tp/(second_vaL_tp+second_vaL_fn)],
        'second_stage_TNR':[second_vaL_tn/(second_vaL_tn+second_vaL_fp)],
        #'last_stage_TPR':[last_vaL_tp/(last_vaL_tp+last_vaL_fn)],
        #'last_stage_TNR':[last_vaL_tn/(last_vaL_tn+last_vaL_fp)],
        'first_stage_train_acc': [h_first.history['acc'][np.argmax(h_first.history['val_acc'])]],
        'second_stage_train_acc':[h_second.history['acc'][np.argmax(h_second.history['val_acc'])]],
        #'last_stage_train_acc':[h_last.history['acc'][np.argmax(h_last.history['val_acc'])]],
        'first_stage_epochs':[args.first_stage_epochs],
        'second_stage_epochs':[args.second_stage_epochs],
        'last_stage_epochs':[args.last_stage_epochs],
        'first_stage_adam_lr':[args.first_stage_adam_lr],
        'second_stage_adam_lr':[args.second_stage_adam_lr],
        'last_stage_adam_lr':[args.last_stage_adam_lr],
        'first_stage_high_lr':[args.first_stage_high_lr],
        'second_stage_high_lr':[args.second_stage_high_lr],
        'last_stage_high_lr':[args.last_stage_high_lr],
        'first_stage_low_lr':[args.first_stage_low_lr],
        'second_stage_low_lr':[args.second_stage_low_lr],
        'last_stage_low_lr':[args.last_stage_low_lr],
        'first_stage_num_round':[args.first_stage_num_round],
        'second_stage_num_round':[args.second_stage_num_round],
        'last_stage_num_round':[args.last_stage_num_round],
        'first_stage_bs':[args.first_stage_bs],
        'second_stage_bs':[args.second_stage_bs],
        'last_stage_bs':[args.last_stage_bs],
        'first_stage_data_num':[args.first_stage_data_num],
        'second_stage_data_num':[args.second_stage_data_num],
        'last_stage_data_num':[args.last_stage_data_num],
        'first_stage_lr_epoch':[args.first_stage_lr_epoch],
        'second_stage_lr_epoch':[args.second_stage_lr_epoch],
        'last_stage_lr_epoch':[args.last_stage_lr_epoch],
        'current_time':[time_record]
    }
    new_df = pd.DataFrame(new_df)
    df = pd.concat([old_df, new_df])
    df.to_excel(excel_name, index=False)
    