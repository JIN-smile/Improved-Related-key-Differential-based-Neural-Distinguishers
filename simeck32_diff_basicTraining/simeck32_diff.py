import numpy as np
from os import urandom


def WORD_SIZE():
    return(16)

MASK_VAL = 2 ** WORD_SIZE() - 1



const_simeck = [0xfffd, 0xfffd, 0xfffd, 0xfffd,
                0xfffd, 0xfffc, 0xfffc, 0xfffc,
                0xfffd, 0xfffd, 0xfffc, 0xfffd,
                0xfffd, 0xfffd, 0xfffc, 0xfffd,
                0xfffc, 0xfffd, 0xfffc, 0xfffc,
                0xfffc, 0xfffc, 0xfffd, 0xfffc,
                0xfffc, 0xfffd, 0xfffc, 0xfffd,
                0xfffd, 0xfffc, 0xfffc, 0xfffd]


def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k))) 


def enc_one_round_simeck(p, k):
    c1 = p[0] 
    c0 = (rol(p[0], 5) & rol(p[0],0)) ^ rol(p[0],1) ^ p[1] ^ k
    return(c0,c1)


def expand_key_simeck(k, t):
    ks = [0 for i in range(t)]
    ks_tmp = [0,0,0,0]
    ks_tmp[0] = k[3]
    ks_tmp[1] = k[2]
    ks_tmp[2] = k[1]
    ks_tmp[3] = k[0]
    ks[0] = ks_tmp[0]
    for i in range(1, t):
        ks[i] = ks_tmp[1]
        tmp = (rol(ks_tmp[1], 5) & rol(ks_tmp[1], 0)) ^ rol(ks_tmp[1], 1) ^ ks[i-1] ^ const_simeck[i-1]
        ks_tmp[1] = ks_tmp[2]
        ks_tmp[2] = ks_tmp[3]
        ks_tmp[3] = tmp
    return(ks)


def encrypt_simeck(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round_simeck((x,y), k)
    return(x, y)


def convert_to_binary(arr,s_groups=1):
  X = np.zeros((8 * WORD_SIZE() * s_groups,len(arr[0])),dtype=np.uint8) 
  for i in range(8 * WORD_SIZE() * s_groups):
    index = i // WORD_SIZE() 
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
    X[i] = (arr[index] >> offset) & 1
  X = X.transpose() 
  return(X)


def make_train_data(n, nr, diff=(0x0,0x40),s_groups=1):
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    num_rand_samples = np.sum(Y == 0)
    ks = expand_key_simeck(keys, nr)
    X_result = []
    
    
    
    for i in range(s_groups):
        plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        plain1l = plain0l ^ diff[0]
        plain1r = plain0r ^ diff[1]
        plain1l[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
        plain1r[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
        ctdata0l, ctdata0r = encrypt_simeck((plain0l, plain0r), ks)
        ctdata1l, ctdata1r = encrypt_simeck((plain1l, plain1r), ks)
        
        delta_ctdata0l = ctdata0l ^ ctdata1l
        delta_ctdata0r = ctdata0r ^ ctdata1r
        
        delta_ctdata0rr = ctdata0l ^ ctdata1l ^ ctdata0r ^ ctdata1r
        
        delta_ctdata0 = ctdata0l ^ ctdata0r
        delta_ctdata1 = ctdata1l ^ ctdata1r

        secondLast_ctdata0r = rol(ctdata0r, 5) & rol(ctdata0r, 0) ^ rol(ctdata0r, 1) ^ ctdata0l
        secondLast_ctdata1r = rol(ctdata1r, 5) & rol(ctdata1r, 0) ^ rol(ctdata1r, 1) ^ ctdata1l
        
 
        delta_secondLast_ctdata0r =  secondLast_ctdata0r ^ secondLast_ctdata1r
        
        
        thirdLast_ctdata0r = ctdata0r ^ rol(secondLast_ctdata0r,5) & rol(secondLast_ctdata0r,0) ^ rol(secondLast_ctdata0r,1)
        thirdLast_ctdata1r = ctdata1r ^ rol(secondLast_ctdata1r,5) & rol(secondLast_ctdata1r,0) ^ rol(secondLast_ctdata1r,1)
        
        
        delta_thirdLast_ctdata0r = thirdLast_ctdata0r ^ thirdLast_ctdata1r


        X_result.append(delta_ctdata0l)
        X_result.append(delta_ctdata0r)

        X_result.append(ctdata0l)
        X_result.append(ctdata0r)
        
        X_result.append(ctdata1l)
        X_result.append(ctdata1r)

        #X_result.append(secondLast_ctdata0r)
        #X_result.append(secondLast_ctdata1r)
        
        X_result.append(delta_secondLast_ctdata0r)
        X_result.append(delta_thirdLast_ctdata0r)
    
    X= convert_to_binary(X_result,s_groups=s_groups)
    #X = np.tile(X,s_groups)
    return (X, Y)


#X, Y = make_train_data(10,10,s_groups=1)


def check_testvector():
    #key = (0x1918, 0x1110, 0x0908, 0x0100)
    key = (0xd7c5, 0x5181, 0x04f4, 0x4056)
    #pt = (0x6565, 0x6877)
    #pt = (0xd1d1, 0x03bd)
    pt = (0xd1d1, 0x03fd)


    ks = expand_key_simeck(key,11)
    print("ks:",ks)
    ct = encrypt_simeck(pt, ks)
    print("ct:",ct)
    #p  = decrypt(ct, ks)
    if ((ct == (0x770d, 0x2c76))):
        print("Testvector verified.")
        return(1)
    else:
        print("Testvector not verified.")
        return(0)


check_testvector()