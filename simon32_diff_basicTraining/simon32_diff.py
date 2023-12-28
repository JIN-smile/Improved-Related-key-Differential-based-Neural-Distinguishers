import numpy as np
from os import urandom


def WORD_SIZE():
    return(16)

MASK_VAL = 2 ** WORD_SIZE() - 1

const = [0xfffd, 0xfffd, 0xfffd, 0xfffd,
         0xfffd, 0xfffc, 0xfffd, 0xfffc,
         0xfffc, 0xfffc, 0xfffd, 0xfffc,
         0xfffc, 0xfffd, 0xfffc, 0xfffd,
         0xfffc, 0xfffd, 0xfffd, 0xfffc,
         0xfffc, 0xfffc, 0xfffc, 0xfffd,
         0xfffd, 0xfffd, 0xfffc, 0xfffc]


def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k))) 


def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL)) 


def enc_one_round_simon(p, k):
    c1 = p[0] 
    c0 = (rol(p[0], 8) & rol(p[0],1)) ^ rol(p[0],2) ^ p[1] ^ k
    return(c0,c1)


def expand_key_simon(k, t):
    ks = [0 for i in range(t)]
    ks[0] = k[3]
    ks[1] = k[2]
    ks[2] = k[1]
    ks[3] = k[0]
    for i in range(t - 4):
        tmp = ror(ks[i+3],3) ^ ks[i+1]
        ks[i+4] = const[i] ^ ks[i] ^ tmp ^ ror(tmp,1)
    return(ks)


def encrypt_simon(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round_simon((x,y), k)
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
    ks = expand_key_simon(keys, nr)
    X_result = []
    
    
    for i in range(s_groups):
        plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        plain1l = plain0l ^ diff[0]
        plain1r = plain0r ^ diff[1]
        plain1l[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
        plain1r[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
        ctdata0l, ctdata0r = encrypt_simon((plain0l, plain0r), ks)
        ctdata1l, ctdata1r = encrypt_simon((plain1l, plain1r), ks)
        
        delta_ctdata0l = ctdata0l ^ ctdata1l
        delta_ctdata0r = ctdata0r ^ ctdata1r
        
        delta_ctdata0rr = ctdata0l ^ ctdata1l ^ ctdata0r ^ ctdata1r
        
        delta_ctdata0 = ctdata0l ^ ctdata0r
        delta_ctdata1 = ctdata1l ^ ctdata1r

        secondLast_ctdata0r = rol(ctdata0r, 8) & rol(ctdata0r, 1) ^ rol(ctdata0r, 2) ^ ctdata0l
        secondLast_ctdata1r = rol(ctdata1r, 8) & rol(ctdata1r, 1) ^ rol(ctdata1r, 2) ^ ctdata1l
        
        
 
        secondLast_ctdata0r = rol(ctdata0r, 8) & rol(ctdata0r, 1) ^ rol(ctdata0r, 2) ^ ctdata0l
        secondLast_ctdata1r = rol(ctdata1r, 8) & rol(ctdata1r, 1) ^ rol(ctdata1r, 2) ^ ctdata1l
 
        delta_secondLast_ctdata0r =  secondLast_ctdata0r ^ secondLast_ctdata1r
        
        
        thirdLast_ctdata0r = ctdata0r ^ rol(secondLast_ctdata0r,8) & rol(secondLast_ctdata0r,1) ^ rol(secondLast_ctdata0r,2)
        thirdLast_ctdata1r = ctdata1r ^ rol(secondLast_ctdata1r,8) & rol(secondLast_ctdata1r,1) ^ rol(secondLast_ctdata1r,2)
        
        
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


