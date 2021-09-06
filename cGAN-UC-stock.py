import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Input, GaussianNoise, multiply, Add, Reshape, GlobalAveragePooling2D, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import hinge
import keras.backend as K
import tensorflow as tf

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

x_train = np.loadtxt('X_train.csv',delimiter=',')
x_train = (x_train - 1.)*20.

x_test = np.loadtxt('X_test.csv',delimiter=',')
x_test = (x_test - 1.)*20.

y_train = np.loadtxt('Y_train.csv',delimiter=',')
y_train = (y_train - 1.)*20.

Input_data = Input(shape=(30, ))
output_F = Dense(128, activation='relu')(Input_data)
output_F = Dropout(0.2)(output_F)
output_F = BatchNormalization()(output_F)
output_F = Dense(128, activation='relu')(output_F)
output_F = Dropout(0.2)(output_F)
output_F = BatchNormalization()(output_F)
output_F = Dense(128, activation='relu')(output_F)
output_F = Dropout(0.2)(output_F)
output_F = BatchNormalization()(output_F)
output_F = Dense(128, activation=None)(output_F)
output_F_pred = Dense(5, activation=None)(output_F)
opt_F = Adam(0.0001)
feat_pred = Model(Input_data, output_F_pred)
feat_pred.compile(optimizer=opt_F, loss='mse')
feat = Model(Input_data, output_F)
feat.compile(optimizer=opt_F, loss='mse')

Input_cond = Input(shape=(128, ))
Input_Y = Input(shape=(5, ))
output_D = Dense(128, activation='relu', input_dim=5)(Input_Y)
cond_encode = Dense(4, activation=None)(Input_cond)
output_D = Dense(128, activation='relu')(output_D)
output_D = Dense(128, activation='relu')(output_D)
output_D = Dense(128, activation='relu')(output_D)
output_D = Dense(128, activation='relu')(output_D)
output_D = Dense(128, activation='relu')(output_D)
output_D = Dense(128, activation='relu')(output_D)
output_D = Dense(4, activation=None)(output_D)
output_D_1 = multiply([output_D, cond_encode])
output_D_1 = Dense(1, activation=None)(output_D_1)
output_D_2 = Dense(1, activation=None)(output_D)
output_D_3 = Add()([output_D_1, output_D_2])
opt_disc = Adam(0.0002*4, beta_1=0., beta_2=0.9)
disc = Model([Input_Y, Input_cond], output_D_3)
disc.compile(optimizer=opt_disc, loss=hinge)

Input_z = Input(shape=(4, ))
cond_encode_G = Dense(4, activation=None, input_dim=128)(Input_cond)
output_G = Dense(4, activation=None, input_dim=4)(Input_z)
output_G = multiply([output_G, cond_encode_G])
output_G = Dense(128, activation='relu')(output_G)
output_G = Dense(128, activation='relu')(output_G)
output_G = Dense(128, activation='relu')(output_G)
output_G = Dense(128, activation='relu')(output_G)
output_G = Dense(128, activation='relu')(output_G)
output_G = Dense(128, activation='relu')(output_G)
output_G = Dense(5, activation=None)(output_G)
gen = Model([Input_z, Input_cond], output_G)

res = gen([Input_z, Input_cond])

disc.trainable = False

valid = disc([res, Input_cond])

opt_comb = Adam(0.0002, beta_1=0., beta_2=0.9)
comb = Model([Input_z, Input_cond], valid)
comb.compile(loss=wasserstein_loss, optimizer=opt_comb)

bs = 512
epochs = int(x_train.shape[0]/float(bs)*2000)
for epoch in range(int(x_train.shape[0]/float(bs)*5000)):
    idx = np.random.randint(0, x_train.shape[0], bs)
    f_loss = feat_pred.train_on_batch(x_train[idx,:],y_train[idx,:])
    if epoch % 1000 == 0:
        print('epoch: ' + str(epoch) + '    f loss: ' + str(f_loss))

for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], bs)
    z = np.random.normal(0, 1, (bs, 4))
    fs = feat.predict(x_train[idx,:])
    fake_samples = gen.predict([z, fs])
    disc_loss = disc.train_on_batch([np.concatenate((y_train[idx,:], fake_samples),0), np.concatenate((fs,fs),0)], np.concatenate((np.ones((bs,1)), -1.0*np.ones((bs,1))),0))
    z = np.random.normal(0, 1, (bs*2, 4))
    gen_loss = comb.train_on_batch([z, np.concatenate((fs,fs),0)], -1.0*np.ones((bs*2,1)))
    if epoch % 1000 == 0:
        print('epoch: ' + str(epoch) + '    disc loss: ' + str(disc_loss) + '    gen loss: ' + str(gen_loss))

for i in range(100):
    z = np.random.normal(0, 1, (x_test.shape[0], 4))
    fs = feat.predict(x_test)
    RES = gen.predict([z, fs])
    np.savetxt("RES_Y3_"+str(i)+".csv", RES/20. + 1., delimiter=",")
