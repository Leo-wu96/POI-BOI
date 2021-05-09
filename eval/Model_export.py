import numpy as np
import scipy.io as io
import os

filename = 'Acc.npy'

fusion_age = './Model/fusion_age'
creative_age = './Model/creative_age'
ad_age = './Model/ad_age'
adver_age = './Model/adver_age'


fusion_gender = './Model/fusion_gender'
creative_gender = './Model/creative_gender'
ad_gender = './Model/ad_gender'
adver_gender = './Model/adver_gender'

fa = np.load(os.path.join(fusion_age,filename))
ca = np.load(os.path.join(creative_age,filename))
ada = np.load(os.path.join(ad_age,filename))
advera = np.load(os.path.join(adver_age,filename))




fg = np.load(os.path.join(fusion_gender,filename))
cg = np.load(os.path.join(creative_gender,filename))
adg = np.load(os.path.join(ad_gender,filename))
adverg = np.load(os.path.join(adver_gender,filename))

io.savemat('model_export_{}.mat'.format(filename),{'fa':fa,'ca':ca,'ada':ada,'advera':advera,'fg':fg,'cg':cg,'adg':adg,'adverg':adverg})
