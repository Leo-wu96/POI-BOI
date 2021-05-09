import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import json
import scipy.io as io
from collections import OrderedDict



#####################Age###########################################################
creative = np.load('log_creativeid_feature.npy')
ad = np.load('log_adid_feature.npy')
adver = np.load('log_aderid_feature.npy')
feature3 = np.load('log_3feature.npy')

ad = 1-np.asarray(eval(str(ad))['valid_1']['multi_error'])
creative = 1-np.asarray(eval(str(creative))['valid_1']['multi_error'])
adver = 1-np.asarray(eval(str(adver))['valid_1']['multi_error'])
feature3 = 1-np.asarray(eval(str(feature3))['valid_1']['multi_error'])

io.savemat("age_doc2vec_result.mat", {"ad": ad, "creative": creative, "adver": adver, "feature3": feature3})



#####################Gender###########################################################
creative = np.load('log_creativeid_feature_g.npy')
ad = np.load('log_adid_feature_g.npy')
adver = np.load('log_aderid_feature_g.npy')
feature3 = np.load('log_3feature_g.npy')

ad = 1-np.asarray(eval(str(ad))['valid_1']['multi_error'])
creative = 1-np.asarray(eval(str(creative))['valid_1']['multi_error'])
adver = 1-np.asarray(eval(str(adver))['valid_1']['multi_error'])
feature3 = 1-np.asarray(eval(str(feature3))['valid_1']['multi_error'])

io.savemat("gender_doc2vec_result.mat", {"ad": ad, "creative": creative, "adver": adver, "feature3": feature3})