import pickle
import gensim
import pandas as pd
import json
import numpy as np
import time
import os
from multiprocessing.dummy import Pool as ThreadPool


sentence_path = '/media/eesissi/Data/zwl/TencentData/process_data/sequence_text_all_dict_user_id_others_id.pkl'
w2v_file_creative = '/media/eesissi/Data/zwl/TencentData/process_data/w2v_sequence_text_user_id_creative_id.pkl'
w2v_file_ad = '/media/eesissi/Data/zwl/TencentData/process_data/w2v_sequence_text_user_id_ad_id.pkl'
w2v_file_product = '/media/eesissi/Data/zwl/TencentData/process_data/w2v_sequence_text_user_id_product_id.pkl'
w2v_file_click_times = '/media/eesissi/Data/zwl/TencentData/process_data/w2v_sequence_text_user_id_click_times.pkl'
user_info = '/media/eesissi/Data/zwl/TencentData/train_preliminary/user.csv'

def time_format():
    return time.strftime("%m-%d-%H-%M", time.localtime())

def lambda_func(x,length):
    if len(x)>length:
        return x[:length]
    else:
        x.extend([0]*(length-len(x)))
        return x


def read_w2v(args):
    """
    sentence: include user id, sentence
    user_label: include user id, gender label, age label

    """

    
    w2v_path, sentence, user_label, filename = args
    if not os.path.exists("./{}".format(filename)):
        os.mkdir('./{}'.format(filename))
    data, gender, age, length, v_mean, v_var = [],[],[],[],[],[]
    model = pickle.load(open(w2v_path,'rb'))
    tag_list = model.wv.index2entity
    vectors = model.wv.vectors
    index = list(range(1,len(tag_list)+1))
    dict_zip = dict(zip(tag_list, index))
    vectors = np.vstack((np.zeros((1, vectors.shape[1])),vectors))
    dict_zip.update({'pad':0})
    for s in sentence:
        data.append([dict_zip.get(item) for item in s[1].split()])
        temp = [model[item] for item in s[1].split()]
        v_mean.append(np.asarray(temp).mean(0))
        v_var.append(np.asarray(temp).var(0))
        length.append(len(s[1].split()))

        gender.append(int(user_label[user_label['user_id'].isin([int(s[0])])].gender)-1)
        age.append(int(user_label[user_label['user_id'].isin([int(s[0])])].age)-1)

    
    mean_len = max(length)
    new_data = list(map(lambda x: lambda_func(x,mean_len), data))
    new_data = np.asarray(new_data)
    data = np.asarray(data)
    gender = np.asarray(gender)
    age = np.asarray(age)
    v_mean = np.asarray(v_mean)
    v_var = np.asarray(v_var)


    

    t = time_format()    
    np.save('./{}/embedding_{}_{}_{}.npy'.format(filename,t,str(vectors.shape[0]),str(vectors.shape[1])), vectors)
    np.save('./{}/origin_data_{}.npy'.format(filename,t), data)
    np.save('./{}/process_data_{}_{}_{}.npy'.format(filename,t,str(new_data.shape[0]),str(new_data.shape[1])), new_data)
    np.save('./{}/gender_label_{}_{}.npy'.format(filename,t,gender.shape[0]), gender)
    np.save('./{}/age_label_{}_{}.npy'.format(filename,t,age.shape[0]), age)
    np.save('./{}/v_mean_{}.npy'.format(filename,t), v_mean)
    np.save('./{}/v_var_{}.npy'.format(filename,t), v_var)



if __name__ == "__main__":
    user_df = pd.read_csv(user_info)
    ss = pickle.load(open(sentence_path,'rb'))
    sentence1 = ss['creative_id']
    sentence2 = ss['ad_id']
    sentence3 = ss['product_id']
    sentence4 = ss['click_times']
    args = [[w2v_file_creative, sentence1, user_df, 'creative_max'],
            [w2v_file_ad, sentence2, user_df, 'ad_max'],
            [w2v_file_product, sentence3, user_df, 'product_max'],
            [w2v_file_click_times, sentence4, user_df, 'click_times_max']]
    pool = ThreadPool(32)
    print('Start build dataset...')
    pool.map(read_w2v, args)