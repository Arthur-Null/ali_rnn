from src.feature_engineering.feature_representation import *
from src.util.log import print_log
import time

def submit(pred, instance_id, file='test_submit.txt'):
    __submit(pred, instance_id, file)

def get_id_list_by_day(day_list):
    day_dict = {}
    day_feature = [time.localtime(x).tm_yday for x in CountingFeature.get_raw_feature('context_timestamp').data]
    day_feature = OnehotFeature(name='day', data=day_feature).data
    for i, day in enumerate(day_feature):
        if not day in day_dict:
            day_dict[day] = []
        day_dict[day].append(i)
    list = []
    for day in day_list:
        list = list + day_dict[day]
    return list



def get_id_dict(type, number=0, test_flag:bool=False, label_flag:bool=False, all = False):
    # Parameter:
    #   type = 'basic_model' or 'ensemble'
    #
    #   number = 0, 1, 2, 3, 4
    #
    #   test_flag=True: using 'test'
    #            =False
    #
    #   label_flag=True: create label at the same time
    #             =False
    #
    # Output:
    #     id_dict = {
    #         'train': [1, 2, 3 ,4 ,5],
    #         'validation': [6, 7, 8 ,9, 10],
    #         'test': [10001, 10002...] (maybe not exist)
    #     }
    #     label_dict = {
    #         'train': [1, 1, 0, 0, 0],
    #         'validation': [0, 1, 0, 0, 0],
    #         'test': [-1, -1, -1...] (maybe not exist)
    #     }(maybe not exist)
    return __get_id_dict(type, number, test_flag, label_flag, all)

train_num = 478138
test_num = 18371
day_length = int(np.ceil(train_num / 5.))
a = np.arange(0, train_num)
import random

random.seed(10)
random.shuffle(a)
random.shuffle(a)
random.shuffle(a)
random.shuffle(a)
random.shuffle(a)
print(a)

day = {
    1: a[0 * day_length:1 * day_length].tolist(),
    2: a[1 * day_length:2 * day_length].tolist(),
    3: a[2 * day_length:3 * day_length].tolist(),
    4: a[3 * day_length:4 * day_length].tolist(),
    5: a[4 * day_length:train_num].tolist(),
}
basic_model_id = {
    0: (day[1] + day[2] + day[3], day[4]),
    1: (day[1] + day[2] + day[4], day[3]),
    2: (day[1] + day[3] + day[4], day[2]),
    3: (day[3] + day[2] + day[4], day[1]),
    4: (day[1] + day[2] + day[3] + day[4], day[5]),
    5: (day[1] + day[2] + day[3] + day[4] + day[5], []),
}

cv_4_fold = {
    0: (day[1] + day[2] + day[3] + day[4], day[5]),
    1: (day[2] + day[3] + day[4], day[1]),
    2: (day[1] + day[3] + day[4], day[2]),
    3: (day[1] + day[2] + day[4], day[3]),
    4: (day[1] + day[2] + day[3], day[4]),
}

cv_5_fold = {
    0: (day[1] + day[2] + day[3] + day[4] + day[5], []),
    1: (day[2] + day[3] + day[4] + day[5], day[1]),
    2: (day[1] + day[3] + day[4] + day[5], day[2]),
    3: (day[1] + day[2] + day[4] + day[5], day[3]),
    4: (day[1] + day[2] + day[3] + day[5], day[4]),
    5: (day[1] + day[2] + day[3] + day[4], day[5]),
}

# single_model_id = (day[1] + day[2] + day[3] + day[4] + day[5] + day[6], day[7])

def divide(list):
    length = len(list)
    train_length = int(length * 3 / 4)
    return list[:train_length], list[train_length:]

# ensemble_id = {
#     0: divide(day[3] + day[4]),
#     1: divide(day[4] + day[5]),
#     2: divide(day[5] + day[6]),
#     3: divide(day[6] + day[7]),
# }

def __get_id_dict(type, number=0, test_flag:bool=False, label_flag:bool=False, all = False):
    if all == True:
        id_pair = (list(range(0, train_num)), [])
    elif type == 'basic_model':
        id_pair = basic_model_id[number]
    elif type == 'cv_4_fold':
        id_pair = cv_4_fold[number]
    elif type == 'cv_5_fold':
        id_pair = cv_5_fold[number]
    else:
        raise NotImplementedError('type can not be {0}'.format(type))
    id_dict = {
        'train': id_pair[0],
        'validation': id_pair[1]
    }
    if test_flag:
        id_dict['test'] = list(range(train_num, train_num + test_num))

    print_log('finished creating id_dict --- train:{0}, validation:{1}'.format(len(id_dict['train']), len(id_dict['validation'])))
    if not label_flag:
        return id_dict

    # need label
    label_dict = {}
    for split, id_list in id_dict.items():
        label_dict[split] = []
        feature = Feature.load('is_trade')
        for id in id_list:
            label_dict[split].append(feature.data[id])
        label_dict[split] = np.asarray(label_dict[split])
    print_log('finished creating label_dict')
    return id_dict, label_dict

def __submit(pred, instance_id, file):
    if len(pred) != test_num:
        raise AssertionError('length of pred shoud be the same as test')

    with open(file, 'w') as fw:
        fw.write('{} {}\n'.format('instance_id', 'predicted_score'))
        for id, p in zip(instance_id, pred):
            fw.write('{} {}\n'.format(id, p))