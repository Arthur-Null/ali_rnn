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
    # for k, v in day_dict.items():
    #     print(k, len(v))
    list = []
    for day in day_list:
        list = list + day_dict[day]
    return list



def get_id_dict(type, number=0, test_flag:bool=False, label_flag:bool=False, all = False, slice_num = 5):
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
    #   slice_num = 5: The number of data division.
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
    return __get_id_dict(type, number, test_flag, label_flag, all, slice_num)

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


#
label = Feature.load('is_trade')
label = label.data[:train_num]
true = []
false = []
for index, i in enumerate(label):
   if abs(i - 1) < 0.0001:
        true.append(index)
   else:
        false.append(index)
random.seed(10)
random.shuffle(true)
random.shuffle(false)

#


# single_model_id = (day[1] + day[2] + day[3] + day[4] + day[5] + day[6], day[7])
#
# def divide(list):
#     length = len(list)
#     train_length = int(length * 3 / 4)
#     return list[:train_length], list[train_length:]

# ensemble_id = {
#     0: divide(day[3] + day[4]),
#     1: divide(day[4] + day[5]),
#     2: divide(day[5] + day[6]),
#     3: divide(day[6] + day[7]),
# }

def __get_id_dict(type, number=0, test_flag:bool=False, label_flag:bool=False, all = False, slice_num = 5):
    global true, false
    true_day_length = int(1. * np.ceil(len(true) / slice_num))
    false_day_length = int(1. * np.ceil(len(false) / slice_num))

    true = np.array(true)
    false = np.array(false)

    day = {}

    for i in range(slice_num):
        day[i] = np.concatenate([true[i * true_day_length:min(len(true), (i + 1) * true_day_length)],
                              false[i * false_day_length:min(len(false), (i + 1) * false_day_length)]])
        random.seed(10)
        random.shuffle(day[i])
        day[i] = day[i].tolist()

    basic_model_id = {}
    for i in range(slice_num - 1):
        tmp = []
        for j in range(slice_num - 1):
            if i == j:
                continue
            tmp = tmp + day[j]
        basic_model_id[i] = (tmp, day[i])

    all_train = []
    except_last_train = []
    for i in range(slice_num):
        if i != slice_num - 1:
            except_last_train = except_last_train + day[i]
        all_train = all_train + day[i]
    basic_model_id[slice_num - 1] = (except_last_train, day[slice_num - 1])
    basic_model_id[slice_num] = (all_train, [])


    if all == True:
        id_pair = (list(range(0, train_num)), [])
    elif type == 'basic_model':
        id_pair = basic_model_id[number]
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

def print_id(slice_num = 5):
    global true, false
    true_day_length = int(1. * np.ceil(len(true) / slice_num))
    false_day_length = int(1. * np.ceil(len(false) / slice_num))

    true = np.array(true)
    false = np.array(false)

    day = {}

    for i in range(slice_num):
        day[i] = np.concatenate([true[i * true_day_length:min(len(true), (i + 1) * true_day_length)],
                                 false[i * false_day_length:min(len(false), (i + 1) * false_day_length)]])
        random.seed(10)
        random.shuffle(day[i])
        day[i] = day[i].tolist()

    all_train = []
    for i in range(slice_num):
        all_train = all_train + day[i]
    return all_train