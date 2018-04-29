from src.feature_engineering.feature_representation import *
import matplotlib.pyplot as plt

def sort_by_time(x):
    return [x[t[1]] for t in time_list]

def restore_by_time(x):
    res = x.copy()
    for i in range(len(x)):
        res[time_list[i][1]] = x[i]
    return res

def item_category_combine(user_feature):
    user_data = Feature.load('user_' + user_feature).data
    item_category_data = Feature.load('item_category_list').data
    label_data = Feature.load('is_trade').data
    tot_num, positive_num = {}, {}
    ctr_list = []
    for x, category_list, label in zip(user_data, item_category_data, label_data):
        tmp_data = []
        for cat in category_list:
            if tot_num.get((x, cat), 0) == 0:
                tot_num[(x, cat)] = 0
                positive_num[(x, cat)] = 0
                tmp_data.append(0)
            else:
                tmp_data.append(1.0 * positive_num[(x, cat)] / tot_num[(x, cat)])
            if label != -1:
                tot_num[(x, cat)] += 1
            if label == 1:
                positive_num[(x, cat)] += 1
        ctr_list.append(max(tmp_data))
    new_name = user_feature + '_item_category_ctr'
    print(new_name)
    NumericalFeature(name=new_name, data=ctr_list).save()

def ctr_combine(user_feature, item_feature):
    user_data = Feature.load('user_' + user_feature).data
    item_data = Feature.load('item_' + item_feature).data
    label_data = Feature.load('is_trade').data
    tot_num, positive_num = {}, {}
    ctr_list = []
    for x, y, label in zip(user_data, item_data, label_data):
        if tot_num.get((x, y), 0) == 0:
            tot_num[(x, y)] = 0
            positive_num[(x, y)] = 0
            ctr_list.append(0)
        else:
            ctr_list.append(1.0 * positive_num[(x, y)] / tot_num[(x, y)])
        if label != -1:
            tot_num[(x, y)] += 1
        if label == 1:
            positive_num[(x, y)] += 1
    new_name = user_feature + '_' + item_feature + '_ctr'
    print(new_name)
    NumericalFeature(name=new_name, data=ctr_list).save()

if __name__ == '__main__':
    time_list = []
    for t in CountingFeature.get_raw_feature(name='context_timestamp').data:
        time_list.append((t, len(time_list)))
    time_list = sorted(time_list, key=lambda x:x[0])

    for x in ['gender_id', 'age_level', 'occupation_id']:
        item_category_combine(x)

    for x in ['gender_id', 'age_level', 'occupation_id']:
        for y in ['price_level', 'sales_level', 'collected_level', 'pv_level', 'city_id']:
            ctr_combine(x, y)