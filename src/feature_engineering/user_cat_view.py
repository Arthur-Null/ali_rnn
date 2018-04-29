from tqdm import tqdm

from src.feature_engineering.feature_representation import MultihotFeature, OnehotFeature, CountingFeature

train_num = 478138
test_num = 42888


def user_cat_view_num():
    cat = MultihotFeature.get_raw_feature(name='item_category_list', threshold=0).data
    user = OnehotFeature.get_raw_feature('user_id').data.tolist()
    timestamp = CountingFeature.get_raw_feature('context_timestamp').data.tolist()
    id = range(train_num + test_num)
    user_cat_view_num = []
    data = list(zip(id, cat, user, timestamp))
    data = sorted(data, key=lambda d: d[3])
    count = {}
    for itr in tqdm(range(train_num + test_num)):
        cat = data[itr][1]
        user = data[itr][2]
        id = data[itr][0]
        if not user in count:
            count[user] = [0] * 17
            for i in cat:
                count[user][i] = 1
            user_cat_view_num.append((id, 0))
        else:
            m = 0
            for i in cat:
                m = m if m > count[user][i] else count[user][i]
                count[user][i] += 1
            user_cat_view_num.append((id, m))
    result = sorted(user_cat_view_num, key=lambda d: d[0])
    result = list(list(zip(*result))[1])
    CountingFeature('user_cat_view_num', result).save()
    return result


def cluster_user_cat_view():
    cat = MultihotFeature.get_raw_feature(name='item_category_list', threshold=0).data
    print(len(cat))
    user = OnehotFeature.load('user_cluster_100').data.tolist()
    print(len(user))
    timestamp = CountingFeature.get_raw_feature('context_timestamp').data.tolist()
    print(len(timestamp))
    id = range(train_num + test_num)
    user_cat_view_num = []
    data = list(zip(id, cat, user, timestamp))
    data = sorted(data, key=lambda d: d[3])
    count = {}
    print(len(data))
    for itr in tqdm(range(train_num + test_num)):
        # print(data[itr])
        cat = data[itr][1]
        user = data[itr][2]
        id = data[itr][0]
        if not user in count:
            count[user] = [0] * 17
            for i in cat:
                count[user][i] = 1
            user_cat_view_num.append((id, 0))
        else:
            m = 0
            for i in cat:
                m = m if m > count[user][i] else count[user][i]
                count[user][i] += 1
            user_cat_view_num.append((id, m))
    result = sorted(user_cat_view_num, key=lambda d: d[0])
    result = list(list(zip(*result))[1])
    CountingFeature('cluster_user_cat_view_num', result).save()
    return result


if __name__ == '__main__':
    data = user_cat_view_num()
    CountingFeature('user_cat_view_num', data).save()
