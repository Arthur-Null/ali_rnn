from src.util.data import get_id_list_by_day
from src.util.feature import feature_product
from src.feature_engineering.feature_representation import Feature, MultihotFeature, Data
from tqdm import tqdm

################################
# config
config_path = 'feature_combination_config'
threshold = 100
del_other = True
multihot_norm = False

################################
def feature_combination():
    trian_num = 478138
    train_pos = 0
    train_neg = 0
    for x in Feature.load('is_trade').data:
        if x == 0:
            train_neg += 1
        if x == 1:
            train_pos += 1
    print(train_pos, train_neg, train_neg + train_pos)

    test_num = 18371
    tot_num = trian_num + test_num

    id_dict = {
            'train': range(tot_num)
        }

    feature_config = open(config_path).read().splitlines()
    data = Data(id_dict=id_dict)
    for feature_name in feature_config:
        if feature_name.startswith('#') or feature_name == '':
            continue
        feature = Feature.load(feature_name)
        if feature.type == 'one-hot':
            data.add_feature_dict(feature.fetch(id_dict, del_other=del_other))
        elif feature.type == 'multi-hot':
            data.add_feature_dict(feature.fetch(id_dict, del_other=del_other, norm=multihot_norm))
        else:
            assert NotImplemented("feature combination doesn't support dense feature")

    feature = data.get_feature('train')
    label = data.get_label('train')

    dict, pos_num, neg_num = {}, {}, {}
    for id in tqdm(range(tot_num)):
        indices = feature[id].nonzero()[1]
        length = len(indices)
        for i in range(length):
            for j in range(i, length):
                pair = (indices[i], indices[j])
                if pair not in dict:
                    dict[pair] = []
                    pos_num[pair] = 0
                    neg_num[pair] = 0
                if label[id] == 0:
                    neg_num[pair] += 1
                if label[id] == 1:
                    pos_num[pair] += 1
                dict[pair].append(id)

    auc_list = []

    for x in dict:
        if (pos_num[x] + neg_num[x] > threshold):
            A = train_neg - neg_num[x]
            B = neg_num[x]
            C = train_pos - pos_num[x]
            D = pos_num[x]
            auc = 0.5 * (A*C + 2*A*D + B*D) / (A+B) / (C+D)
            auc_list.append((x, auc))

    auc_list = sorted(auc_list, key=lambda x:x[1], reverse=True)
    for i in range(10):
        print(i, auc_list[i])
    print(1000, auc_list[1000])
    print(5000, auc_list[5000])
    print(10000, auc_list[10000])

    # 1k
    chosen = set()
    for i in range(1000):
        chosen.add(auc_list[i][0])
    feature_comb = [[] for i in range(tot_num)]
    num_list = [0 for i in range(tot_num)]
    for x in chosen:
        for id in dict[x]:
            feature_comb[id].append(x)
            num_list[id] += 1

    MultihotFeature(name='feature_comb_1k', data=feature_comb, threshold=0).save()

    # 5k
    chosen = set()
    for i in range(5000):
        chosen.add(auc_list[i][0])

    feature_comb = [[] for i in range(tot_num)]
    for x in chosen:
        for id in dict[x]:
            feature_comb[id].append(x)

    MultihotFeature(name='feature_comb_5k', data=feature_comb, threshold=0).save()

if __name__ == '__main__':
    # feature_combination()
    # feature = MultihotFeature.load('feature_comb_1k')
    # data = feature.data
    # print(len(data))
    # for i in range(10):
    #     print(len(data[i]), data[i][0], data[i][1], data[i][2], data[i][-1])
    feature_product('user_cluster_100', 'item_sales_level', 'user_item_sales_level').save()
    feature_product('user_cluster_100', 'item_collected_level', 'user_item_collected_level').save()
    feature_product('user_cluster_100', 'item_pv_level', 'user_item_pv_level').save()
    feature_product('user_cluster_100', 'item_brand_id_t5', 'user_item_brand_id_t5').save()