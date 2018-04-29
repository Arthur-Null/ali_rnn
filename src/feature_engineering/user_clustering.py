import sys
sys.path.append('../..')
from src.util.data import get_id_list_by_day
from src.feature_engineering.feature_representation import Feature, MultihotFeature, OnehotFeature, Data
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np

################################
# config
config_path = 'feature_clustering_config'
del_other = True
multihot_norm = False

################################
def user_clustering(n_clustering = 10):
    trian_num = 478138
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
            data.add_feature_dict(feature.fetch(id_dict))

    feature = data.get_feature('train')
    label = data.get_label('train')

    kmeans = KMeans(n_clusters=n_clustering, random_state=0).fit(feature)
    print("n_clustering = ", n_clustering)
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)

    OnehotFeature(name='user_cluster_' + str(n_clustering), data=kmeans.labels_, threshold=0).save()

    cnt = np.zeros((n_clustering, 2))
    tmp = 0
    tmp0 = 0
    tmp1 = 0
    for id in tqdm(range(trian_num)):
        # print(type(kmeans.labels_[tmp]))
        # print(type(label[id]))
        lb = int(label[id] == 1)
        cnt[kmeans.labels_[tmp]][lb] += 1
        if lb == 0:
            tmp0 += 1
        else:
            tmp1 += 1
        tmp += 1

    ctr = 1.0 * tmp1 / (tmp0 + tmp1)
    print("initial ctr =", ctr)
    for i in range(n_clustering):
        if cnt[i][0] + cnt[i][1] == 0:
            break
        print("", cnt[i][0] + cnt[i][1], (1.0 * cnt[i][1] / (cnt[i][0] + cnt[i][1]) - ctr) / ctr)
