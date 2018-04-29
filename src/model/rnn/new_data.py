import sys
sys.path.append('../../..')
import pickle as pkl

from tqdm import tqdm

from src.feature_engineering.feature_API import *
from src.util.data import get_id_list_by_day

train_num = 478138
test_num = 42888
max_seq_len = 50

user = OnehotFeature.get_raw_feature('user_id').data.tolist()
timestamp = CountingFeature.get_raw_feature('context_timestamp').data.tolist()
id = range(train_num + test_num)

id_dict = {
    'train': get_id_list_by_day([1, 2, 3, 4, 5, 6, 7]),
    'test': get_id_list_by_day([8])
}
data = fetch_data(id_dict=id_dict, config_path='feature_config', missing_handler='mean', del_other=False)
feature_size = data.get_feature_size()
feature = data.get_feature('train').toarray()
label = data.get_label('train')

data = list(zip(id[:train_num], user[:train_num], timestamp[:train_num], feature, label))
data = sorted(data, key=lambda d: (d[1], d[2]))

trainseq = []



userseq = {}
max = 0.
for itr in tqdm(range(train_num)):
    user = data[itr][1]
    feature = data[itr][3]
    label = data[itr][4]
    id = data[itr][0]
    if user not in userseq:
        userseq[user] = [[], []]
    userseq[user][0].append(feature)
    userseq[user][1].append(label)
    # trainseq[id] = (userseq[user], label)

for user in userseq:
    trainseq.append(userseq[user])
print(len(trainseq))
trainseq = list(zip(*trainseq))





f = open('feature.pkl', 'wb')
pkl.dump(trainseq, f)
f.close()
