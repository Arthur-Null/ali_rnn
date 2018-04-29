from src.feature_engineering.feature_representation import *
from scipy.sparse.csr import csr_matrix

def feature_product(feature1, feature2, name='__product', threshold=5):
    if isinstance(feature1, str):
        feature1 = Feature.load(feature1)
    if isinstance(feature2, str):
        feature2 = Feature.load(feature2)
    if not isinstance(feature1, OnehotFeature):
        raise NotImplemented('feature1 should be Onehot')
    if not isinstance(feature2, OnehotFeature):
        raise NotImplemented('feature2 should be Onehot')
    list = []
    for x, y in zip(feature1.data, feature2.data):
        list.append((x, y))
    return OnehotFeature(name=name, data=list, threshold=threshold)


def feature_to_list(id_dict:dict, dense_feature_dict:dict, sparse_feature_dict:dict, sparse_size_list:list, missing_handler:str):
    feature_dict = {}
    for split in id_dict.keys():
        feature_dict[split] = __to_list(len(id_dict[split]), dense_feature_dict[split], sparse_feature_dict[split], sparse_size_list, missing_handler)
    return feature_dict

def __to_list(length, dense:list, sparse, sparse_size_list, missing_handler:str):
    feature = [[] for x in range(length)]
    offset = 0
    for dense_feature in dense:
        for i, x in enumerate(dense_feature):
            if missing_handler != 'del' or x != -1:
                feature[i].append((offset, x))
        offset += 1
    for size, sparse_feature in zip(sparse_size_list, sparse):
        for i, feature_list in enumerate(sparse_feature):
            for pair in feature_list: #pair: (index, val)
                feature[i].append((pair[0] + offset, pair[1]))
        offset += size

    return feature

def feature_to_sparse_matrix(id_dict:dict, dense_feature_dict:dict, sparse_feature_dict:dict, sparse_size_list:list, missing_handler:str):
    feature_dict = {}
    for split in id_dict.keys():
        feature_dict[split] = __to_sparse_matrix(len(id_dict[split]), dense_feature_dict[split], sparse_feature_dict[split], sparse_size_list, missing_handler)
    return feature_dict

def __to_sparse_matrix(length, dense, sparse, sparse_size_list, missing_handler:str):
    row, col, data = [], [], []
    offset = 0
    for dense_feature in dense:
        for i, x in enumerate(dense_feature):
            if missing_handler != 'del' or x != -1:
                row.append(i)
                col.append(offset)
                data.append(x)
        offset += 1
    for size, sparse_feature in zip(sparse_size_list, sparse):
        for i, feature_list in enumerate(sparse_feature):
            for pair in feature_list:
                row.append(i)
                col.append(offset + pair[0])
                data.append(pair[1])
        offset += size
    return csr_matrix((np.asarray(data), (np.asarray(row), np.asarray(col))), shape=(length, offset))

def feature_to_padding(id_dict:dict, dense_feature_dict:dict, sparse_feature_dict:dict, sparse_size_list:list):
    max_category_size_list = [0 for size in sparse_size_list]
    for split in sparse_feature_dict.keys():
        data = sparse_feature_dict[split]
        for id, feature in enumerate(data):
            for line in feature:
                max_category_size_list[id] = max(max_category_size_list[id], len(line))
    feature_dict = {}
    for split in dense_feature_dict.keys():
        feature_dict[split] = __to_padding(dense_feature_dict[split], sparse_feature_dict[split], sparse_size_list, max_category_size_list)
    return feature_dict, max_category_size_list

def __to_padding(dense, sparse, sparse_size_list, max_category_size_list):
    if len(dense) > 0:
        mask = np.stack(dense, axis=1)
        ids = np.ones_like(mask) * list(range(mask.shape[1]))
        for max_size, feature in zip(max_category_size_list, sparse):
            _ids, _mask = __padding(feature, max_size)
            ids = np.concatenate((ids, _ids), axis=1)
            mask = np.concatenate((mask, _mask), axis=1)
    else:
        ids = []
        mask = []
        for max_size, feature in zip(max_category_size_list, sparse):
            _ids, _mask = __padding(feature, max_size)
            ids.append(_ids)
            mask.append(_mask)
        ids = np.concatenate(ids, axis=1)
        mask = np.concatenate(mask, axis=1)
    return np.concatenate((ids, mask), axis=1)

def __padding(feature, max_size):
    ids = np.zeros(shape=(len(feature), max_size))
    mask = np.zeros(shape=(len(feature), max_size))
    for i, line in enumerate(feature):
        line = sorted(line, key=lambda x: x or np.inf)
        for j, id in enumerate(line):
            if id == 0:
                break
            ids[i][j] = id
            mask[i][j] = 1
    return ids, mask

def deal(list:list, map:dict):

    row = []
    col = []
    value = []

    max = 0

    for index, j in enumerate(list):
        for i in j:
            row.append(index)
            col.append(map.get(i[0], 0))
            value.append(i[1])
        max = index

    max = max + 1
    return csr_matrix((np.asarray(value), (np.asarray(row), np.asarray(col))), shape=(max, len(map) + 1))


def compress_feature(feature:dict, threhold):
    feature_map = {}

    for j in feature['train']:
        for i in j:
            if feature_map.get(i[0], 0) == 0:
                feature_map[i[0]] = 1
            else:
                feature_map[i[0]] += 1

    useful_map = {}

    for j in feature_map.items():
        if feature_map.get(j[0]) > threhold:
            useful_map[j[0]] = len(useful_map) + 1
    print(len(useful_map))
    feature['train'] = deal(feature['train'], useful_map)
    feature['test'] = deal(feature['test'], useful_map)
    if feature.get('validation', 0) != 0:
        feature['validation'] = deal(feature['validation'], useful_map)
    return feature
