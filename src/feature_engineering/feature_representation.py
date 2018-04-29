import abc
from sklearn.externals import joblib
import os
import numpy as np
from src.util.log import print_log
from scipy.sparse.csr import csr_matrix
from scipy.sparse import hstack

default_data_path = os.path.join(os.path.dirname(__file__), '../../data')

class Feature(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.length = 1
        # one-hot / multi-hot must change length to category num

    def save(self, dir=os.path.join(default_data_path, 'feature')):
        if not os.path.exists(dir):
            os.makedirs(dir)
        joblib.dump(self, os.path.join(dir, self.name + '.pkl'))
        if not self.name in ['instance_id', 'context_id', 'is_trade']:
            with(open(os.path.join(os.path.dirname(__file__), 'all_features'), 'a+')) as f:
                print('{0}\n#type: {1}\n'.format(self.name, self.type), file=f)

    def save_as(self, name, dir=os.path.join(default_data_path, 'feature')):
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.name = name
        joblib.dump(self, os.path.join(dir, name + '.pkl'))
        if not name in ['instance_id', 'context_id', 'is_trade']:
            with(open(os.path.join(os.path.dirname(__file__), 'all_features'), 'a+')) as f:
                print('{0}\n#type: {1}\n'.format(name, self.type), file=f)

    @staticmethod
    def load(name, dir=os.path.join(default_data_path, 'feature')):
        return joblib.load(os.path.join(dir, name + '.pkl'))

class CountingFeature(Feature):
    def __init__(self, name, data):
        super(CountingFeature, self).__init__(name, 'counting')
        self.data, self.info = self.manipulate(data)

    @staticmethod
    def get_raw_feature(name, raw_name=None):
        if raw_name == None:
            raw_name = name
        data = joblib.load(os.path.join(default_data_path, 'pretreatment/raw-{0}.pkl'.format(raw_name)))
        data = np.asarray([int(x) for x in data])
        return CountingFeature(name, data)

    def manipulate(self, input):
        feature = np.asarray([int(x) for x in input]) if isinstance(input, list) else input
        sum, num = 0, 0
        for x in feature:
            if x != -1:
                sum += x
                num += 1
        info = {
            'max': np.max(feature),
            'min': np.min(feature),
            'mean': 1.0 * sum / num,
            'var': np.var(feature),
            'missing_rate': 1. * np.sum(feature == -1) / len(input)
        }
        print_log('manipulation of {0}(type: {1}) finished'.format(self.name, self.type))
        return feature, info

    def fetch(self, id_dict:dict, missing_handler='nothing'):
        output = {}
        for split, id_list in id_dict.items():
            row, col, val = [], [], []
            for i, id in enumerate(id_list):
                if self.data[id] != -1 or missing_handler == 'nothing':
                    row.append(i)
                    col.append(0)
                    val.append(self.data[id])
                elif missing_handler == 'mean':
                    row.append(i)
                    col.append(0)
                    val.append(self.info['mean'])
            output[split] = csr_matrix((np.asarray(val), (np.asarray(row), np.asarray(col))), shape=(len(id_list), 1))
        return output


class NumericalFeature(Feature):
    def __init__(self, name, data):
        super(NumericalFeature, self).__init__(name, 'numerical')
        self.data, self.info = self.manipulate(data)

    @staticmethod
    def get_raw_feature(name, raw_name=None):
        if raw_name == None:
            raw_name = name
        data = joblib.load(os.path.join(default_data_path, 'pretreatment/raw-{0}.pkl'.format(raw_name)))
        data = np.asarray([float(x) for x in data])
        return NumericalFeature(name, data)

    def manipulate(self, input):
        feature = np.asarray([float(x) for x in input]) if isinstance(input, list) else input
        sum, num = 0, 0
        for x in feature:
            if x != -1:
                sum += x
                num += 1
        info = {
            'max': np.max(feature),
            'min': np.min(feature),
            'mean': 1.0 * sum / num,
            'var': np.var(feature),
            'missing_rate': 1. * np.sum(feature == -1) / len(input)
        }
        print_log('manipulation of {0}(type: {1}) finished'.format(self.name, self.type))
        return feature, info

    def fetch(self, id_dict:dict, missing_handler='nothing'):
        output = {}
        for split, id_list in id_dict.items():
            row, col, val = [], [], []
            for i, id in enumerate(id_list):
                if self.data[id] != -1 or missing_handler == 'nothing':
                    row.append(i)
                    col.append(0)
                    val.append(self.data[id])
                elif missing_handler == 'mean':
                    row.append(i)
                    col.append(0)
                    val.append(self.info['mean'])
            output[split] = csr_matrix((np.asarray(val), (np.asarray(row), np.asarray(col))), shape=(len(id_list), 1))
        return output

class OnehotFeature(Feature):
    def __init__(self, name, data, threshold=0):
        super(OnehotFeature, self).__init__(name, 'one-hot')
        self.data, self.info = self.manipulate(data, threshold)
        self.id2val = {}

    @staticmethod
    def get_raw_feature(name, threshold=0, raw_name=None):
        if raw_name == None:
            raw_name = name

        data = joblib.load(os.path.join(default_data_path, 'pretreatment/raw-{0}.pkl'.format(raw_name)))
        return OnehotFeature(name, data, threshold)

    def manipulate(self, input, threshold):
        id2cat, cat2id, count_info = {}, {}, {}
        for x in input:
            if not x in count_info:
                count_info[x] = 0
            count_info[x] += 1

        cat2id['__other__'] = 0
        id2cat[0] = '__other__'
        other_num = 0
        for key, val in count_info.items():
            if val < threshold or key == '-1':
                other_num += val
            else:
                cat2id[key] = len(cat2id)
                id2cat[cat2id[key]] = key
        feature = np.asarray([cat2id.get(x, cat2id['__other__']) for x in input])
        info = {
            'category_num': len(cat2id),
            'biggest_size': max(count_info.values()),
            'threshold': threshold,
            'tail_size': other_num
        }
        self.length = len(cat2id)
        self.cat2id = cat2id
        self.id2cat = id2cat
        print_log('manipulation of {0}(type: {1}) finished'.format(self.name, self.type))
        return feature, info

    def add_value(self, cat2val:dict):
        for cat, val in cat2val.items():
            self.id2val[self.cat2id.get(cat, 0)]=val
        if '__other__' in cat2val:
            self.id2val[0] = cat2val['__other__']

    def fetch(self, id_dict:dict, del_other=False):
        train_cat_set = set()
        for id in id_dict['train']:
            x = self.data[id]
            train_cat_set.add(x)

        output = {}
        for split, id_list in id_dict.items():
            row, col, val = [], [], []
            for i, id in enumerate(id_list):
                xx = self.data[id]
                if not xx in train_cat_set:
                    xx = 0
                if xx != 0 or del_other == False:
                    row.append(i)
                    col.append(xx)
                    # val.append(self.id2val.get(xx, 1))
                    # TODO
                    val.append(1)
            output[split] = csr_matrix((np.asarray(val), (np.asarray(row), np.asarray(col))),
                                       shape=(len(id_list), self.length))
        return output

class MultihotFeature(Feature):
    def __init__(self, name, data, threshold):
        super(MultihotFeature, self).__init__(name, 'multi-hot')
        self.data, self.info = self.manipulate(data, threshold)
        self.id2val = {}

    @staticmethod
    def get_raw_feature(name, threshold=0, raw_name=None):
        if raw_name == None:
            raw_name = name
        data = joblib.load(os.path.join(default_data_path, 'pretreatment/raw-{0}.pkl'.format(raw_name)))
        data = [x.split(';') for x in data]
        return MultihotFeature(name, data, threshold)

    def manipulate(self, input, threshold):
        id2cat, cat2id, count_info = {}, {}, {}
        for line in input:
            for x in line:
                if not x in count_info:
                    count_info[x] = 0
                count_info[x] += 1

        cat2id['__other__'] = 0
        id2cat[0] = '__other__'
        other_num = 0
        for key, val in count_info.items():
            if val < threshold or key == '-1':
                other_num += val
            else:
                cat2id[key] = len(cat2id)
                id2cat[cat2id[key]] = key
        feature = []
        for line in input:
            feature.append([cat2id.get(x, cat2id['__other__']) for x in line])
        info = {
            'category_num': len(cat2id),
            'biggest_size': max(count_info.values()),
            'threshold': threshold,
            'tail_size': other_num
        }
        self.length = len(cat2id)
        self.cat2id = cat2id
        self.id2cat = id2cat
        print_log('manipulation of {0}(type: {1}) finished'.format(self.name, self.type))
        return feature, info

    def add_value(self, cat2val:dict):
        for cat, val in cat2val.items():
            self.id2val[self.cat2id.get(cat, 0)]=val
        if '__other__' in cat2val:
            self.id2val[0] = cat2val['__other__']

    def fetch(self, id_dict:dict, del_other=False, norm=False):
        train_cat_set = set()
        for id in id_dict['train']:
            for x in self.data[id]:
                train_cat_set.add(x)

        output = {}
        for split, id_list in id_dict.items():
            row, col, val = [], [], []
            for i, id in enumerate(id_list):
                num = len(self.data[id])
                for x in self.data[id]:
                    xx = x
                    if not xx in train_cat_set:
                        xx = 0
                    if xx != 0 or del_other == False:
                        row.append(i)
                        col.append(xx)
                        if norm:
                            val.append(1.0 * self.id2val.get(xx, 1) / num)
                        else:
                            val.append(self.id2val.get(xx, 1))
            output[split] = csr_matrix((np.asarray(val), (np.asarray(row), np.asarray(col))),
                                       shape=(len(id_list), self.length))
        return output

class Data(object):
    def __init__(self, id_dict, ):
        self.id_dict = id_dict
        self.length_list = []
        self.feature_dict = {}
        for split in id_dict.keys():
            self.feature_dict[split] = csr_matrix(np.ones((len(id_dict[split]),0)))

    def add_feature_dict(self, feature_dict):
        length = None
        for split in self.id_dict.keys():
            self.feature_dict[split] = hstack((self.feature_dict[split], feature_dict[split]))
            if length is None:
                length = feature_dict[split].shape[1]
        self.length_list.append(length)

    def merge_with(self, other):
        self.length_list += other.length_list
        for split in self.id_dict.keys():
            self.feature_dict[split] = hstack((self.feature_dict[split], other.feature_dict[split]))

    def get_feature(self, split):
        return self.feature_dict[split].tocsr()

    def get_label(self, split):
        label = Feature.load('is_trade')
        return np.asarray([label.data[i] for i in self.id_dict[split]])

    def get_instance_id(self, split):
        instance_id = Feature.load('instance_id')
        return np.asarray([instance_id.data[i] for i in self.id_dict[split]])

    def get_feature_size(self):
        sum = 0
        for x in self.length_list:
            sum += x
        return sum

    @property
    def num_features(self):
        return self.get_feature_size()

    @property
    def num_fields(self):
        return len(self.length_list)

    @property
    def field_sizes(self):
        return self.length_list

    @property
    def field_index_start(self):
        return [sum(self.field_sizes[:i]) for i in range(self.num_fields)]

    @property
    def train_size(self):
        return self.get_label('train').shape[0]

    @property
    def valid_size(self):
        return self.get_label('validation').shape[0]

    @property
    def test_size(self):
        return self.get_label('test').shape[0]

    def split_csr(self, csr_mat):
        csc_mat = csr_mat.tocsc()
        fields = []
        field_index_start = self.field_index_start
        for i in range(self.num_fields - 1):
            field_start = field_index_start[i]
            field_end = field_index_start[i+1]
            field_i = csc_mat[:, field_start:field_end]
            fields.append(field_i)
        fields.append(csc_mat[:, field_index_start[-1]:])
        return fields

    @staticmethod
    def sparse2input(sparse_mat):
        if type(sparse_mat) is list:
            inputs = []
            for sm in sparse_mat:
                inputs.append(Data.sparse2input(sm))
            return inputs
        else:
            coo_mat = sparse_mat.tocoo()
            indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
            values = coo_mat.data
            shape = coo_mat.shape
            return indices, values, shape

    def batch_generator(self, gen_type='train', batch_size=1000, shuffle=False, split_fields=True):
        data = self.get_feature(gen_type)
        label = self.get_label(gen_type)
        assert data.shape[0] == label.shape[0], 'input and output should have same length'
        total_size = label.shape[0]
        indices = np.arange(total_size)
        if shuffle:
            np.random.shuffle(indices)
        # print('total steps:', int(np.ceil(total_size * 1. / batch_size)))
        for i in range(int(np.ceil(total_size * 1. / batch_size))):
            batch_indices = indices[batch_size * i : batch_size * (i+1)]
            batch_data = data[batch_indices]
            batch_label = label[batch_indices]
            if split_fields:
                batch_data = self.split_csr(batch_data)
            batch_data = Data.sparse2input(batch_data)
            yield batch_data, batch_label
