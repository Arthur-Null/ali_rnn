import sys
sys.path.append('../../')
from src.feature_engineering.feature_analysis import *
from src.util.log import print_log
from src.feature_engineering.feature_combination import feature_combination
from src.feature_engineering.user_cat_view import user_cat_view_num, cluster_user_cat_view
from src.feature_engineering.user_clustering import user_clustering

import time

default_data_path = os.path.join(os.path.dirname(__file__), '../../data')

def __pretreatment():
    # mkdir
    if not os.path.exists(os.path.join(default_data_path, 'pretreatment')):
        os.mkdir(os.path.join(default_data_path, 'pretreatment'))
    if not os.path.exists(os.path.join(default_data_path, 'feature')):
        os.mkdir(os.path.join(default_data_path, 'feature'))

    dict = {}
    # get raw category
    with open(os.path.join(default_data_path, 'raw data/round1_train.txt')) as f:
        category_list = f.readline().split()

    # read data & split
    data_dict = {}
    for split in ['train', 'test']:
        data_dict[split] = []
        with open(os.path.join(default_data_path, 'raw data/round1_{0}.txt'.format(split))) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                data_list = line.split()
                if split == 'test':
                    data_list.append(-1)
                data_dict[split].append((int(data_list[16]), data_list))

    # train reorder
    data_dict['train'] = sorted(data_dict['train'], key=lambda data: data[0])


    # merge train and test
    for category in category_list:
        dict[category] = []
    for split in ['train', 'test']:
        for x in data_dict[split]:
            for category, data in zip(category_list, x[1]):
                dict[category].append(data)

    # write
    for key, val in dict.items():
        joblib.dump(val, os.path.join(default_data_path, 'pretreatment/raw-{0}.pkl'.format(key)))
        print_log('pretreatment of {0} finished!'.format(key))

def get_basic_feature():
    # item-11
    OnehotFeature.get_raw_feature(name='item_id', threshold=0).save()
    OnehotFeature.get_raw_feature(name='item_id', threshold=3).save_as('item_id_t3')
    OnehotFeature.get_raw_feature(name='item_id', threshold=5).save_as('item_id_t5')
    OnehotFeature.get_raw_feature(name='item_id', threshold=10).save_as('item_id_t10')
    MultihotFeature.get_raw_feature(name='item_category_list', threshold=3).save()
    # TODO
    # MultihotFeature.get_raw_feature(name='item_property_list', threshold=0, raw_name='item_property_list').save()
    # MultihotFeature.get_raw_feature(name='item_property_list_5w', threshold=3, raw_name='item_property_list').save()
    # MultihotFeature.get_raw_feature(name='item_property_list_3w', threshold=12, raw_name='item_property_list').save()
    # MultihotFeature.get_raw_feature(name='item_property_list_1w', threshold=95, raw_name='item_property_list').save()
    OnehotFeature.get_raw_feature(name='item_brand_id', threshold=0).save()
    OnehotFeature.get_raw_feature(name='item_brand_id', threshold=3).save_as('item_brand_id_t3')
    OnehotFeature.get_raw_feature(name='item_brand_id', threshold=5).save_as('item_brand_id_t5')
    OnehotFeature.get_raw_feature(name='item_brand_id', threshold=10).save_as('item_brand_id_t10')
    OnehotFeature.get_raw_feature(name='item_city_id', threshold=0).save()
    OnehotFeature.get_raw_feature(name='item_city_id', threshold=3).save_as('item_city_id_t3')
    OnehotFeature.get_raw_feature(name='item_city_id', threshold=5).save_as('item_city_id_t5')
    OnehotFeature.get_raw_feature(name='item_city_id', threshold=10).save_as('item_city_id_t10')
    # add num format
    OnehotFeature.get_raw_feature(name='item_price_level').save()
    OnehotFeature.get_raw_feature(name='item_sales_level').save()
    OnehotFeature.get_raw_feature(name='item_collected_level').save()
    OnehotFeature.get_raw_feature(name='item_pv_level').save()

    # user-7
    # OnehotFeature.get_raw_feature(name='user_id_6w', threshold=3, raw_name='user_id').save()
    # OnehotFeature.get_raw_feature(name='user_id_3w', threshold=4, raw_name='user_id').save()
    # OnehotFeature.get_raw_feature(name='user_id_1w', threshold=7, raw_name='user_id').save()
    OnehotFeature.get_raw_feature(name='user_gender_id', threshold=0).save()
    OnehotFeature.get_raw_feature(name='user_occupation_id', threshold=0).save()
    # add num format    
    OnehotFeature.get_raw_feature(name='user_age_level').save()
    OnehotFeature.get_raw_feature(name='user_star_level').save()

    # context-4
    # OnehotFeature.get_raw_feature(name='context_id', threshold=0).save()
    OnehotFeature.get_raw_feature(name='context_page_id', threshold=0).save()

    # shop-7
    OnehotFeature.get_raw_feature(name='shop_id').save()
    OnehotFeature.get_raw_feature(name='shop_id', threshold=3).save_as('shop_id_t3')
    OnehotFeature.get_raw_feature(name='shop_id', threshold=5).save_as('shop_id_t5')
    OnehotFeature.get_raw_feature(name='shop_id', threshold=10).save_as('shop_id_t10')
    NumericalFeature.get_raw_feature(name='shop_review_positive_rate').save()
    # add num format
    OnehotFeature.get_raw_feature(name='shop_review_num_level').save()
    OnehotFeature.get_raw_feature(name='shop_star_level').save()
    NumericalFeature.get_raw_feature(name='shop_score_service').save()
    NumericalFeature.get_raw_feature(name='shop_score_delivery').save()
    NumericalFeature.get_raw_feature(name='shop_score_description').save()

    # instance_id && label
    CountingFeature.get_raw_feature(name='instance_id').save()
    NumericalFeature.get_raw_feature(name='is_trade').save()

    # predict_category_property
    # TODO
    # predict_category_property = []
    # for line in joblib.load(os.path.join(default_data_path, 'pretreatment/raw-predict_category_property.pkl')):
    #     tmp_list = []
    #     for x in line.split(';'):
    #         split_category_property = re.split(r'[:,]', x)
    #         category = split_category_property[0]
    #         for i, property in enumerate(split_category_property):
    #             if i != 0:
    #                 tmp_list.append((category, property))
    #     predict_category_property.append(tmp_list)
    # MultihotFeature('predict_category_property', predict_category_property, threshold=3).save()

def get_time_feature():
    # timestamp
    sec_list, min_list, hour_list, day_list = [], [], [], []
    timestamp_data = CountingFeature.get_raw_feature(name='context_timestamp').data
    for timestamp in timestamp_data:
        time_local = time.localtime(timestamp)
        sec_list.append(time_local.tm_sec)
        min_list.append(time_local.tm_min)
        hour_list.append(time_local.tm_hour)
        day_list.append(time_local.tm_wday)
    OnehotFeature(name='context_timestamp_sec', data=sec_list).save()
    OnehotFeature(name='context_timestamp_min', data=min_list).save()
    OnehotFeature(name='context_timestamp_hour', data=hour_list).save()
    OnehotFeature(name='context_timestamp_day', data=day_list).save()


def get_num_feature():
    def cat2num(name, norm=False):
        xx = joblib.load(os.path.join(default_data_path, 'pretreatment/raw-{0}.pkl'.format(name)))
        xx = np.array([float(x) for x in xx])
        if norm:
            _min = xx[xx > -1].min()
            _max = xx.max()
            def foo(x):
                return -1 if x == -1 else ((x - _min) / (_max - _min))
            xx = np.array([foo(x) for x in xx])
        NumericalFeature(name=name + '_num' + ('_norm' if norm else ''), data=xx).save()

    cat2num('item_price_level')
    cat2num('item_sales_level')
    cat2num('item_collected_level')
    cat2num('item_pv_level')
    cat2num('user_age_level')
    cat2num('user_star_level')
    cat2num('shop_review_num_level')
    cat2num('shop_star_level')
    cat2num('item_price_level', True)
    cat2num('item_sales_level', True)
    cat2num('item_collected_level', True)
    cat2num('item_pv_level', True)
    cat2num('user_age_level', True)
    cat2num('user_star_level', True)
    cat2num('shop_review_num_level', True)
    cat2num('shop_star_level', True)


def get_cat_feature():
    # shop_review_positive_rate 50%: 1 9: -1 others: log(1-x) ~normal
    # shop_service_score 0.27%: 1 60: -1 others: ~normal
    # shop_delivery_score 0.22%: 1 60: -1 others: ~normal
    # shop_description_score 0.3%: 1 60: -1 others: ~normal

    def num2cat(name, foo, bar, suffix):
        xx = joblib.load(os.path.join(default_data_path, 'pretreatment/raw-{0}.pkl'.format(name)))
        xx = np.array([float(x) for x in xx])
        xx_min, xx_max = foo(xx)
        xx_cat = np.array([bar(x, xx_min, xx_max) for x in xx])
        OnehotFeature(name=name + suffix, data=xx_cat).save()

    def foo(x):
        _ = np.log(1-x[(x > 0) & (x < 1)])
        return _.min(), _.max()
    def bar(x, _min, _max):
        return -1 if x == -1 else (101 if x == 1 else int(np.floor((np.log(1-x) - _min) / (_max - _min) * 100)))
    num2cat('shop_review_positive_rate', foo, bar, '_cat100')

    def foo(x):
        _ = x[x != -1]
        return _.min(), _.max()
    def bar(x, _min, _max):
        return -1 if x == -1 else int(np.floor((x - _min) / (_max - _min) * 100))
    num2cat('shop_score_service', foo, bar, '_cat100')
    num2cat('shop_score_delivery', foo, bar, '_cat100')
    num2cat('shop_score_description', foo, bar, '_cat100')

def get_item_prop_list():
    if not os.path.exists(os.path.join(default_data_path,
                                       'pretreatment/raw-{0}.pkl'.format('item_property_list_sorted'))):
        raw_data = joblib.load(os.path.join(default_data_path, 'pretreatment/raw-{0}.pkl'.format('item_property_list')))
        prop_freq = {}
        for row in raw_data:
            props = row.strip().split(';')
            props = [int(x) for x in props]
            for prop in props:
                if prop in prop_freq:
                    prop_freq[prop] += 1
                else:
                    prop_freq[prop] = 1
        print('# properties:', len(prop_freq), 'max appearance:', np.max(list(prop_freq.values())))
        item_prop_list_sorted = []
        for row in raw_data:
            props = row.strip().split(';')
            props = [int(x) for x in props]
            props = sorted(props, key=lambda x: -prop_freq[x])
            # row = ';'.join(map(str, props))
            item_prop_list_sorted.append(props)
        joblib.dump(item_prop_list_sorted, os.path.join(default_data_path, 'pretreatment/raw-{0}.pkl'.
                                                        format('item_property_list_sorted')))
    else:
        item_prop_list_sorted = joblib.load(os.path.join(default_data_path,
                                                         'pretreatment/raw-{0}.pkl'.format('item_property_list_sorted')))
    # max length = 100
    def foo(top=100):
        item_prop_list_top = []
        top_prop_freq = {}
        for props in item_prop_list_sorted:
            props = props[:top]
            item_prop_list_top.append(props)
            for prop in props:
                if prop in top_prop_freq:
                    top_prop_freq[prop] += 1
                else:
                    top_prop_freq[prop] = 1
        print('# properties (top %d):' % top, len(top_prop_freq))
        return item_prop_list_top, top_prop_freq

    prop_list_top10, top10_prop_freq = foo(top=10)
    prop_list_top15, top15_prop_freq = foo(top=15)
    prop_list_top20, top20_prop_freq = foo(top=20)

    MultihotFeature(name='item_property_list_top10_t3', data=prop_list_top10, threshold=3).save()
    MultihotFeature(name='item_property_list_top10_t5', data=prop_list_top10, threshold=5).save()
    MultihotFeature(name='item_property_list_top10_t10', data=prop_list_top10, threshold=10).save()
    MultihotFeature(name='item_property_list_top15_t3', data=prop_list_top15, threshold=3).save()
    MultihotFeature(name='item_property_list_top15_t5', data=prop_list_top15, threshold=5).save()
    MultihotFeature(name='item_property_list_top15_t10', data=prop_list_top15, threshold=10).save()
    MultihotFeature(name='item_property_list_top20_t3', data=prop_list_top20, threshold=3).save()
    MultihotFeature(name='item_property_list_top20_t5', data=prop_list_top20, threshold=5).save()
    MultihotFeature(name='item_property_list_top20_t10', data=prop_list_top20, threshold=10).save()

def get_pred_cat_prop():
    if not os.path.exists(os.path.join(default_data_path,
                                       'pretreatment/raw-{0}.pkl'.format('predict_category_property_sorted'))):
        raw_data = joblib.load(os.path.join(default_data_path, 'pretreatment/raw-{0}.pkl'.format('predict_category_property')))
        cat_prop_pairs = []
        cat_prop_freq = {}
        for row in raw_data:
            row = row.strip()
            if row == '-1':
                cat_prop_pairs.append([])
                continue
            cat_props = row.strip().split(';')
            tmp_list = []
            for cat_prop in cat_props:
                cat, props = cat_prop.split(':')
                props = props.split(',')
                for prop in props:
                    if prop != '-1':
                        cp = cat + ':' + prop
                        tmp_list.append(cp)
                        if cp in cat_prop_freq:
                            cat_prop_freq[cp] += 1
                        else:
                            cat_prop_freq[cp] = 1
            cat_prop_pairs.append(tmp_list)
        cat_prop_sorted = []
        for cps in cat_prop_pairs:
            cps = sorted(cps, key=lambda x: -cat_prop_freq[x])
            cat_prop_sorted.append(cps)
        joblib.dump(cat_prop_sorted, os.path.join(default_data_path,
                                                  'pretreatment/raw-{0}.pkl'.format('predict_category_property_sorted')))
    else:
        cat_prop_sorted = joblib.load(os.path.join(default_data_path,
                                                   'pretreatment/raw-{0}.pkl'.format('predict_category_property_sorted')))

    MultihotFeature(name='predicted_category_property_t3', data=cat_prop_sorted, threshold=3).save()
    MultihotFeature(name='predicted_category_property_t5', data=cat_prop_sorted, threshold=5).save()
    MultihotFeature(name='predicted_category_property_t10', data=cat_prop_sorted, threshold=10).save()


def get_psedo_label():
    cat_fields = ['item_id', 'item_brand_id', 'item_city_id', 'user_id', 'user_gender_id', 'user_occupation_id',
                  # 'context_id',
                  'context_page_id', 'shop_id']
    set_fields = ['item_category_list', 'item_property_list_sorted']
    cross_fields = ['predict_category_property_sorted']
    count_fields = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_age_level',
                    'user_star_level', 'shop_review_num_level', 'shop_star_level']
    cat_fields += count_fields
    set_fields += cross_fields
    num_fields = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
    time_fields = ['context_timestamp']

    def load_feature(feature_name):
        return joblib.load(os.path.join(default_data_path, 'pretreatment/raw-{0}.pkl'.format(feature_name)))

    train_size = 478138
    n_folds = np.arange(train_size)
    np.random.shuffle(n_folds)
    n_folds = np.split(n_folds, [(train_size // 10) * i for i in range(1, 10)])
    id2fold = {}
    for i, fold_i in enumerate(n_folds):
        for _id in fold_i:
            id2fold[_id] = i
    is_trade = load_feature('is_trade')

    def cat2pseudo(field_name, feat_type='cat', top=100, mode='nfold'):
        features = load_feature(field_name)
        n_feat_total = []
        n_feat_pos = []
        for fold_i in n_folds:
            feat_total_i = {}
            feat_pos_i = {}
            for i in fold_i:
                f = features[i]
                if feat_type == 'cat':
                    f = int(f)
                    feat_total_i[f] = feat_total_i.get(f, 0) + 1
                    feat_pos_i[f] = feat_pos_i.get(f, 0) + (1 if is_trade[i] == '1' else 0)
                elif feat_type == 'set':
                    if type(f) is str:
                        f = list(int(x) for x in f.strip().split(';'))
                    else:
                        f = f[:top]
                    for ff in f:
                        feat_total_i[ff] = feat_total_i.get(ff, 0) + 1
                        feat_pos_i[ff] = feat_pos_i.get(ff, 0) + (1 if is_trade[i] == '1' else 0)
            n_feat_total.append(feat_total_i)
            n_feat_pos.append(feat_pos_i)
        feat_total = {}
        feat_pos = {}
        for i in range(10):
            feat_total_i = n_feat_total[i]
            feat_pos_i = n_feat_pos[i]
            for k in feat_total_i.keys():
                if k in feat_total:
                    feat_total[k] += feat_total_i[k]
                    feat_pos[k] += feat_pos_i[k]
                else:
                    feat_total[k] = feat_total_i[k]
                    feat_pos[k] = feat_pos_i[k]
        pseudo_features = []
        for i, f in enumerate(features):
            if i < train_size:
                if mode == 'nfold':
                    fold_i = id2fold[i]
                    if feat_type == 'cat':
                        f = int(f)
                        pos_i = (feat_pos.get(f, 0) - n_feat_pos[fold_i].get(f, 0))
                        total_i = (feat_total.get(f, 0) - n_feat_total[fold_i].get(f, 0))
                        pseudo_features.append(pos_i / total_i if total_i else -1)
                    elif feat_type == 'set':
                        if type(f) is str:
                            f = list(int(x) for x in f.strip().split(';'))
                        else:
                            f = f[:top]
                        tmp = []
                        for ff in f:
                            pos_i = (feat_pos.get(ff, 0) - n_feat_pos[fold_i].get(ff, 0))
                            total_i = (feat_total.get(ff, 0) - n_feat_total[fold_i].get(ff, 0)) or 1
                            if total_i:
                                tmp.append(pos_i / total_i)
                        pseudo_features.append(np.mean(tmp) if len(tmp) else -1)
                elif mode == '1out':
                    if feat_type == 'cat':
                        f = int(f)
                        pos_i = feat_pos.get(f, 0) - (1 if is_trade[i] == '1' else 0)
                        total_i = feat_total.get(f, 0) - 1
                        pseudo_features.append(pos_i / total_i if total_i else -1)
                    elif feat_type == 'set':
                        if type(f) is str:
                            f = list(int(x) for x in f.strip().split(';'))
                        else:
                            f = f[:top]
                        tmp = []
                        for ff in f:
                            pos_i = feat_pos.get(ff, 0) - (1 if is_trade[i] == '1' else 0)
                            total_i = feat_pos.get(ff, 0) - 1
                            if total_i:
                                tmp.append(pos_i / total_i)
                        pseudo_features.append(np.mean(tmp) if len(tmp) else -1)
            else:
                if feat_type == 'cat':
                    f = int(f)
                    pos_i = feat_pos.get(f, 0)
                    total_i = feat_total.get(f, 0)
                    pseudo_features.append(pos_i / total_i if total_i else -1)
                elif feat_type == 'set':
                    if type(f) is str:
                        f = list(int(x) for x in f.strip().split(';'))
                    else:
                        f = f[:top]
                    tmp = []
                    for ff in f:
                        pos_i = feat_pos.get(ff, 0)
                        total_i = feat_total.get(ff, 0)
                        if total_i:
                            tmp.append(pos_i / total_i)
                    pseudo_features.append(np.mean(tmp) if len(tmp) else -1)
        pseudo_features = np.array(pseudo_features)
        print(field_name, len(pseudo_features[pseudo_features == -1]) / len(pseudo_features),
              np.min(pseudo_features[pseudo_features > -1]), np.max(pseudo_features), np.mean(pseudo_features[pseudo_features > -1]))
        # plt.hist(pseudo_features)
        # plt.show()
        NumericalFeature(name=field_name + ('_top%d' % top if top < 100 else '') + '_pseudo' +
                              ('_1out' if mode == '1out' else ''), data=pseudo_features).save()

    for cat in cat_fields:
        cat2pseudo(cat, mode='nfold')
    cat2pseudo(set_fields[0], feat_type='set', mode='nfold')
    cat2pseudo(set_fields[1], feat_type='set', top=10, mode='nfold')
    cat2pseudo(set_fields[1], feat_type='set', top=15, mode='nfold')
    cat2pseudo(set_fields[1], feat_type='set', top=20, mode='nfold')
    cat2pseudo(set_fields[2], feat_type='set', mode='nfold')

    for cat in cat_fields:
        cat2pseudo(cat, mode='1out')
    cat2pseudo(set_fields[0], feat_type='set', mode='1out')
    cat2pseudo(set_fields[1], feat_type='set', top=10, mode='1out')
    cat2pseudo(set_fields[1], feat_type='set', top=15, mode='1out')
    cat2pseudo(set_fields[1], feat_type='set', top=20, mode='1out')
    cat2pseudo(set_fields[2], feat_type='set', mode='1out')


def handle_raw_feature():
    # __pretreatment()
    # with(open(os.path.join(os.path.dirname(__file__), 'all_features'), 'w')) as f:
    #     print('#all fearture names\n', file=f)
    # get_basic_feature()
    # get_time_feature()
    # get_num_feature()
    # get_cat_feature()
    # get_item_prop_list()
    # get_pred_cat_prop()
    # get_psedo_label()
    # user_clustering(10)
    # user_clustering(100)
    # user_cat_view_num()
    # cluster_user_cat_view()
    feature_combination()
    basic_analysis(open(os.path.join(os.path.dirname(__file__), 'all_features')).read().splitlines())


if __name__ == '__main__':
    handle_raw_feature()
