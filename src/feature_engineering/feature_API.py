from src.feature_engineering.feature_representation import *
from src.util.feature import *

def fetch_data(id_dict: dict, config_path='feature_config', missing_handler='nothing', del_other=False, multihot_norm=False):
    return __fetch_data(id_dict, config_path, missing_handler, del_other, multihot_norm)

def __fetch_data(id_dict: dict, config_path='feature_config', missing_handler='nothing', del_other=False, multihot_norm=False):
    if not missing_handler in ['nothing', 'mean', 'del']:
        raise NotImplemented('missing_handler can not be {0}'.format(missing_handler))

    feature_config = open(config_path).read().splitlines()
    data = Data(id_dict=id_dict)
    for feature_name in feature_config:
        if feature_name.startswith('#') or feature_name == '':
            continue
        feature = Feature.load(feature_name)
        if feature.type == 'counting' or feature.type == 'numerical':
            data.add_feature_dict(feature.fetch(id_dict, missing_handler=missing_handler))
        elif feature.type == 'one-hot':
            data.add_feature_dict(feature.fetch(id_dict, del_other=del_other))
        elif feature.type == 'multi-hot':
            data.add_feature_dict(feature.fetch(id_dict, del_other=del_other, norm=multihot_norm))
    return data
