from src.feature_engineering.feature_representation import *

feature_dir = 'data/feature'

def html_print(type, fp, feature_list, info_type_list):
    print('<h3> {0} --- {1}</h3>'.format(type, len(feature_list)), file=fp)
    print('<table border="1" style="table-layout: fixed;">', file=fp)

    print('<tr>', file=fp)
    print('<th> name </th>', file=fp)
    for info_type in info_type_list:
        print('<th width=\"100px\">{0}</th>'.format(info_type), file=fp)
    print('</tr>', file=fp)

    for feature in feature_list:
        print('<tr>', file=fp)
        print('<td>{0}</td>'.format(feature.name), file=fp)
        for info_type in info_type_list:
            print('<td>{0}</td>'.format(feature.info[info_type]), file=fp)
        print('</tr>', file=fp)

    print('</table>', file=fp)

def basic_analysis(feature_list):
    dict = {'one-hot':[], 'multi-hot':[], 'counting':[], 'numerical':[]}
    for feature in feature_list:
        if feature.startswith('#') or feature == '':
            continue
        feature = Feature.load(feature)
        dict[feature.type].append(feature)
    with open(os.path.join(os.path.dirname(__file__), 'feature analysis.html'), 'w') as fp:

        html_print('counting', fp, dict['counting'],
                   ['max', 'min', 'mean', 'var', 'missing_rate'])

        html_print('numerical', fp, dict['numerical'],
                   ['max', 'min', 'mean', 'var', 'missing_rate'])

        html_print('one-hot', fp, dict['one-hot'],
                   ['category_num', 'biggest_size', 'threshold', 'tail_size'])

        html_print('multi-hot', fp, dict['multi-hot'],
                   ['category_num', 'biggest_size', 'threshold', 'tail_size'])

if __name__ == '__main__':
    basic_analysis(open(os.path.join(os.path.dirname(__file__), 'all_features')).read().splitlines())