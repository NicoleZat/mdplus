from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def get_data(input_file):
    index = -1
    X = []
    y =[]
    features = ['age', 'amonth', 'aweekend', 'elective', 'female', 'hcup_ed', 'hosp_division', 'hosp_nis', 'i10_ndx', 'i10_npr', 'los', 'mdc', 'mdc_nopoa', 'nis_stratum', 'pay1', 'pl_nchs', 'race', 'totchg', 'tran_in', 'zipinc_qrtl']
    with open(input_file) as fo:
        for line in fo:
            index +=1
            if index == 0:
                header_index = []
                header = line[:-1].split(',')
                for i in range(len(header)):
                    if header[i] in features:
                        header_index.append(i)
            else:
                tempX = []
                split_line = line[:-1].split(',')
                for i in header_index:
                    try:
                        tempX.append(float(split_line[i]))
                    except:
                        tempX.append(0)
                y.append(split_line[4])
                X.append(tempX)
    out_headers = []
    for i in header_index:
        out_headers.append(header[i])
    print('**************')
    print(index)
    return out_headers,X,y

def feature_select(headers, X, y, num_features_to_select):
    top_features = []
    selector = RFE(RandomForestClassifier(), n_features_to_select=num_features_to_select)
    selector = selector.fit(X, y)
    rank = selector.ranking_
    for i in range(len(rank)):
        if rank[i] == 1:
            top_features.append(headers[i])
    return top_features

headers,X,y = get_data('NIS_2019_Core.csv')
print(feature_select(headers, X, y, 5))

headers2,X2,y2 = get_data('NIS_2018_Core.csv')
print(feature_select(headers2, X2, y2, 5))

headers3,X3,y3 = get_data('NIS_2017_Core.csv')
print(feature_select(headers3, X3, y3, 5))

headers4,X4,y4 = get_data('NIS_2016_Core.csv')
print(feature_select(headers4, X4, y4, 5))
