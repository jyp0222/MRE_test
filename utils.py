import numpy as np

def split_types(types):
    filtered_types = ["Other", "None", "none", "NA"]
    types = [ele for ele in types if ele not in filtered_types]
    np.random.shuffle(types)
    n_types = len(types)
    avg_n_types = n_types // 2
    base_types = types[:avg_n_types]
    novel_types = types[avg_n_types:]
    return base_types, novel_types
