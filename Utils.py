import _pickle
import numpy as np


def UniqueValues(data: list, col: int) -> tuple:
    value, count = np.unique([_[col].strip()
                              for _ in data], return_counts=True)
    return value.tolist(), count.tolist()


def GroupByPivot(data: list, col: int) -> dict:
    dic = {}
    for item in data:
        if item[col].strip() in dic.keys():
            dic[item[col].strip()].append(item)
        else:
            dic.update({item[col].strip(): [item]})
    return dic


if __name__ == "__main__":

    data = []
    with open('data.pkl', 'rb') as f:
        data = _pickle.load(f)

    # print(UniqueValues(data[1:], 1))
    # print(UniqueValues(data[1:], 2))
    # print(UniqueValues(data[1:], 5))
    # print(UniqueValues(data[1:], 6))
    # print(UniqueValues(data[1:], 7))

    print(GroupByPivot(data[1:], 7))

    pass
