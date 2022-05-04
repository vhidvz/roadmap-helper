import xlrd
import _pickle


if __name__ == "__main__":
    
    loc = 'MeasurementsCluster.xlsx'

    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)

    data = []
    for i in range(sheet.nrows):
        data.append(sheet.row_values(i))

    with open('data.pkl', 'wb') as f:
        _pickle.dump(data, f)

    pass