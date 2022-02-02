import os
import pandas as pd

def create_DataSet():
    file_list = os.listdir("./DataSet/")
    data_list = []
    for i in file_list:
        data_list.append(pd.read_csv("./DataSet/{}".format(i)))

    data = pd.concat(data_list)
    
    # # -------------------------------------------------------------
    # # ou pour enregistrer un nouveau csv :
    # data.to_csv( "data.csv", index=False, encoding='utf-8-sig')
    # # -------------------------------------------------------------

    return data