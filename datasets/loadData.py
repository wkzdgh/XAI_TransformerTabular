import numpy as np
import openml
from sklearn.preprocessing import LabelEncoder
import torch
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import random

def data_split(X, y, nan_mask, indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_openml, attribute_names, class_names, cat, cont, y_l_enc, data) -> None:
        self.id = dataset_openml.id
        self.name = dataset_openml.name
        self.openml_url = dataset_openml.openml_url
        self.original_data_url = dataset_openml.original_data_url
        self.attribute_names =  attribute_names
        self.class_names = class_names
        self.dataCat = cat
        self.dataCont = cont
        self.y_encoder = y_l_enc
        
        self.cat = data.cat
        self.cont = data.cont
        self.cat_mask = data.cat_mask
        self.cont_mask = data.cont_mask
        self.y = data.y
        self.num_classes = data.num_classes
        self.cls = data.cls
        self.cls_mask = data.cls_mask
        self.cont_scaler = data.scaler

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return np.concatenate((self.cls[idx], self.cat[idx])), self.cont[idx], self.y[idx], np.concatenate((self.cls_mask[idx], self.cat_mask[idx])), self.cont_mask[idx]


def getDataFromDataset(dataset_openml, task, k=5):
    X, y, categorical_indicator, attribute_names = dataset_openml.get_data(dataset_format="dataframe", target=dataset_openml.default_target_attribute)
    
    # Crear y añadir variables aleatorias (ruido)
    for i in range(0, X.shape[1]):
        X["random"+str(i)] = [random.random() for _ in range(0, X.shape[0])]
        attribute_names.append("random"+str(i))

    attribute_names.insert(0, "[CLS]")
    if task in ["binary", "multiclass"]:
        class_names = list(y.unique())
    else:
        class_names = []
    
    categorical_names = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
    cont_names = list(X.columns[con_idxs])
    cat_encoders = list()
    for col in categorical_names:
        X[col] = X[col].astype("object")

    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)

    cat_dims = []
    for col in categorical_names:
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
        cat_encoders.append(l_enc)
    for col in cont_names:
        X.fillna(X.loc[X.index.values, col].mean(), inplace=True)
        
    cat = list(zip(cat_idxs, categorical_names, cat_dims, cat_encoders))
    cont = list(zip(con_idxs, cont_names))

    y = y.values
    y_l_enc = LabelEncoder()
    if task != 'regression': 
        y = y_l_enc.fit_transform(y)
    num_classes = len(np.unique(y))

    kf = StratifiedKFold(n_splits=k)
    folders_index = {}
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        folders_index[i] = {"train_index":train_index,"test_index": test_index}

    folders = {}
    for _, key in enumerate(folders_index):
        print("Folder " + str(key))
        X_train, y_train = data_split(X, y, nan_mask, folders_index[key]["train_index"])
        X_test, y_test = data_split(X, y, nan_mask, folders_index[key]["test_index"])

        train_test = [(X_train, y_train), (X_test, y_test)]
        datos = list()
        for x_y in train_test:
            x_mask =  x_y[0]['mask'].copy()
            x_data = x_y[0]['data'].copy()

            cat_data = x_data[:,cat_idxs].copy().astype(np.int64) #categorical columns
            cont_data = x_data[:,con_idxs].copy().astype(np.float32) #numerical columns
            cat_mask = x_mask[:,cat_idxs].copy().astype(np.int64) #categorical columns
            cont_mask = x_mask[:,con_idxs].copy().astype(np.int64) #numerical columns

            if task in ['binary', 'multiclass']:
                y_data = x_y[1]['data']#.astype(np.float32)
            elif task == 'regression':
                y_data = x_y[1]['data'].astype(np.float32)
            else:
                print("\n\nERROR - la tarea no entra entre las aceptadas")

            cls = np.zeros_like(y_data, dtype=int)
            cls_mask = np.ones_like(y_data, dtype=int)

            scaler = StandardScaler()
            cont_data = scaler.fit_transform(cont_data)

            data = namedtuple("data", ["cat", "cont", "cat_mask", "cont_mask", "y", "num_classes", "cls", "cls_mask", "scaler"])
            datos.append(data(cat_data, cont_data, cat_mask, cont_mask, y_data, num_classes, cls, cls_mask, scaler))
        
        folders[key] = {"train": datos[0], "test": datos[1]}

    return attribute_names, class_names, cat, cont, y_l_enc, folders


def kfold(id_dataset, seed, task, k=5):
    np.random.seed(seed)
    dataset_openml = openml.datasets.get_dataset(id_dataset)
    attribute_names, class_names, cat, cont, y_l_enc, folders = getDataFromDataset(dataset_openml, task, k)

    for _, key in enumerate(folders):
        datasetTrain = Dataset(dataset_openml, attribute_names, class_names, cat, cont, y_l_enc, folders[key]["train"])
        datasetTest = Dataset(dataset_openml, attribute_names, class_names, cat, cont,  y_l_enc, folders[key]["test"])
        folders[key]["train"] = datasetTrain
        folders[key]["test"] = datasetTest

    return folders




