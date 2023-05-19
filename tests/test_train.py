from model.train import get_csvs_df
from model.train import train_model
import os
import pytest
import numpy as np


def test_csvs_no_files():
    with pytest.raises(RuntimeError) as error:
        get_csvs_df("./")
    assert error.match("No CSV files found in provided data")


def test_csvs_no_files_invalid_path():
    with pytest.raises(RuntimeError) as error:
        get_csvs_df("/invalid/path/does/not/exist/")
    assert error.match("Cannot use non-existent path provided")


def test_csvs_creates_dataframe():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    datasets_directory = os.path.join(current_directory, 'datasets')
    result = get_csvs_df(datasets_directory)
    assert len(result) == 20

def test_train_model():
    xtrain = np.array([[8,171,42,29,160,35.48224692,0.082671083,22],
                       [3,108,63,45,297,49.37516891,0.100979095,46]]).reshape(2,-1)
    ytrain = np.array([1,0])
    xtest = np.array([[8,171,42,29,160,35.48224692,0.082671083,22],
                       [3,108,63,45,297,49.37516891,0.100979095,46]]).reshape(2,-1)
    ytest = np.array([1,0])
    
    print(xtrain.shape,ytrain.shape)
    mymodel = train_model(0.01,xtrain,xtest,ytrain,ytest)
    preds = mymodel.predict([[8,171,42,29,160,35.48224692,0.082671083,22]])
    print(preds)
    np.testing.assert_almost_equal(preds,[1])

