# - Increase the embeddings factos
# - Decrease the batch size
# - Add Batch Normalization
# - Try LSTM, Bidirectional RNN, stack RNN
# - Try with more dense layers or more rnn outputs
# -  etc. Or even try a new architecture!
import math
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Read in data set
train = pd.read_table("./data/train.tsv")
test = pd.read_table("./data/test.tsv")
print(train.shape)
print(test.shape)
train.head(2)
test.head(2)
submission = pd.read_csv("./data/sample_submission.csv")
submission.head(3)


# Data imputation
def dat_impute(dat):
    for column in dat:
        dat[column].fillna(value="missing", inplace=True)


dat_impute(train)
dat_impute(test)
train.head(2)
test.dtypes


# Categorical data encoding
def dat_encoder(train, test):
    for column in ['category_name', 'brand_name']:
        le = LabelEncoder()
        le.fit(np.hstack([train[column], test[column]]))
        train[column] = le.transform(train[column])
        test[column] = le.transform(test[column])
        del le


dat_encoder(train, test)

# Feature engineering - Raw text features
from keras.preprocessing.text import Tokenizer

raw_text = np.hstack([train.item_description.str.lower(), train.name.str.lower()])
raw_text

# Tokenizer
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)
tok_raw

train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
train.head(3)

# SEQUENCES VARIABLES ANALYSIS
max_name_seq = np.max([np.max(train.seq_name.apply(lambda x: len(x))), np.max(test.seq_name.apply(lambda x: len(x)))])
max_seq_item_description = np.max([np.max(train.seq_item_description.apply(lambda x: len(x)))
                                      , np.max(test.seq_item_description.apply(lambda x: len(x)))])
print("max name seq " + str(max_name_seq))
print("max item desc seq " + str(max_seq_item_description))
train.seq_name.apply(lambda x: len(x)).hist()
train.seq_item_description.apply(lambda x: len(x)).hist()

# EMBEDDINGS MAX VALUE
# Base on the histograms, we select the next lengths
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_TEXT = np.max([np.max(train.seq_name.max())
                      , np.max(test.seq_name.max())
                      , np.max(train.seq_item_description.max())
                      , np.max(test.seq_item_description.max())]) + 2
MAX_CATEGORY = np.max([train.category_name.max(), test.category_name.max()]) + 1
MAX_BRAND = np.max([train.brand_name.max(), test.brand_name.max()]) + 1
MAX_CONDITION = np.max([train.item_condition_id.max(), test.item_condition_id.max()]) + 1

# SCALE target variable
train["target"] = np.log(train.price + 1)
target_scaler = MinMaxScaler(feature_range=(-1, 1))
train["target"] = target_scaler.fit_transform(train.target.reshape(-1, 1))
pd.DataFrame(train.target).hist()

# EXTRACT DEVELOPTMENT TEST
dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.99)
print(dtrain.shape)
print(dvalid.shape)

# KERAS DATA DEFINITION
from keras.preprocessing.sequence import pad_sequences


def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        , 'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ)
        , 'brand_name': np.array(dataset.brand_name)
        , 'category_name': np.array(dataset.category_name)
        , 'item_condition': np.array(dataset.item_condition_id)
        , 'num_vars': np.array(dataset[["shipping"]])
    }
    return X


X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(test)

# KERAS MODEL DEFINITION
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def get_model():
    # params
    dr_r = 0.1

    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_name = Embedding(MAX_TEXT, 50)(name)
    emb_item_desc = Embedding(MAX_TEXT, 50)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)

    # rnn layer
    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_name)

    # main layer
    main_l = concatenate([
        Flatten()(emb_brand_name)
        , Flatten()(emb_category_name)
        , Flatten()(emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , num_vars
    ])
    main_l = Dropout(dr_r)(Dense(128)(main_l))
    main_l = Dropout(dr_r)(Dense(64)(main_l))

    # output
    output = Dense(1, activation="linear")(main_l)

    # model
    model = Model([name, item_desc, brand_name
                      , category_name, item_condition, num_vars], output)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", rmsle_cust])

    return model


model = get_model()
model.summary()

# FITTING THE MODEL
BATCH_SIZE = 20000
epochs = 5

model = get_model()
model.fit(X_train, dtrain.target, epochs=epochs, batch_size=BATCH_SIZE
          , validation_data=(X_valid, dvalid.target)
          , verbose=1)

# EVLUEATE THE MODEL ON DEV TEST: What is it doing?
val_preds = model.predict(X_valid)
val_preds = target_scaler.inverse_transform(val_preds)
val_preds = np.exp(val_preds) + 1


# mean_absolute_error, mean_squared_log_error
def rmsle(y, y_pred):
    # Source: https://www.kaggle.com/marknagelberg/rmsle-function
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0 / len(y))) ** 0.5


y_true = np.array(dvalid.price.values)
y_pred = val_preds[:, 0]
v_rmsle = rmsle(y_true, y_pred)
print(" RMSLE error on dev test: " + str(v_rmsle))

# CREATE PREDICTIONS
preds = model.predict(X_test, batch_size=BATCH_SIZE)
preds = target_scaler.inverse_transform(preds)
preds = np.exp(preds) - 1

submission = test[["test_id"]]
submission["price"] = preds

submission.to_csv("./myNNsubmission.csv", index=False)
submission.price.hist()
