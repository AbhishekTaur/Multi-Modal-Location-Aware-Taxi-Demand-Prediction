from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, concatenate, LSTM, Reshape, Embedding

from config import max_count

models = ["MLP", "LSTM"]


def build_model(num_lags, model="MLP", summary=True, date_time=False, combined_model=False, event_info=False):
    assert model in models
    input_lags = Input(shape=(num_lags,), name="pickup_lags")
    inputs = []
    concat = []
    x = input_lags
    if event_info:
        input_events = Input(shape=(max_count,), name="events")
        events_feat = Embedding(max_count, 8)(input_events)
        events_feat = LSTM(16)(events_feat)
        inputs.append(input_events)
        concat.append(events_feat)
    if date_time:
        input_date_time = Input(shape=(75,), name="date_time")
        inputs.append(input_date_time)
        concat.append(input_date_time)
    if combined_model:
        pickup_zone = Input(shape=(4,), name="pickup_zone")
        inputs.append(pickup_zone)
        concat.append(pickup_zone)
    if model.upper() == "MLP":
        # x = concatenate([x] + concat)
        x = MLP(x)
    elif model.upper() == "LSTM":
        x = Reshape((1, 48))(x)
        x = SLSTM(x)

    if len(inputs) > 0:
        x = concatenate([x] + concat)
        # x = Dense(units=10, activation="relu")(x)
        inputs.append(input_lags)
    else:
        inputs = input_lags
    preds = Dense(units=1, name="predicted")(x)
    model = Model(inputs, preds)
    model.compile(loss="mean_squared_error", optimizer="adam")
    if summary:
        print(model.summary())
    return model


def MLP(x):
    x = Dense(units=100, activation="relu")(x)
    return x


def SLSTM(x):
    x = LSTM(units=100, activation='relu')(x)
    return x
