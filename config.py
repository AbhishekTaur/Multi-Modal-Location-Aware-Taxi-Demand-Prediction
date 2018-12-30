from sklearn.preprocessing import LabelBinarizer


def binary_encoder(y):
    encoder = LabelBinarizer()
    encoder.fit(y)
    return encoder


max_count = 50
p_zones = [97, 25, 181, 189]
years = ['2017', '2018']
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
processed_taxi_data_root = 'processed_taxi_data'
encoder = dict(
    weekday=binary_encoder(range(1, 8)),
    day=binary_encoder(range(1, 32)),
    month=binary_encoder(months),
    hour=binary_encoder(range(24)),
    minute=binary_encoder(range(2))
)
pzone_encoder = binary_encoder(p_zones)
exp_logs_root = "./exp_logs/"
