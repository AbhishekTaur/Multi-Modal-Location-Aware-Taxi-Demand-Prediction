from os.path import join

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
# plt.style.use('ggplot')
from config import p_zones
import numpy as np
import matplotlib.pyplot as plt
params = {
   'axes.labelsize': 12,
#    'text.fontsize': 12,
   'legend.fontsize': 10,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   'text.usetex': False,
   'figure.figsize': [7.5, 4.5]
}
matplotlib.rcParams.update(params)

for pickup_zone in p_zones:
    for i in range(48, 220, 192):
        idx = pd.date_range('2017-01-02', '2017-01-09')

        df = pd.read_csv(join("processed_taxi_data", "{}.csv".format(pickup_zone)))["pickup_no"][i:i+336]
        # series = pd.Series(df, index=range(168))
        # series.plot()
        plt.plot(df.tolist())
# plt.xticks(np.arange(0, 168, 1.0))

# plt.xticks(weekdays)
plt.legend(p_zones)
plt.grid()
plt.xlim(0, 336)
plt.xticks(range(24,313,48), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

plt.ylabel("Number of pickups")
plt.xlabel("Day of the week")
# plt.show()
plt.savefig("{}.pdf".format("timeSeries"), format='pdf', dpi=1200)