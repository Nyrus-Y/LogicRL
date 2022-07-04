import matplotlib.pyplot as plt
import csv
import numpy as np

log_file_path = "../training/PPO_logs/CoinJumpEnv-v0/PPO_CoinJumpEnv-v0_log_0.csv"
axis_min = -6

with open(log_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader, None) # skip header
    logs = np.array([[int(row[0]), int(row[1]), float(row[2])] for row in reader])


print(logs.shape)
#plt.plot(logs[::10, -1])
plt.plot(logs[:, 0], logs[:, -1])

mean_n = 50
logsx = np.mean(logs[:(logs.shape[0]//mean_n)*mean_n, -1]
               .reshape((-1, mean_n)), axis=1)
print(logsx.shape, logs[::mean_n, 0].shape)
plt.plot(logs[::mean_n, 0][:-1], logsx)

rmin = np.min(logs[:,-1]) if axis_min is None else axis_min
rmax = np.max(logs[:,-1])
plt.ylim(rmin-0.5, rmax+0.5)
plt.yticks(np.arange(rmin, rmax, 2))
plt.show()
