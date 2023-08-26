import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random
writer = SummaryWriter(log_dir='board_logs')
resus = {k: random.randint(0,3) for k in range(500)}
fig, ax = plt.subplots()
ax.bar(range(len(resus)), resus.values())
writer.add_figure('histt', fig, global_step=0)
writer.flush()
#fig, ax = plt.subplots()
ax.clear()
resus = {k: random.randint(0,255) for k in range(500)}
ax.bar(resus.keys(), resus.values())
writer.add_figure('histt', fig, global_step=1)
writer.flush()
