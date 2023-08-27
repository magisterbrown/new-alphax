from tqdm import tqdm
import time
progress = tqdm(total=100, desc='Loading first batch')
for i in range(100):
    progress.update(1)
    time.sleep(0.2)
    stt = str(progress)
