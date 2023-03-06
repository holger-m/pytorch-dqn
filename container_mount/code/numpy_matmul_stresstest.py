import numpy as np
import time
import torch

cuda = torch.device('cuda')

n = 35000
#M_np = np.random.normal(size=(n, n))
M_np = np.random.normal(size=(n, n)).astype(dtype=np.float32)
M = torch.from_numpy(M_np).cuda()
torch.cuda.synchronize()

start_time_ns = time.time_ns()
N = torch.matmul(M, M)
torch.cuda.synchronize()
end_time_ns = time.time_ns()

diff_time_s = (end_time_ns - start_time_ns)/1000000000
print('Matmul time in sec: ' + str(round(diff_time_s, 1)))
