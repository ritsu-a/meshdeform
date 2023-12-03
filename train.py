from dataset.dataset import NoisyPcdDataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from model.G3DCODED import AE_AtlasNet_Humans
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
import os
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt

from utils.mesh_process import tetrahedralize, load_mesh, compute_tetrahedron_centroids, save_pointcloud, save_mesh, plt_mesh, plt_meshes, plt_mesh2, plt_mesh3



from pytorch3d.loss import chamfer_distance


save_dir = "./outputs/1204/"
os.makedirs("./outputs/", exist_ok =True)
os.makedirs(save_dir, exist_ok =True)
os.makedirs(save_dir + "/video", exist_ok = True)
os.makedirs(save_dir + "/imgs", exist_ok = True)
os.makedirs(save_dir + "/pointcloud", exist_ok = True)
os.makedirs(save_dir + "/ply", exist_ok = True)
os.makedirs(save_dir + "/png", exist_ok = True)


def worker_init_fn(worker_id, base_seed=None):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)

class IterationBasedBatchSampler(Sampler):
    """Wraps a BatchSampler.

    Resampling from it until a specified number of iterations have been sampled

    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter







net = AE_AtlasNet_Humans(num_points=200)
net = net.cuda()
max_iter = 3000

lr = 0.001
params = net.parameters()
optimizer = torch.optim.Adam(params, lr=lr)


# TODO: use cfg for hyperparameters
dataset_train = NoisyPcdDataset(mode="train")
print(f"Length of training dataset: {len(dataset_train)}")

sampler = RandomSampler(dataset_train, replacement=False)
batch_sampler = BatchSampler(sampler, batch_size=8, drop_last=True)
batch_sampler = IterationBasedBatchSampler(
                batch_sampler, num_iterations=max_iter, start_iter=0
            )
traindataloader = iter(
    DataLoader(
        dataset_train,
        batch_sampler=batch_sampler,
        num_workers=1,
        worker_init_fn=lambda worker_id: worker_init_fn(
            worker_id, base_seed=1
        ),
    )
)




# Starting training

tic = time.time()


losses = []
errors = []

mse = torch.nn.MSELoss() # Mean squared error

for iteration in tqdm(range(max_iter)):
    cur_iter = iteration + 1
    time_dict = {}
    optimizer.zero_grad()
    
    
    loss = 0
    data_batch = next(traindataloader)

    data_time = time.time() - tic
    time_dict["time_data"] = data_time

    # Copy data from cpu to gpu
    data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
    # Forward
    predict = net(data_batch)
    chamfer_loss, _ = chamfer_distance(predict, data_batch["verts"])
    loss += chamfer_loss
    losses.append(loss.item())


    with torch.no_grad():
        mse_error = mse(data_batch["verts"], predict)
        errors.append(mse_error.item())

    loss.backward()


## Saving loss and evaluation

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Total_Loss')
plt.title('Loss Decrease Over Time')
plt.savefig(os.path.join(save_dir, 'total_loss.png'), dpi=300)
plt.close()


plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('MSE Error for Verts')
plt.title('Error')
plt.savefig(os.path.join(save_dir, 'mse_error.png'), dpi=300)
plt.close()


for mode in ["test", "valid"]:
    dataset_test = NoisyPcdDataset(mode=mode)
    test_data_loader = DataLoader(
                dataset_test,
                batch_size=1,
                num_workers=1,
                shuffle=False,
                drop_last=False,
                pin_memory=False,
            )
    print(len(test_data_loader))
    imgs = []
    with torch.no_grad():
        net.eval()
        for iteration, data_batch in enumerate(test_data_loader):
            curr_dict = {}
            cur_iter = iteration + 1
            
            data_batch_cuda = {
                k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)
            }
            # Forward
            predict = net(data_batch_cuda).squeeze(0).detach().cpu()
            verts = data_batch["verts"].squeeze(0)
            noisy_pcd = data_batch["noisy_pcd"].squeeze(0)

            faces = data_batch["faces"]
            plt_mesh3(verts, noisy_pcd, predict, faces, save_dir + f"imgs/{mode}_{iteration}.png")
            imgs.append(save_dir + f"imgs/{mode}_{iteration}.png")
            print(data_batch["idx"], data_batch["file_id"], iteration)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, channels = cv2.imread(save_dir + "imgs/{mode}_0.png").shape
    frame_size = (width, height)
    out = cv2.VideoWriter(save_dir + "video/{mode}.mp4", fourcc, 5, frame_size)


    for img in imgs:
        cv_dst = cv2.imread(img)
        out.write(cv_dst)