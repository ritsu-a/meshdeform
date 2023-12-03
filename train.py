from dataset.dataset import NoisyPcdDataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler


dataset_train = NoisyPcdDataset(mode="train")

print(len(dataset_train))


# TODO: use cfg for hyperparameters
sampler = RandomSampler(dataset_train, replacement=False)
batch_sampler = BatchSampler(sampler, batch_size=8, drop_last=True)
tune_real_loader = iter(
    DataLoader(
        dataset_train,
        batch_sampler=batch_sampler,
        num_workers=cfg.TUNE.NUM_WORKERS,
        worker_init_fn=lambda worker_id: worker_init_fn(
            worker_id, base_seed=cfg.RNG_SEED if cfg.RNG_SEED >= 0 else None
        ),
    )
)