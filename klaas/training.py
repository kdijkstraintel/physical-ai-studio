from getiaction import Trainer
from getiaction.data import LeRobotDataModule
from getiaction.policies import SmolVLA
from lightning.pytorch.callbacks import ModelCheckpoint
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import os

def create_model(dataset_stats):
    model = SmolVLA(load_vlm_weights=True, dataset_stats=dataset_stats)
    return model

def create_datamodule(n_action_steps, fps=30, batch_size=64):
    delta = [i/fps for i in range(n_action_steps)]

    hf_dataset = LeRobotDataset(root="/home/kdijkstr/datasets/place1", repo_id="")
    datamodule = LeRobotDataModule(dataset=hf_dataset,
                                   train_batch_size=batch_size,
                                   # data_format="lerobot",
                                   data_format="getiaction",
                                   delta_timestamps={"action": delta})
    return datamodule

def create_trainer(max_epochs, exp_name, init_only=False):
    save_freq = min(max_epochs, 1)

    save_checkpoint_cb = ModelCheckpoint(dirpath="./experiments/checkpoints",
                                         filename="epoch-{epoch}",
                                         save_top_k=1,
                                         every_n_epochs=save_freq,
                                         verbose=True,)

    trainer = Trainer(max_epochs=max_epochs,
                      accelerator="gpu",
                      devices=1,
                      precision=32,
                      enable_checkpointing=True,
                      check_val_every_n_epoch=1,
                      fast_dev_run=init_only,
                      experiment_name=exp_name,
                      callbacks=[save_checkpoint_cb],
                      )
    return trainer


def main():
    epochs=10000
    datamodule = create_datamodule(n_action_steps=50)
    model = create_model(dataset_stats=datamodule.train_dataset.stats)
    trainer = create_trainer(max_epochs=epochs, exp_name="smolvla_first")
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # SmolVLA loads some weights on other GPUs
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()