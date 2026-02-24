import os

from lerobot.configs.types import NormalizationMode
from lightning.pytorch.callbacks import ModelCheckpoint

from getiaction import Trainer
from getiaction.data import LeRobotDataModule
from getiaction.policies.lerobot import SmolVLA
from PIL import Image
import torch

def show_losses():
    import seaborn as sbs
    import pandas as pd
    import matplotlib.pyplot as plt
    file = "/home/kdijkstr/projects/geti-action/library/src/getiaction/cli/experiments/lightning_logs/version_43/metrics.csv"
    df = pd.read_csv(file, index_col=0)

    loss_cols = [
        'train/loss',
        'train/losses_after_forward',
        'train/losses_after_rm_padding'
    ]

    df_melted = df.melt(
        id_vars=[c for c in df.columns if c not in loss_cols],
        value_vars=loss_cols,
        var_name='loss_type',
        value_name='loss_value'
    )

    sbs.lineplot(df_melted, x="step", y="loss_value", hue="loss_type")
    plt.show()
    print(df.columns)

def train(init_only, max_epochs):
    model = SmolVLA(load_vlm_weights=True)

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
                      experiment_name="smolvla_xai",
                      callbacks=[save_checkpoint_cb],
                      )
    fps = 30
    # fps = 10
    batch_size = 64
    # batch_size = 8
    delta = [i/fps for i in range(model.hparams["n_action_steps"])]
    datamodule = LeRobotDataModule(repo_id="kdijkstr/place1_small",
                                   train_batch_size=batch_size,
                                   data_format="lerobot",
                                   delta_timestamps={"action": delta})

    trainer.fit(model=model, datamodule=datamodule)
    return model



def create(max_epochs=20):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # SmolVLA loads some weights on other GPUs

    model = train(init_only=False, max_epochs=max_epochs)
    return model

def load(checkpoint = "/home/kdijkstr/projects/geti-action/experiments/checkpoints/epoch-epoch=19.ckpt"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # SmolVLA loads some weights on other GPUs
    model = train(init_only=True, max_epochs=0)
    with torch.serialization.safe_globals([NormalizationMode]):
        state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict["state_dict"])
    return model


def sample(repo_id="kdijkstr/place1_small", n_action_steps=50):
    # model = load()
    fps = 30
    batch_size = 64
    delta = [i/fps for i in range(n_action_steps)]
    datamodule = LeRobotDataModule(repo_id=repo_id,
                                   train_batch_size=batch_size,
                                   data_format="lerobot",
                                   delta_timestamps={"action": delta})

    for batch in datamodule.train_dataloader():
        yield batch

def blend_and_save(batch, explain, folder):
    # scale images.
    mn, mx = batch["observation.images.top"].min(), batch["observation.images.top"].max()
    batch["observation.images.top"] = (batch["observation.images.top"] - mn) / (mx - mn) * 255
    mn, mx = batch["observation.images.wrist"].min(), batch["observation.images.wrist"].max()
    batch["observation.images.wrist"] = (batch["observation.images.wrist"] - mn) / (mx - mn) * 255

    # blend images
    tops = (explain["observation.images.top"] * 0.5 + batch["observation.images.top"].to("cpu") * 0.5).type(torch.uint8)
    wrists = (explain["observation.images.wrist"] * 0.5 + batch["observation.images.wrist"].to("cpu") * 0.5).type(torch.uint8)

    # save images
    for image_id, (top, wrist) in enumerate(zip(tops, wrists)):
        top_arr = top.numpy().transpose(1, 2, 0)
        image = Image.fromarray(top_arr)
        os.makedirs(folder + "/top", exist_ok=True)
        image.save(folder + f"/top/{image_id:05d}.png")

        wrist_arr = wrist.numpy().transpose(1, 2, 0)
        image = Image.fromarray(wrist_arr)
        os.makedirs(folder + "/wrist", exist_ok=True)
        image.save(folder + f"/wrist/{image_id:05d}.png")

def run():
    # model = create()
    model = load().to("cuda")
    for batch_id, batch in enumerate(sample()):
        # to GPU
        batch = {key:value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        action, explain = model.select_action_with_explain(batch)
        folder = f"./experiments/xai/batch_{batch_id}"
        blend_and_save(batch, explain, folder=folder)
        exit(0)


from tools.video import OpenCVCapture, TorchCapture, OverLayCapture, LiveView


def overlay():
    streams = []
    main_crop = [71, 187, 351, 277]
    wrist_crop = (4, 2, 474, 633)

    main_capture = OpenCVCapture(source=2, resolution=(480, 640), fps=25)
    main_video1 = TorchCapture(source="/home/kdijkstr/.cache/huggingface/lerobot/kdijkstr/place1/videos/chunk-000/observation.images.top/episode_000000.mp4", speed=1.0)
    #main_video2 = CropAndResize(TorchCapture(source="/home/kdijkstr/.cache/huggingface/lerobot/kdijkstr/dual_cam_cube1/videos/chunk-000/observation.images.front/episode_000000.mp4", speed=1.0), crop=main_crop, resize=(512, 512))
    main_video = OverLayCapture([main_video1, main_capture])

    overlay = OverLayCapture([main_capture, main_video1], "overlay")
    view = LiveView([overlay])
    view.display_loop()


if __name__ == "__main__":
    run()
