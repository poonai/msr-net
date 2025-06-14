import matplotlib
from torch import nn
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from torch.nn.utils import parameters_to_vector
import logging
import time
import modal

import math
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import decode_image
from torchvision import transforms
from PIL import Image


image_preprocess = transforms.Compose(
    [
        transforms.ConvertImageDtype(torch.float32),
    ]
)

def get_preprocessed_img(path):
    return image_preprocess(decode_image(path))


class LowLightDataset(Dataset):
    def __init__(self, img_dir="/root/data/data"):
        x_dir = Path(os.path.join(img_dir, "low_light_images_patches"))
        self.low_light_images = [file for file in x_dir.glob("*") if file.is_file()]
        self.img_dir = img_dir
        self.transform = image_preprocess

    def __len__(self):
        return len(self.low_light_images)

    def __getitem__(self, idx):
        low_light_image = self.low_light_images[idx]
        low_light_image_split = low_light_image.name.split("_")
        low_light_image_id = low_light_image_split[0]
        low_light_image_patch_id = low_light_image_split[len(low_light_image_split) - 1]

        high_quality_image = f"{low_light_image_id}_patch_{low_light_image_patch_id}"
        high_quality_image = os.path.join(
            self.img_dir, "hq_images_patches", high_quality_image
        )

        try:
            return get_preprocessed_img(low_light_image),get_preprocessed_img(high_quality_image)
        except Exception as e:
            print(f"unable to decode {low_light_image} {high_quality_image}: {e}")
            raise e


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def log_base(base, x):
    return torch.log(x + 1e-7) / torch.log(torch.tensor(float(base)) + 1e-7)


class LogTransformationConv(nn.Module):
    def __init__(self):
        super(LogTransformationConv, self).__init__()
        ## scales are mentioned in the paper. Doing it exactly as paper says.
        ## Sometime, I'm not but here I'm.
        self.log_transformation_scales = [1, 10, 100, 300]
        self.conv_fuse = nn.Conv2d(
            3 * len(self.log_transformation_scales), 3, kernel_size=1
        )
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        log_transformed = []
        for log_transformation_scale in self.log_transformation_scales:
            log_transformed.append(
                log_base(
                    (log_transformation_scale + 1),
                    (log_transformation_scale + 1) * input,
                )
            )

        concated_log_transformation = torch.cat(log_transformed, dim=1)
        logging.debug(
            f"concated log transformation shape: {concated_log_transformation.size()}",
        )
        fused = self.relu(self.conv_fuse(concated_log_transformation))
        logging.debug(f"fused log transformation shape: {fused.size()}")
        output = self.conv_out(fused)
        logging.debug(f"log transformation layer output: {output.size()}")
        return output


DIFFERENCE_OF_CONVOLUTION_DEPTH = 10


class DifferenceOfConvolution(nn.Module):
    def __init__(self):
        super(DifferenceOfConvolution, self).__init__()
        self.msr_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), nn.ReLU()
                )
                for _ in range(DIFFERENCE_OF_CONVOLUTION_DEPTH)
            ]
        )

        self.avg_conv = nn.Conv2d(3 * DIFFERENCE_OF_CONVOLUTION_DEPTH, 3, kernel_size=1)

    def forward(self, input):
        msr_outputs = []
        for conv in self.msr_convs:
            x1 = input
            if len(msr_outputs) > 1:
                x1 = msr_outputs[-1]
            msr_output = conv(x1)
            msr_outputs.append(msr_output)

        concated_msr = torch.concat(msr_outputs, dim=1)
        logging.debug(f"contacted msr shape: {concated_msr.size()}")

        avg_msr = self.avg_conv(concated_msr)

        logging.debug(f"avg msr shape {avg_msr.size()}")
        # removing the illumination
        return input - avg_msr


class ColorRestoration(nn.Module):
    def __init__(self):
        super(ColorRestoration, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        logging.debug(f"color restoration tensor shape {output.size()}")
        return output


class NormLoss(nn.Module):
    def __init__(self, regularization_param=1e-6):
        super(NormLoss, self).__init__()
        self.regularization_param = regularization_param

    def forward(self, output, target, weights):
        prediction_loss = torch.mean(
            torch.norm((target - output).flatten(start_dim=1), p=2, dim=1) ** 2
        )

        params_vector = parameters_to_vector(weights)
        regularization_loss = (
            self.regularization_param * torch.norm(params_vector, p=2) ** 2
        )
        output = prediction_loss + regularization_loss
        return output


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.log_transformation_layer = LogTransformationConv()
        self.difference_of_convolution = DifferenceOfConvolution()
        self.color_restoration = ColorRestoration()

    def forward(self, input):
       # log_transformed = self.log_transformation_layer(input)
        raw_image = self.difference_of_convolution(input)
        output = self.color_restoration(raw_image)
        return output


def entry_point(batch_size=64, img_dir="/root/remote/data", device=any):
    dataset = LowLightDataset(img_dir=img_dir)

    train_size = len(dataset) * 0.8
    test_size = len(dataset) - int(train_size)
    logging.info(
        f"total sample: {len(dataset)} training size: {train_size} test_size: {test_size} expected batch size: {train_size / batch_size}"
    )
    train_data, test_data = random_split(dataset, [int(train_size), test_size])

    train_loader = DataLoader(
        train_data,
        batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,  # Maintain workers between epochs
        prefetch_factor=2,  # Load batches in advance
    )

    model = Model().to(device)

    train(train_loader, model, device=device)
    
    model.eval()
    
    test_img = img_for_inference("/home/saint/msr-net/data/low_light_images/1_ll_4.png")
    
    output = model.forward(test_img)
    
    save_inference_img(output=output, path = "before.png")
    
    torch.save(model.state_dict(), "model.pt")
    
    model = Model()
    
    model.load_state_dict(torch.load("model.pt"))
    
    model.eval()
    
    output = model.forward(test_img)
    
    save_inference_img(output=output, path = "after.png")
    


def train(
    data_loader: DataLoader,
    model: nn.Module,
    device: any,
    print_every=2,
    learning_rate=0.001,
    n_epoch=1,
):
    print_loss_total = 0
    loss_fn = NormLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start = time.time()
    model.train()

    print("starting training")
    for epoch in range(n_epoch + 1):
        loss = train_epoch(
            data_loader, model, loss_fn, optimizer, device=device, epoch=epoch
        )
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            time_left = 0
            percent = epoch / n_epoch
            if percent != 0:
                time_left = timeSince(start, percent)
            print(
                "%s (%d %d%%) %.4f"
                % (
                    time_left,
                    epoch,
                    epoch / n_epoch * 100,
                    print_loss_avg,
                )
            )


def img_for_inference(path):
    img = get_preprocessed_img(path=path)
    return img.unsqueeze(0)

def save_inference_img(output, path):
    img = (output[0] * 255).squeeze(0).permute(1, 2, 0)
    image = Image.fromarray(img.detach().to(torch.uint8).numpy())
    image.save(path)

def train_epoch(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: any,
    epoch: int,
):
    total_loss = 0
    start_time = time.time()
    print_every = 10
    for batch_idx, data in enumerate(data_loader):
        input, target = data
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        output = model(input)
        loss = loss_fn(output, target, model.parameters())
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % print_every == 0:
            elapsed = time.time() - start_time
            iters_left = len(data_loader) - (batch_idx + 1)
            eta = elapsed / (batch_idx + 1) * iters_left
            avg_loss = total_loss / (batch_idx + 1)
            print(f"avg loss {avg_loss:.4f} ETA {eta / 60:.1f} EPOCH {epoch}")
        

    return total_loss / len(data_loader)


app = modal.App("msr-net")

base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pydantic==2.9.1"
)
torch_image = base_image.pip_install(
    "torch>=2.7.1",
    "tensorboard==2.17.1",
    "numpy<2",
    "torchvision>=0.22.1",
    "matplotlib>=3.10.3",
)

with torch_image.imports():
    import os
    import torch
    from torch.utils.data import DataLoader, random_split
    from torch import nn
    import os
    from torch.nn.utils import parameters_to_vector
    import logging
    import time
    from torch.utils.data import Dataset
    from pathlib import Path
    from torchvision.io import decode_image
    from torchvision import transforms

gpu = "T4"
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

volume = modal.Volume.from_name("msr-net-data", create_if_missing=False)


@app.function(
    image=torch_image,
    gpu=gpu,
    timeout=10 * HOURS,
    volumes={"/root/remote/": volume},
    memory=1500,
)
def modal_entrypoint(batch_size=64, img_dir="/root/remote/data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entry_point(batch_size=batch_size, device=device, img_dir=img_dir)


def check():
    device = torch.device("cpu")
    model = Model()

    model.load_state_dict(torch.load("model.pt"))
    
    model.eval()
    
     
    test_img = img_for_inference("/home/saint/msr-net/data/low_light_images/1_ll_4.png")
    
    output = model.forward(test_img)
    
    output = model.forward(test_img)
    
    save_inference_img(output=output, path = "after.png")


@app.local_entrypoint()
def main():
    img_dir = "data"
    modal_entrypoint.local(batch_size=64, img_dir=img_dir)
    #check()
