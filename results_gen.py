from tagger.model import DualInputCNN
from tagger.data_loader import CustomDataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main_tagger():
    global device
    model_tagger_path = "models\\tagger.pth"
    model_tagger = DualInputCNN()
    model_tagger.load_state_dict(torch.load(model_tagger_path, map_location=device))
    model_tagger = model_tagger.to(device)
    model_tagger.eval()

    root_dir = "datasets\\Results_datasets\\tagger\\frames"
    csv_file = "datasets\\Results_datasets\\tagger\\frames\\frames_labels.csv"

    resize_size = [232]
    crop_size = [224]

    transform = transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

    dataloader = DataLoader(dataset, batch_size=1)

    bit_accuracies = validate_model(model_tagger, dataloader, device)

    labels_order = ["noche", "soleado", "nublado", "lluvia", "neblina", "sombras"]
    for i, accuracy in enumerate(bit_accuracies, start=1):
        label = labels_order[i - 1]
        print(f"Accuracy for {label}: {accuracy * 100:.2f}%")


def validate_model(model, dataloader, device="cpu"):
    model.to(device)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    correct_bits = [0, 0, 0, 0, 0, 0]

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        print(batch_idx)
        if inputs.shape[0] == 1:
            continue
        height = inputs.size(2)
        upper_section = inputs[:, :, : int(height * 0.25), :]
        lower_section = inputs[:, :, int(height * 0.25) :, :]

        upper_section, lower_section = upper_section.to(device), lower_section.to(
            device
        )
        labels = labels.to(device).type(torch.float32)

        outputs = model(upper_section, lower_section)
        preds = torch.sigmoid(outputs).cpu().detach().numpy() > 0.5
        labels_bool = np.array(labels.cpu().detach().numpy()).astype(bool)
        correct_bits += np.sum(preds == labels_bool, axis=0)

    print(correct_bits)


if __name__ == "__main__":
    main_tagger()
