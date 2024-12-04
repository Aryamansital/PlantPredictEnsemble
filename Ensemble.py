from argparse import ArgumentParser
import pandas as pd
from urllib.request import urlopen
from PIL import Image
import torch
import torchvision.transforms as transforms
from utils import get_model
import warnings
warnings.filterwarnings("ignore")


def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return df['species'].to_dict()


def get_ensemble_models(args, num_classes):
    """Load multiple models for the ensemble."""
    model_names = args.models.split(",")  # Comma-separated model names
    pretrained_paths = args.pretrained_paths.split(",")  # Comma-separated pretrained paths
    
    if len(model_names) != len(pretrained_paths):
        raise ValueError("Number of models must match the number of pretrained paths.")

    models = []
    for model_name, pretrained_path in zip(model_names, pretrained_paths):
        args.model = model_name.strip()
        args.pretrained_path = pretrained_path.strip()
        
        # Load the model with the specific pretrained path
        model = get_model(args, num_classes)
        model.to(args.device)
        model.eval()
        models.append(model)
    return models


def main(args):
    cid_to_spid = load_class_mapping(args.class_mapping)
    spid_to_sp = load_species_mapping(args.species_mapping)
    device = torch.device(args.device)

    # Load ensemble models
    models = get_ensemble_models(args, len(cid_to_spid))

    # Define preprocessing transforms
    if args.pretrained:
        transform = transforms.Compose([
            transforms.Resize(size=args.image_size),
            transforms.CenterCrop(size=args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=args.image_size),
            transforms.CenterCrop(size=args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4425, 0.4695, 0.3266], std=[0.2353, 0.2219, 0.2325])
        ])

    # Load the image
    img = None
    if 'https://' in args.image or 'http://' in args.image:
        img = Image.open(urlopen(args.image))
    elif args.image is not None:
        img = Image.open(args.image)

    if img is not None:
        img = transform(img).unsqueeze(0).to(device)

        # Get predictions from all models
        outputs = []
        with torch.no_grad():
            for model in models:
                outputs.append(model(img))

        # Average the outputs
        output = torch.mean(torch.stack(outputs), dim=0)

        # Extract top-3 predictions
        top3_probabilities, top3_class_indices = torch.topk(output.softmax(dim=1) * 100, k=3)
        top3_probabilities = top3_probabilities.cpu().detach().numpy()
        top3_class_indices = top3_class_indices.cpu().detach().numpy()

        for proba, cid in zip(top3_probabilities[0], top3_class_indices[0]):
            species_id = cid_to_spid[cid]
            species = spid_to_sp[species_id]
            print(species_id, species, proba)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--image", type=str, default='C:\\Users\\aryam\\Downloads\\lupinus_polyphyllus_lindl.jpeg')
    parser.add_argument("--class_mapping", type=str, default='class_mapping.txt')
    parser.add_argument("--species_mapping", type=str, default='species_id_to_name.txt')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--models", type=str, default='inception_v4,efficientnet_b4,inception_resnet_v2,vit_base_patch16_224')  # List of models 
    parser.add_argument("--pretrained_paths", type=str, default='Weights/inception_v4_weights_best_acc.tar,Weights/efficientnet_b4_weights_best_acc.tar,Weights/inception_resnet_v2_weights_best_acc.tar,Weights/vit_base_patch16_224_weights_best_acc.tar')  # List of paths 
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--crop_size", type=int, default=224)

    args = parser.parse_args()
    main(args)
