import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm  # progress bar

from unet import UNetMSSRMLP_B


def parse_args():
    parser = argparse.ArgumentParser(
        description="GoPro MSSR-MLP-B Inference with PSNR/SSIM",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example usage:
  python3 test_inference.py \
    -w weights.pth \
    -i Datasets/test/GoPro/input_crops \
    -t Datasets/test/GoPro/target_crops \
    -o outputs \
    -b 4
"""
    )
    parser.add_argument(
        '--weights', '-w',
        type=str,
        required=True,
        help="Path to the .pth model weights file"
    )
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        default="Datasets/test/GoPro/input_crops",
        help="Directory containing input test crops"
    )
    parser.add_argument(
        '--target_dir', '-t',
        type=str,
        required=True,
        help="Directory containing ground-truth images"
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default="outputs",
        help="Directory to save model outputs"
    )
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=4,
        help="Batch size for inference"
    )
    return parser.parse_args()


class GoProTestDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, transform=None):
        self.files = sorted(os.listdir(input_dir))
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = UNetMSSRMLP_B(
        in_channels=3,
        out_channels=3,
        base_channels=42,
        enc_depths=[2, 4, 12],
        bottleneck_depth=4,
        dec_depths=[12, 4, 2],
        mssr_window_size=4,
        mssr_step_sizes=[0, 2, 4],
        input_size=256
    ).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    # Measure inference FLOPs (MACs) using ptflops
    try:
        from ptflops import get_model_complexity_info
        macs, params_str = get_model_complexity_info(
            model, (3, 256, 256), as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
        print(f"Inference complexity: {macs} | Model params (ptflops): {params_str}")
        # Approximate training FLOPs: forward + backward ~ 2x inference MACs
        print(f"Approx. training FLOPs (MACs): {macs} x 2")
    except ImportError:
        print("Install ptflops (`pip install ptflops`) to measure FLOPs")

    # Also report exact parameter count:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_ds = GoProTestDataset(args.input_dir, transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    os.makedirs(args.output_dir, exist_ok=True)

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for batch, fnames in tqdm(test_loader, desc="Inference", unit="batch"):
            batch = batch.to(device)
            outputs = model(batch)
            outputs = torch.clamp(outputs, 0, 1)
            for out_tensor, fname in zip(outputs, fnames):
                # save output image
                out_img = transforms.ToPILImage()(out_tensor.cpu())
                out_img.save(os.path.join(args.output_dir, fname))

                # load ground truth
                gt_img = Image.open(os.path.join(args.target_dir, fname)).convert('RGB')
                gt_tensor = transforms.ToTensor()(gt_img)

                # compute PSNR and SSIM
                out_np = out_tensor.cpu().permute(1,2,0).numpy()
                gt_np  = gt_tensor.permute(1,2,0).numpy()
                psnr = peak_signal_noise_ratio(gt_np, out_np, data_range=1.0)
                ssim = structural_similarity(gt_np, out_np, channel_axis=2, data_range=1.0)

                total_psnr += psnr
                total_ssim += ssim
                count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"Processed {count} images.")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    print(f"Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
