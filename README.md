# Autonomous Driving Scene Understanding
## Multi-Class Image Segmentation — Full Pipeline

### Project Structure
```
autonomous_segmentation/
│
├── dataset.py          Module 1 — Cityscapes dataset, augmentations, dataloaders
├── models.py           Module 2 — U-Net++, DeepLabv3+, PSPNet, SegFormer, Mask2Former
├── training.py         Module 3 — Loss functions, optimizers, training loop, TensorBoard
├── evaluation.py       Module 4/5 — Metrics (IoU/mIoU/Dice/PA/HD) + visualizations
├── extensions.py       Module 6/7 — Instance seg, domain adaptation, ONNX export, LaTeX
├── run_pipeline.py     Master orchestration script
├── requirements.txt    Python dependencies
└── README.md           This file
```

### Quick Start

#### 1. Install dependencies
```bash
# PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

#### 2. Download Cityscapes
Register at https://www.cityscapes-dataset.com/ and download:
- `gtFine_trainvaltest.zip` (~241 MB)
- `leftImg8bit_trainvaltest.zip` (~11 GB)

Extract to `/data/cityscapes/` with structure:
```
/data/cityscapes/
  ├── gtFine/train/  val/  test/
  └── leftImg8bit/train/  val/  test/
```

#### 3. Train a single model
```bash
python run_pipeline.py \
  --data /data/cityscapes \
  --model segformer_b2 \
  --epochs 100 \
  --batch_size 4 \
  --lr 6e-5 \
  --optimizer adamw \
  --scheduler warmup_cosine \
  --output_dir outputs/
```

#### 4. Evaluate only (load checkpoint)
```bash
python run_pipeline.py \
  --data /data/cityscapes \
  --model segformer_b2 \
  --eval_only \
  --ckpt outputs/segformer_b2/checkpoints/best.pth
```

#### 5. Compare all architectures
```bash
python run_pipeline.py \
  --data /data/cityscapes \
  --compare_all \
  --epochs 50 \
  --output_dir outputs/comparison/
```
Generates: model_comparison.png, efficiency_plot.png, metrics_table.tex, report_skeleton.tex

#### 6. Export for deployment
```bash
python run_pipeline.py \
  --data /data/cityscapes \
  --model segformer_b2 \
  --eval_only \
  --ckpt outputs/segformer_b2/checkpoints/best.pth \
  --export_onnx
```

### TensorBoard
```bash
tensorboard --logdir outputs/segformer_b2/tb_logs
```

### Models Available
| Name           | Params | Architecture                     |
|----------------|--------|----------------------------------|
| unetplusplus   | 31 M   | Dense skip + attention gates     |
| deeplabv3plus  | 59 M   | ResNet-101 + ASPP decoder        |
| pspnet         | 46 M   | ResNet-50 + Pyramid Pooling      |
| segformer_b0   | 3.7 M  | Lightweight transformer          |
| segformer_b2   | 27 M   | Accurate transformer             |
| mask2former    | 44 M   | Query-based, panoptic-capable    |

### Metrics Computed
- mIoU (mean Intersection over Union)
- Dice Coefficient (per-class and mean)
- Pixel Accuracy
- Mean Pixel Accuracy
- Hausdorff Distance (boundary quality, opt-in)
- Inference FPS

### Outputs
```
outputs/<model>/
  ├── checkpoints/best.pth
  ├── tb_logs/               TensorBoard event files
  ├── training_curves.png
  ├── iou_<model>.png        Per-class IoU bar chart
  ├── vis/                   Overlay visualizations
  ├── analysis/              Critical analysis JSON
  ├── <model>.onnx           (if --export_onnx)
  └── <model>_scripted.pt    TorchScript export

outputs/                     (compare_all mode)
  ├── model_comparison.png
  ├── efficiency_plot.png
  ├── metrics_table.tex      LaTeX-ready table
  ├── report_skeleton.tex    Full IEEE report skeleton
  ├── references.bib         BibTeX entries
  └── all_results.json
```

### Cityscapes 19 Classes
road, sidewalk, building, wall, fence, pole,
traffic light, traffic sign, vegetation, terrain,
sky, person, rider, car, truck, bus, train,
motorcycle, bicycle
