import torch
import torch.nn as nn
import time
from pathlib import Path
from typing import Type, List, Dict, Any, Union
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from optimizer_dynamic.metaprogram_model_builder import MetaModelBuilder
from optimizer_dynamic.compiled_model_builder import CompiledDynamicModel as LegacyDynamicModel

# --- Native PyTorch Model Definitions ---

# region VGG11 Native
class NativeVGG11(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
# endregion

# region ResNet18 Native
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class NativeResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
# endregion

# region U-Net Native
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class BaseNativeUNet(nn.Module):
    def __init__(self, in_channels, out_channels, depths):
        super().__init__()
        self.inc = DoubleConv(in_channels, depths[0])
        
        self.downs = nn.ModuleList()
        for i in range(len(depths) - 1):
            self.downs.append(Down(depths[i], depths[i+1]))
            
        self.ups = nn.ModuleList()
        for i in range(len(depths) - 1, 0, -1):
            self.ups.append(Up(depths[i] + depths[i-1], depths[i-1]))
            
        self.outc = nn.Conv2d(depths[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        skip_connections.append(self.inc(x))
        
        for i, down in enumerate(self.downs):
            skip_connections.append(down(skip_connections[-1]))
            
        x = skip_connections.pop()
        
        for i, up in enumerate(self.ups):
            x = up(x, skip_connections.pop())
            
        return self.outc(x)

class NativeUNetDepth2(BaseNativeUNet):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__(n_channels, n_classes, [64, 128])
        
class NativeUNetDepth3(BaseNativeUNet):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__(n_channels, n_classes, [64, 128, 256])

class NativeUNetDepth5(BaseNativeUNet):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__(n_channels, n_classes, [64, 128, 256, 512, 1024])

# endregion

# --- Benchmarking Utilities ---

def benchmark(model_class: Type[nn.Module], input_tensor: torch.Tensor, is_native: bool, config_path: Union[str, None] = None, n_runs=15):
    """Measures creation and forward pass time."""
    t0 = time.perf_counter()
    if is_native:
        model = model_class()
    else:
        assert config_path is not None, "config_path must be provided for dynamic models"
        model = model_class(config_path)
    creation_time = (time.perf_counter() - t0) * 1000
    
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(input_tensor)
        forward_time = ((time.perf_counter() - t0) / n_runs) * 1000
        
    return creation_time, forward_time

def plot_results(results: List[Dict[str, Any]]):
    """Generates and saves a bar chart of the benchmark results."""
    labels = [r['name'] for r in results]
    native_times = [r['native_forward'] for r in results]
    legacy_times = [r['legacy_forward'] for r in results]
    meta_times = [r['meta_forward'] for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(18, 10))
    rects1 = ax.bar(x - width, native_times, width, label='Native PyTorch', color='cornflowerblue')
    rects2 = ax.bar(x, legacy_times, width, label='Legacy Dynamic (Loop)', color='mediumseagreen')
    rects3 = ax.bar(x + width, meta_times, width, label='Meta Dynamic (No Loop)', color='salmon')

    ax.set_ylabel('Forward Pass Time (ms)')
    ax.set_title('Performance Benchmark: Native vs. Dynamic Model Builders', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for rect_group in [rects1, rects2, rects3]:
        for rect in rect_group:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    
    output_path = "benchmark_results_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n✅ Benchmark graph saved to '{output_path}'")


# --- Main Execution ---

if __name__ == "__main__":
    config_dir = Path(__file__).parent
    
    TEST_CASES = [
        {"name": "VGG-11", "config": config_dir / "vgg11_config.json", "native_class": NativeVGG11, "input": torch.randn(2, 3, 224, 224)},
        {"name": "ResNet-18", "config": config_dir / "resnet18_config.json", "native_class": NativeResNet18, "input": torch.randn(2, 3, 224, 224)},
        {"name": "U-Net (Depth 2)", "config": config_dir / "unet_depth2_config.json", "native_class": NativeUNetDepth2, "input": torch.randn(1, 3, 128, 128)},
        {"name": "U-Net (Depth 3)", "config": config_dir / "unet_depth3_config.json", "native_class": NativeUNetDepth3, "input": torch.randn(1, 3, 256, 256)},
        {"name": "U-Net (Depth 5)", "config": config_dir / "unet_depth5_config.json", "native_class": NativeUNetDepth5, "input": torch.randn(1, 3, 512, 512)},
    ]

    print("🚀 Exhaustive Benchmark: Native vs. Legacy vs. Meta Builders 🚀")
    print("="*70)
    print("Comparing performance of handwritten models against two dynamic builders.")
    print("1. Legacy Dynamic: Uses a 'for' loop in the forward pass.")
    print("2. Meta Dynamic: Generates a loop-free forward pass.")
    print("="*70 + "\n")

    all_results = []
    for case in TEST_CASES:
        print(f"📊 Benchmarking Architecture: {case['name']}")
        print("-" * 50)
        
        # New "Meta" model
        _, meta_forward = benchmark(MetaModelBuilder, case["input"], is_native=False, config_path=str(case["config"]))
        print(f"   🔹 Meta Dynamic (No Loop): {meta_forward:.2f}ms / iter")
        
        # Legacy model
        _, legacy_forward = benchmark(LegacyDynamicModel, case["input"], is_native=False, config_path=str(case["config"]))
        print(f"   🔸 Legacy Dynamic (Loop):  {legacy_forward:.2f}ms / iter")

        # Native model
        _, native_forward = benchmark(case["native_class"], case["input"], is_native=True)
        print(f"   ⭐ Native PyTorch:           {native_forward:.2f}ms / iter")
        
        overhead_legacy = ((legacy_forward / native_forward) - 1) * 100
        overhead_meta = ((meta_forward / native_forward) - 1) * 100
        print(f"   📈 Overhead: Legacy={overhead_legacy:+.1f}%, Meta={overhead_meta:+.1f}%")
        print("-" * 50 + "\n")
        
        all_results.append({
            "name": case["name"],
            "meta_forward": meta_forward,
            "legacy_forward": legacy_forward,
            "native_forward": native_forward
        })
    
    plot_results(all_results)
    
    print("="*70)
    print("✅ Benchmark Finished.")
    print("="*70) 