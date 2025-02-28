import torch
import onnxruntime as ort

print("PyTorch Version:", torch.__version__)
print("PyTorch GPU Available:", torch.cuda.is_available())
print("ONNX Runtime Providers:", ort.get_available_providers())