# xfeatc
cpp port the [xfeat](https://github.com/verlab/accelerated_features).

# Usage
### 1. Prepare onnxruntime.    
Download onnxruntime suitable for your system from [here](https://github.com/microsoft/onnxruntime/releases).
  (I used v1.18.0). And unzip it, and put it to somewhere you like, for example `D:/software/onnxruntime-win-x64-1.18.0`

### 2. Build
```bash
cmake .. -DONNXRUNTIME_ROOT=YOUR_ONNXRUNTIME_DIR
```

### 3. Run
You can directly run the `DetectDemo` and `MatchDemo`, the data has been prepared in the `data/` folder.

<mark> I have fixed the input image dimension to 1x1x640x640 when exporting the onnx model, so these demos can 
only support 640x640 gray images for now. </mark>

# Export onnx
if you are interested in how to export onnx using pytorch, here is my code:

```python
import torch
from modules.xfeat import XFeatModel
import onnxruntime as ort

# os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

# set the model to evaluation mode
net = XFeatModel().eval()

# load the pretrained weights
net.load_state_dict(torch.load("weights/xfeat.pt", map_location=torch.device('cpu')))

# Random input
x = torch.randn(1, 1, 640, 640)

# export to ONNX
torch.onnx.export(net, x, "xfeat.onnx", verbose=True,
                  input_names=['input'],
                  output_names=['output_feats', "output_keypoints", "output_heatmap"],
                  opset_version=11)

print("ONNX model saved as xfeat.onnx")

# check the onnx model with onnxruntime
ort_session = ort.InferenceSession("xfeat.onnx")
print("ONNX model loaded successfully")

outputs = ort_session.run(None, {"input": x.numpy()})

# pytorch model outputs
torch_outputs = net(x)

# compare the outputs
for i in range(len(outputs)):
    print(f"onnx output shape {i}: {outputs[i].shape}")
    print(f"torch output shape {i}: {torch_outputs[i].shape}")
    print(f"Output {i} comparison: {torch.allclose(torch_outputs[i], torch.tensor(outputs[i]))}")
    print(f'Output {i} max diff: {torch.max(torch.abs(torch_outputs[i] - torch.tensor(outputs[i])))}')
    print("\n")

```




