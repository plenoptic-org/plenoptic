---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Please provide a short, reproducible example of the error, for example:
```
import plenoptic as po
import torch
import imageio

img = torch.tensor(imageio.imread('data/einstein.png'), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
model = SomeModel()
met = po.synth.Metamer(img, model)
# this raises an error
met.synthesize(max_iter=100)
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**System (please complete the following information):**
 - OS: [e.g. Mac (with version), Ubuntu 18.04]
 - Python version [e.g. 3.7]
 - Pytorch version [e.g., 1.4]
 - Plenoptic version [e.g. 0.1]

**Additional context**
Add any other context about the problem here.
