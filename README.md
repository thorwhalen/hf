# hf

Simple Mapping interface to HuggingFace

To install:	```pip install hf```


# Examples

```python
from hf.base import HfModels, HfDatasets

d = HfDatasets()
list(d)
# ['stingning/ultrachat', 'allenai/WildChat-1M']

data = d['stingning/ultrachat']
```


