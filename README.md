# AMF_GZS_SBIR
Implementation of work "Augmented Multi-Modality Fusion for GeneralizedZero-Shot Sketch-based Visual Retrieval". 



## Data
---
### DomainNet

A new generalized zero-shot sketch-based image retrieval evaluation protocol constructed on [DomainNet dataset](http://ai.bu.edu/M3SDA/).

#### 1. "Sketch-like" and "Photo-like" domains.

"Sketch-like" domains: Sketch(Sk), Quickdraw(Qu).

"Photo-like" domains: Real(Re), Painting(Pa), Infograph(In), Clipart(Cl).

#### 2. *Seen* and *Unseen* classes splits.

[DomainNet](http://ai.bu.edu/M3SDA/) contains 345 categories intotal. 45 classes never present in [ImageNet](https://www.image-net.org/) are chosen as [*unseen*](/data/DomainNet/domainnet_test_classes.txt), and the rest 300 are [*seen*](/data/DomainNet/domainnet_train_classes.txt). 

#### 3. [Train and test datasets](/data/DomainNet/train_test_splits) follow the same splits as [DomainNet](http://ai.bu.edu/M3SDA/)


#### 4. Experiments details

**Training** stage: "sketch-like" domain *seen* categories training split data are available, and "photo-like" domain *seen* categories all data (including train and test splits) build up the retrieval gallery.

**Test** stage: Both "sketch-like" domain *seen* categories training and test splits data are evaluated. For retrieval gallery made up with the "photo-like" domain, all *unseen* categories data (including train and test splits), together with the same number of randomly selected *seen* categories samples build up the retrieval gallery., to avoid the influence of the data distribution imbalance.

### Sketchy-Extended

21 categories not in ImageNet are treated as *unseen* categories, and the rest 104 classes are for training. For generalized experiments, we follow the previous works setting.

### TUBerlin-Extended

30 categories never present in ImageNet are treated as *unseen* categories, and the rest 220 classes are for training. For generalized experiments, we follow the previous works setting.

### Semantic dataset

We adopt the word-to-vector model pre-trained on Google News dataset (~ 100 billion words) [link](https://arxiv.org/pdf/1301.3781.pdf).

## Dependencies

```python
PyTorch
Numpy
Pandas
Sklearn
```

## Training

```python
python main.py
```

## Evaluation

Evaluation metrics: For *seen* (S) and *unseen* (U) categories, we report the mAP@all, Prec@100, mAP@200, Prec@200. To evaluate the model's generalization ability, we also report the harmonic mean of the seen and unseen categories results: $H=\frac{2 \times S \times H}{S + H} $.

## Citation

If you think this work is interesting, please cite:
```

```

## Contact

If you have any questions about this work, feel free to contact
- tjing@tulane.edu
