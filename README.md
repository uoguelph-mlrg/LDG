# LDG

PyTorch code for our paper on [Learning Temporal Attention in Dynamic Graphs with Bilinear Interactions](https://arxiv.org/abs/1909.10367).

## Data

All data are uploaded to this repo. The original data can be accessed [here](http://realitycommons.media.mit.edu/socialevolution1.html).

Before running the code, unpack `Proximity.csv.bz2`, e.g. by running `bzip2 -d Proximity.csv.bz2` inside the SocialEvolution folder.

## Examples

Running the baseline DyRep model [1]:

`python main.py --log_interval 300  --epochs 5 --data_dir ./SocialEvolution/`.

Running our latent dynamic graph (LDG) model with a learned graph, sparse prior and biliear interactions:

`python main.py --log_interval 300  --epochs 5 --data_dir ./SocialEvolution/ --encoder mlp --bilinear --sparse`

Note that our default option is to filter `Proximity` events by their probability: `--prob 0.8`. In the DyRep paper, they use all events, i.e. `--prob 0.8`. When we compare results in our paper, we use the same `--prob 0.8` for all methods.

If you make use of this code, we appreciate it if you can cite our paper as follows:

```
@ARTICLE{Knyazev2019-zj,
  title         = "Learning Temporal Attention in Dynamic Graphs with Bilinear Interactions",
  author        = "Knyazev, Boris and Augusta, Carolyn and Taylor, Graham W",
  month         =  sep,
  year          =  2019,
  archivePrefix = "arXiv",
  primaryClass  = "stat.ML",
  eprint        = "1909.10367"
}
```

[1] [Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, and Hongyuan Zha. DyRep: Learning
representations over dynamic graphs. In ICLR, 2019](https://openreview.net/forum?id=HyePrhR5KX)
