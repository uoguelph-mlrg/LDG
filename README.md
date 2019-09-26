# LDG

PyTorch code for our paper on [Learning Temporal Attention in Dynamic Graphs with Bilinear Interactions](https://arxiv.org/abs/1909.10367).

Data can be accessed [here](http://realitycommons.media.mit.edu/socialevolution1.html).

Running the model with a learned graph, sparse prior and biliear interactions:

`python main.py --log_interval 300  --epochs 5 --data_dir ./SocialEvolution/ --encoder mlp --bilinear --sparse`

If you make use of this code, we appreciate it if you can cite our paper as follows:

```
@ARTICLE{Knyazev2019-zj,
  title         = "Learning Temporal Attention in Dynamic Graphs with Bilinear
                   Interactions",
  author        = "Knyazev, Boris and Augusta, Carolyn and Taylor, Graham W",
  month         =  sep,
  year          =  2019,
  archivePrefix = "arXiv",
  primaryClass  = "stat.ML",
  eprint        = "1909.10367"
}
```
