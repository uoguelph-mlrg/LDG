# LDG

PyTorch code for our paper on [Learning Temporal Attention in Dynamic Graphs with Bilinear Interactions](https://arxiv.org/abs/1909.10367).

Data can be accessed [here](http://realitycommons.media.mit.edu/socialevolution1.html).

Running the model with a learned graph, sparse prior and biliear interactions:

`python main.py --log_interval 300  --epochs 5 --data_dir ./SocialEvolution/ --encoder mlp --bilinear --sparse`
