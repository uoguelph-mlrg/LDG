# LDG

PyTorch code for our paper on [Learning Temporal Attention in Dynamic Graphs with Bilinear Interactions](https://arxiv.org/abs/1909.10367).

## Data

### Social Evolution

When using this dataset, you must comply with their **conditions** specified [here](http://realitycommons.media.mit.edu/socialevolution1.html).

**Option 1:**

The original data can be accessed [here](http://realitycommons.media.mit.edu/socialevolution1.html).
Once you download their zip file, unpack it to the `SocialEvolution` folder, then inside that folder unpack `Proximity.csv.bz2`, e.g. by running `bzip2 -d Proximity.csv.bz2`.
You can then run our code and it will generate a preprocessed `data_prob0.8.pkl` file that will be reused every time you run our code.

**Option 2:**

Instead of using original data, you can directly download `data_prob0.8.pkl` from [here](https://drive.google.com/file/d/1VTcJaAX1FcnAEVKId40er5JnV7ZKpybz/view?usp=sharing) and put it to the `SocialEvolution` folder.

### Github

The original data can be accessed [here](https://www.gharchive.org/). 
When using this dataset, you must comply with their **licenses** specified [here](https://github.com/igrigorik/gharchive.org#licenses).

In this repo we extract a subnetwork of 284 users with relatively dense events between each other. 
Each user initiated at least 200 communication and 7 association events during the year of 2013. 
"Follow" events in 2011-2012 are considered as initial associations. Communication events include: Watch, Star, Fork, Push, Issues, IssueComment, PullRequest, Commit. This results in a dataset of 284 nodes and around 10k training events (from December to August 2013) and 8k test events (from September to December 2013) .

We provide the preprocessed pkl files in the `Github` folder so that you do not need to access the original data to run our code.


## Examples

### Social Evolution

Running the baseline DyRep model [1] on Social Evolution:

`python main.py --log_interval 300  --data_dir ./SocialEvolution/`.

Running our latent dynamic graph (LDG) model with a learned graph, sparse prior and biliear interactions:

`python main.py --log_interval 300  --data_dir ./SocialEvolution/ --encoder mlp --soft_attn --bilinear --sparse`

Note that on Social Evolution our default option is to filter `Proximity` events by their probability: `--prob 0.8`. In the DyRep paper, they use all events, i.e. `--prob 0.8`. When we compare results in our paper, we use the same `--prob 0.8` for all methods.

### GitHub

To run Github experiments, use the same arguments, but add `--dataset github --data_dir ./Github`.

To use the Frequency bias, add the `--freq` flag.


### Other datasets

I provide the base class [data_loader.py](data_loader.py), showing which class attributes and functions must be implemented if you want to train our model on other datasets. Plus I added [example_data_loader.py](example_data_loader.py), showing a minimal example of using the base class.


## Citation 

If you make use of this code, we appreciate it if you can cite our paper as follows:

```
@ARTICLE{knyazev2019learning,
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
