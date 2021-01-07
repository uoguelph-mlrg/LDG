
The original data can be accessed [here](https://www.gharchive.org/).
When using this dataset, you must comply with their **licenses** specified [here](https://github.com/igrigorik/gharchive.org#licenses).

In this repo we extract a subnetwork of 284 users with relatively dense events between each other.
Each user initiated at least 200 communication and 7 association events during the year of 2013.
"Follow" events in 2011-2012 are considered as initial associations. Communication events include: Watch, Star, Fork, Push, Issues, IssueComment, PullRequest, Commit. This results in a dataset of 284 nod$

We provide the preprocessed pkl files in the `Github` folder so that you do not need to access the original data to run our code.
