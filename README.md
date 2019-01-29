# RAMP starting kit Garbage Images Classification

The dataset was found here [https://github.com/garythung/trashnet]

_Authors: Hao Liu, Jiaxin Gao, Robin Duraz, Trung Vu Thanh, Théo Cornille_

Garbages are omnipresent in today's society and many recyclable trash end up being thrown away and left untreated. In many countries, people are asked to sort their trash according to some categories, but depending on where, it is more or less done. Automatically recognizing and then separating all kinds of trash can make the task of treating garbages much easier.
Thus, this RAMP challenge is about image classification of 6 kinds of trash:
cardboard, glass, metal, paper, plastic, and the last class is composed of other kinds of trash.

#### Set up for RAMP

Open a terminal and

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
  ```
  
2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [dedicated notebook](starting_kit.ipynb).

### Submissions

Execute
```
ramp_test_submission
```
to try the starting kit submission. You can also use
```
ramp_test_submission --submission=other_dir
```
to test any other submission located in a directory in the submission directory.
