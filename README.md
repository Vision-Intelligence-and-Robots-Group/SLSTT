# README

### Short and Long Range Relation Based Spatio-Temporal Transformer for Micro-Expression Recognition

This is the offical code for SLSTT. See our paper on [[IEEE]](https://ieeexplore.ieee.org/document/9915457) and [[arXiv]](https://arxiv.org/abs/2112.05851)
![framework](images/framework.png)


The whole code is not available now, but you can use

```
python LOSO.py -e
```

to test the Existing models.

### Requirements&Environments

* I train this model on single 3090
* you can import my environment by environments.yaml, which may add many unnecessary packages.You can also simply pip install when you find it is necessary.

### Before training

1. ##### How the dataset should be located

   you can see the following data directory

   -data -data_raw

   ​		  -landmarks

   Please put the raw data in <data_raw>;

   for example:

   -data -data_raw

   ​							-casme2

   ​											-sub01

   ​											-sub02

   ​											...						

2. ##### change the PATH

   Change line 16 PATH = 'xxx', this path should be where you put your datasets, for example:

   home  -Leo  -data -data_raw

   ​		                        -landmarks

   you should let PATH = '/home/Leo/data'

### Start training

1. ##### How to start training

   you can simply use:

   `python main.py --vit -s 1 --dir ./`

   to run the basic vit_mer, for further training(there is also alternative option) please carefully read the code.

### Evaluate

```
python main.py --vit --resume xxx --evaluate
```

fill the xxx with your checkpoint path

also you can evaluate with LOSO

```
python LOSO.py -e
```

please ensure you model are in the same directory with LOSO.py.

### One more thing

You can also evaluate my already-trained model. I have uploaded it to onedrive.


### Correction of Confusion Matrix

On IEEE version, we mis-swapped the predictions and targets when generate visualised confusion matrix images. Sorry for any confusion this may have caused. We have corrected our arXiv version and show the correct one here.

![Confusion Matrix](images/SLSTT_CMs.png)

### Citiation

If you find our paper and this code useful in your research, please consider citing:
@article{zhang2022short,
  title={Short and long range relation based spatio-temporal transformer for micro-expression recognition},
  author={Zhang, Liangfei and Hong, Xiaopeng and Arandjelovi{\'c}, Ognjen and Zhao, Guoying},
  journal={IEEE Transactions on Affective Computing},
  year={2022},
  publisher={IEEE},
  pages={1-13},
  doi={10.1109/TAFFC.2022.3213509}
  }
