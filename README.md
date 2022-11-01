# README

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