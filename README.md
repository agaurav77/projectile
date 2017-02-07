The problem statement is in `problem.txt`.

All python scripts are here for reference, they may contain old code. Latest code is in the IPython2 notebook. Simply install Jupyter and in the directory do

```
$ jupyter notebook
```

This should start a server (on my PC it starts on localhost:8888). Then open that address in your browser and open the IPython notebook `Predicting Projectiles.ipynb`.

You should have:
* Tensorflow
* matplotlib
* numpy (optionally scipy)

Additionally install mathjax for your system. Jupyter doesn't render math well otherwise.

Other Files
-----------


Old code may be in:

* Run `main.py` with Python2 to see a MLP NN simulation
* Run `RNN.py` with Python2 to see a RNN simulation

There are two folders `results_1000` and `results_30000` as well, which contain samples of predicted projectile paths after 1000 and 30000 epochs of training.
