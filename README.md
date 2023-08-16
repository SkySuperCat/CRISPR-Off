# CRISPROff: sgRNA Off-Target Prediction with Boosting Deep Learning

This is the official repository of the paper: **CRISPROff for sgRNA off-target prediction based on boosting deep learning**.

## PREREQUISITE:

CRISPROFF was conducted using TensorFlow version 2.3.2 and Python version 3.6. 

To set up the required environment:
conda env create -f environment.yml


## Data Preprocessing:

To preprocess your own data, first navigate to the `/data` directory:
```
cd /data
```

Then, run the respective Python files for different datasets. Note: You'll need to manually change the file name in the codes.

### GUIDE-seq:
- On Windows:
```
python guide_preprocess.py
```
- On Linux:
```
python3 guide_preprocess.py
```

### CIRCLE-seq:
- On Windows:
```
python CIRCLE_process.py
```
- On Linux:
```
python3 CIRCLE_process.py
```

### GloVe Embedding:
- On Windows:
```
python glove_process.py
```
- On Linux:
```
python3 glove_process.py
```

After running the preprocessing scripts, `.pkl` files with different dimensions will be created for training. 

## Model Training:

You can modify or add your own models in `model_get.py`. To train your own model, navigate back to the root directory and start the Jupyter notebook:
```
jupyter notebook
```

Then, execute the code in `train.ipynb` on your local server.

## Model Evaluation:

You can either use your trained model for evaluation or use our pre-trained models available in the `saved_model` folder. Perform the evaluations using `cross_validation.ipynb`.

## Benchmark datasets:

- GUIDE-seq(Hek293t)
- GUIDE-seq(K562)
- CIRCLE-seq
- SITE-seq
- ELEVATION
