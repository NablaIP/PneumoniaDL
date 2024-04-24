# PneumoniaDL

In this repository, we analyze XRay's from a population of people which have Pneumonia of the viral and bacterial type.

We employ a CNN model to classify the images into Normal, Viral Pneumonia and Bacterial Pneumonia.

The dataset being used is from Kaggle and can be found [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

Note that, we have performed an initial analysis of the dataset in `data_exploration.ipynb` which rescales the images and uses only a grayscale channel. Then, we have prepared a dataset and a model class which are used in the `ml_notebook.ipynb` to perform an initial training of the model and thereafter validation of the results.

## Next Steps:

- Further training needs to be performed over a longer range of epochs.
- Hyperparameter tuning search.