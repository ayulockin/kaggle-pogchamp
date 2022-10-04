# It's Corn (PogChamps #3)

Classification pipeline for [Kaggle Pogchamp Series](https://www.kaggle.com/competitions/kaggle-pog-series-s01e03/overview) competition built using TF/Keras and W&amp;B.


## Dataset

Download dataset:

```
kaggle competitions download -c kaggle-pog-series-s01e03
```

## Train

```
python train.py --config configs/baseline --wandb --log_model
```

The training and validation metrics will be logged to Weights and Biases. You can find my training logs in this Weights and Biases workspace: https://wandb.ai/ayush-thakur/pogchamp

The model checkpoints will be logged to the same and you can visit the W&B run page > artifacts to select the model you want to use for creating `submission.csv`.

The `<path/to/model/artifact>` is the Full Name shown in the artifact page screenshot below:

![image](https://user-images.githubusercontent.com/31141479/193711574-f8dc1f06-8e7b-4b0a-b9e0-e6f820b07bfd.png)


## Test

```
python test.py --config configs/baseline --model_artifact_path <path/to/model/artifact>
```

This will write the predictions to a `submission.py` file. Download it and upload it for submission.
