# How to reproduce the experiments results

* First, train a model under inductive setting. 
```
bash scripts/run_small_dataset.sh $GPU_NUMBER
```

GPU_NUMBER is the index of GPU to use. For example, ```bash scripts/run_small_dataset.sh 3``` means using GPU 3 to train the models.

* Second, perform the edit on pretrained model.

```
bash scripts/eval.sh $GPU_NUMBER $PATH_TO_PRETRAINED_MODEL
```

PATH_TO_PRETRAINED_MODEL is the absolute path to the pretrained model. Remember to set ```$model, $dataset``` in the eval.sh to match the type and dataset of pretrained model.