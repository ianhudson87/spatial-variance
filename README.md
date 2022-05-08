# my master's project :D

## Supervised models:
### Training
``python supervisedLearning.py [task] [gpu] [model]``

this will create folder in the ``./runs/`` folder. The name of the folder will contain the task name, model name, and date.
    
example: ``python supervisedLearning.py undersample 0 dncnn_dynamic_specnorm``

#### task (string):

the task you want to use. The script looks at json config file for parameters, which is stored under the ``task_configs`` folder.

- "undersample": undersample the k-space of your input image. Sampling rate determined by undersample_options.json

- "vnoise": different gaussian noise levels in different patches of the images. Patch size, min var, and max var, all determined by vnoise_options.json.
  Note: for uniform gaussian noise, just set min var and max var to the same thing

- "quarter": each quarter of the image gets a different noise level. 
  
#### gpu (int):
just what gpu number you want to use

#### model (string):
  Relevant ones:
- "dncnn_spec" - dncnn with spectral normalization. 

- "dncnn_dynamic_specnorm" - dncnn with a single dynamic single kernel block. There is 1 layer as dncnn output into the dynamic kernel generator. Uses spectral norm for the dncnn part.

- "dncnn_dynamic_specnorm_more_out_layers" - dncnn with a single dynamic single kernel block. There are many layers as dncnn output into the dynamic kernel generator. Uses spectral norm for the dncnn part.

- "dncnn_dynamic_specnorm_more_dyn_layers" - dncnn with multiple dynamic single kernel block. There is always 1 layer as input and 1 layer as output in the dynamic kernel part. Uses spectral norm for the dncnn part.

All models are defined in the ``model_zoo`` folder.

### Testing
To run trained models on the test set:
``python supervisedLearningTesting.py [task] [gpu #] [checkpoint_name] [epoch] [model_name]``

Example: ``python supervisedLearningTesting.py vnoise 0 dncnn_vnoise_10-15-19-26 50 dncnn``

This will create a folder in the test_logs folder. The folder name will have the name of the model, the name of the task, and the date.

task, gpu, model_name are all the same as before.
- Note: the model structure has to be the same as when training because the testing script just loads the weights.
- Also, the task used will look at the task config json file, so you most likely want to make sure that the task config is the same as when testing

#### checkpoint_name (string):
folder name in the ``./runs/`` folder that contains the checkpoint you want ot use

#### epoch (int):
which epoch you want to use from your checkpoints folder.

## Model based learning
To run RED with grid search:
``modelBasedLearning.py [task] [gpu #] [checkpoint_name] [epoch] [model_name]``

The results will be put in a folder under the test_logs folder. The folder name will contain 'modelbased_learning', the name of the denoiser used, the task used, and the date

Example ``python modelBasedLearning.py undersample 5 dncnn_dynamic_specnorm_more_dyn_layers_vnoise_4-8-7-13 22 dncnn_dynamic_specnorm_more_dyn_layers``

#### Arguments
Uses the same command line arguments as supervised learning testing. The model that you pass in is used as the denoiser in RED.

#### Gridsearch
For gridsearch, the hyperparameters that are tested are on line 260 of modelBasedlearning.py. Each combination of parameters gets tested on the validation set, specified in the image_test_nums parameters (line 265)
