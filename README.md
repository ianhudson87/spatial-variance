# my master's project :D

## how to use:

## Supervised models:
### Training
``python supervisedLearning.py [task] [gpu] [model]``

#### task (string):

the task you choose looks under the task_configs json file for parameters

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
    
example: ``python supervisedLearning.py undersample 0 dncnn_dynamic_specnorm``

### Output and testing
