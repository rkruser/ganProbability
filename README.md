Todos:
------
- Finish creating loaders and fill in some model functions (30 min)
- Test the loaders and models (1 hr)
- Write a class parser for "total" so it can run (20 min)
- Create experimental templates based on "total" (20 min)
- Create experiment modules for "total" (1 hr)
- Put matfile paths in the master config (10 min)
- Use a pretrained "deep features" network (add into regressor model) (1 hr)
- Perform specific experiments (2 hrs)

Plan:
-----
Every neural net has a class wrapper that can load it from a file,
initialize it, etc.
Every training model has a class wrapper that can train it (with
    learning rate, etc.), sample it, extract probabilities, test it, 
      and so forth.
    Space can be saved by having general functions/templates which do this, and 
    modifying them in each class
Every dataset has an mltemplate
Every dataset has a loader class that can upsample or downsample it, other things
A module type that can serve for miscellaneous code
Use tensorboard perhaps
Perhaps have a data modifier class that automatically changes how the data
is loaded depending on the net parameters
Make it easy to do large hyperparameter searches


Modifying the mlworkflow pipeline to use one large batch of arguments
  and include all dependencies by default. Secondary/tertiary configs
  use the main config, but can also supercede it
  - Also, make it so you can just call [Operator].save instead of 
    [Operator].files.save


Later...
Implement/use FlowGAN




Add module for runRegressor (done)
Make standard templates for training gan, sampling, running regressor (done)
Add GAN training plots (do next)
Add module that trains regressor by directly sampling from generator,
    skipping saving the training data (this allows an entire experiment
    to be run in a single pipe)
add modules for stackGAN
 - Loader module - get networks/stages, all config args
 - Trainer module - train stages 1/2
 - Sampling module - sample stages 1/2
 - Regressor module (not been done yet)
 - Other analysis

add modules for other models / datasets
