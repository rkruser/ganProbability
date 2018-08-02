Todos:
------
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
