# Variable_Transformer

```sh
.
├── LICENSE
├── README.md
└── src
    ├── data_provider    # data load, preprocessing, dataloader setting
    │   ├── dataloader.py
    │   └── dataset.py
    ├── layers    # layers for models (attention, embedding, etc.)
    │   ├── Attention.py
    │   ├── Embed.py
    │   └── Transformer_Enc.py
    ├── models
    │   ├── LSTM_AE.py
    │   ├── LSTM_VAE.py
    │   ├── VariableTransformer1.py
    │   ├── VariableTransformer2.py
    │   ├── VariableTransformer3.py
    │   └── VariableTransformer4.py
    ├── utils    # utils
    │   ├── metrics.py   # metrics for inference
    │   ├── tools.py    # adjust learning rate, visualization, early stopping
    │   └── utils.py    # seed setting, load model, version build, progress bar, check points, log setting
    ├── config.yaml    # configure
    ├── main.py    # main code
    ├── model.py    # model build (build, train, validation,test, inference)
    └── run.sh    # shell script for experiment
   

5 directories, 18 files

```
