# Variable_Transformer

```sh
.
├── LICENSE
├── README.md
├── image
├── notebook
└── src
    ├── data_provider    # data load, preprocessing, dataloader setting
    │   ├── dataloader.py
    │   └── dataset.py
    ├── layers    # layers for models (attention, embedding, etc.)
    │   ├── Attention.py
    │   ├── Embed.py
    │   └── Transformer_Enc.py
    ├── models
    │   ├── VTTPAT.py
    │   └── VTTSAT.py
    ├── utils    # utils
    │   ├── metrics.py   # metrics for inference
    │   ├── tools.py    # adjust learning rate, visualization, early stopping
    │   └── utils.py    # seed setting, load model, version build, progress bar, check points, log setting
    ├── scripts    # utils
    ├── config.yaml    # configure
    │   ├── run.sh
    │   └── test.sh
    ├── main.py    # main code
    └── model.py    # model build (build, train, validation,test, inference)
   

5 directories, 18 files

```
