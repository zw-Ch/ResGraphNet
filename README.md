# ResGraphNet
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7213337.svg)](https://doi.org/10.5281/zenodo.7213337)<br>
ResGraphNet is a deep neural network used to predict temperature time series. It effectively combines Graph neural network (GNN) with ResNet modules.<br>
The paper is available in [https://doi.org/10.1016/j.aiig.2022.11.001](https://www.sciencedirect.com/science/article/pii/S2666544122000314).

## Installation
ResGraphNet is based on [Pytorch](https://pytorch.org/docs/stable/index.html) and [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)<br>
Firstly please create a virtual environment for yourself<br>
`conda create -n your-env-name python=3.11`<br><br>
Then, there are some Python packages need to be installed <br>
```
conda install pytorch torchvision torchaudio cudatoolkit=12.1
conda install pyg -c pyg
conda install matplotlib
```

<!---
`conda install statsmodels`<br>
-->

## Training and Testing Models
The running programs based on different models is in [run](https://github.com/zw-Ch/ResGraphNet/tree/main/run),  and you can type the following command to train the model:<br>
`python run_ResGraphNet.py`<br>

## Graphical User Interface (GUI)
We also provide a GUI program [run_python.py](https://github.com/zw-Ch/ResGraphNet/blob/main/gui/run_python.py) so that you can test each dataset and each network model more quickly and intuitively, as shown in the following figure:<br>

![image](https://github.com/zw-Ch/ResGraphNet/blob/main/gui_example.png)
