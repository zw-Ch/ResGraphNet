# ResGraphNet
ResGraphNet is a deep neural network used to predict temperature time series. It effectively combines Graph neural network (GNN) with ResNet modules.<br>
The paper is available in <br>

## Installation
ResGraphNet is based on [Pytorch](https://pytorch.org/docs/stable/index.html) and [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)<br>
Firstly please create a virtual environment for yourself<br>
`conda create -n your-env-name python=3.9`<br><br>
Then, there are some Python packages need to be installed<br>
`conda install pytorch torchvision torchaudio cudatoolkit=11.3`<br>
`conda install pyg -c pyg`<br>
`conda install matplotlib`<br>
<!---
`conda install statsmodels`<br>
-->

## Training and Testing Models
The running programs based on different models is in [run](https://github.com/czw1296924847/ResGraphNet/run),  and you can type the following command to train the model:<br>
`python run_ResGraphNet.py`<br>

## Graphical User Interface (GUI)
We also provide a GUI program [run_python.py](https://github.com/czw1296924847/ResGraphNet/blob/main/gui/run_python.py) so that you can test each dataset and each network model more quickly and intuitively, as shown in the following figure:<br>

![image](https://github.com/czw1296924847/ResGraphNet/blob/main/gui_example.png)
