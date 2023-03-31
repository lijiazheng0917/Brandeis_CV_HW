1. First you need to create a pytorch environment using conda:
   ```
   conda create -n pt python=3.8
   conda activate pt
   # choose the appropriate pytorch version according to your device
   conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
   ```
2. To train and evaluate the model, simply do:
   ```
   python main.py
   ```
    Or you can specify the hyperparemeters as:
    ```
    python main.py --bs 32 --epochs 10 --lr 1e-4 --seed 42 --save-model True
    ```

   