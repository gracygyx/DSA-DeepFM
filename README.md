## DSA-DeepFM: A DeepFM Model with Dual-Stage Attention for Predicting Anticancer Synergistic Drug Combinations

- Framework
  ![](https://github.com/gracygyx/DSA-DeepFM/blob/master/pictures/framework.png

- Dual-Stage Attention mechanism
  ![](https://github.com/gracygyx/DSA-DeepFM/blob/master/pictures/Attention.png)

### Data preparation



- We preprocessed the O'Neil dataset and saved it in npy format,  which can be downloaded from Google drive. 

       ```
       link: https://drive.google.com/drive/folders/1KQU7kKH-MFl2bJhsrLv1ZHxm35Nvhhsf?usp=drive_link
       ```




- Place the downloaded npy files in the "DSA-DeepFM" directory.



### Train and Test

Our program is easy to train and test,  just need to run "main_DSA_DeepFM_oneil.py". 

```
python main_DSA_DeepFM_oneil.py
```

### Performance on DrugCombDB and O'Neil datasets

- DrugCombDB dataset
  ![](https://github.com/gracygyx/DSA-DeepFM/blob/master/pictures/DrugCombDB.png)

- O'Neil dataset
  ![](https://github.com/gracygyx/DSA-DeepFM/blob/master/pictures/oneil.png)

