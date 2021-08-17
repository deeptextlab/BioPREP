# BioPREP

<img width="720" alt="Overview" src="https://user-images.githubusercontent.com/63843498/121888927-89094d80-cd53-11eb-9364-707008ffcaac.png">

This repository contains **Dataset (BioPREP: Biomedical Predicate Relation Extraction with entity-filtering by PKDE4J)** and code implementation for fine-tuning pretrained model, as well as inferencing unlabeled dataset using fine-tuned model.

Our open dataset, BioPREP, is based on SemMedDB (https://skr3.nlm.nih.gov/SemMed/index.html). We have extracted entities of sentences in SemMedDB using PKDE4J (link: http://informatics.yonsei.ac.kr/pkde4j/#), then replaced them with Entity Type. After preprocessing the dataset, we comprehensively evaluated performance of several architectures based on neural networks. Following the result of experiments, we found that BioBERT-based model outperformed other models for predicate classification.

For users who want to use BioBERT-based fine-tuned model, not going through the training process of pre-trained model, we uploaded fine-tuned model at Google Drive link. By executing *test.py*, you can just infer the predicate/frame type of unlabelled datasets.

## Requirements

torch

tensorflow

keras

numpy

pandas

sklearn

## How to Use

### 1. Git clone and Set the directory

```python
git clone https://github.com/deeptextlab/BioPREP.git
```

### 2. Train models

You can either fine-tune BERT-based pretrained models(BERT-base, BioBERT and SciBERT), or train neural network-based models(CNN, Multichannel CNN, LSTM, BiLSTM and CNN + LSTM). We simply used the extension of pretrained BERT models for our research.

```python
python3 train.py --model_type='BioBERT' \ 
		--label_type='Predicate' \
  		--seed=42 \
    		--epochs=20 \
      		--batch_size=16 \
        	--test_size=0.2 \
          	--max_len=512 \
                --lr=5e-5 \
              	--eps=1e-8 \
                --eval_interval=5 \
                --output_dir_path='/models' \
                --data_file_path='/BioPREP/train.csv'
```

Values for arguments above are default settings, so if you just run

```python
python3 train.py
```

the training process will run with given default settings.

To train under diffenent conditions, change arguments as you want.

For argument 'model_type', you can select among ***'BioBERT'***, ***'SciBERT'*** and ***'BERT-base'***.

For argument 'label_type', you can choose between ***'Predicate'*** and ***'FrameNet'***.

If you want to use your own dataset, please fit your columns into 'text', 'predicate_answer' and 'framenet_answer', as well as column names.

Note that you can also manage when the fine-tuned weights would be saved by using argument 'eval_interval'. For example, if you give value 5 for argument eval_interval, while setting epoch as 20, the training process would save the weight every 5 epoch: 5, 10, 15 and 20. You can also specify where to save your model by changing the value of argument 'output_dir_path'.

### 3. Inference with unlabelled dataset

With unlabelled dataset, you can infer answers when running the codes below.

```python
python3 test.py --model_type='BioBERT' \ 
		--label_type='Predicate' \
      		--batch_size=16 \
          	--max_len=512 \
                --output_dir_path='/models' \
                --data_file_path='/BioPREP/train.csv'
```

If you want to use other settings, change the arguments.

## Citation

If you want to use our dataset BioPREP, please cite as below.

```
@article{10.1016/j.jbi.2021.103888,
	author={Gibong Hong, Yuheun Kim, Yeonjung Choi, Min Song},
	(in press),
	title={{BioPREP}: Deep Learning-based Predicate Classification with SemMedDB},
	journal={Journal of Biomedical Informatics},
	year={2021},
	month={08},
	doi={10.1016/j.jbi.2021.103888},
	url={https://doi.org/10.1016/j.jbi.2021.103888}
	}
```

## Contact Information

If you have any questions or issues, please contact Gibong Hong <gibong.hong@yonsei.ac.kr> or create a Github issue.