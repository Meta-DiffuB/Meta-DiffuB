# Meta-DiffuB: A Contextualized Sequence-to-Sequence Text Diffusion Model with Meta-Exploration
<p align="center">
    Yun-Yen Chuang · Hung-Min Hsu · Kevin Lin · Chen-Sheng Gu · Ling-Zhen Li · Ray-I Chang · Hung-yi Lee<br>
    <a href="https://neurips.cc/media/neurips-2024/Slides/95436_gptxu0a.pdf">[Slide]</a>
    <a href="https://neurips.cc/media/PosterPDFs/NeurIPS%202024/95436.png?t=1729485777.3869653">[Poster]</a>  
    <img src="/img/MAXORA_AI.jpg" alt="圖片" style="float: right; width: 200px; height: auto;">
</p>

## Meta-DiffuB
![Image Alt text](/img/Meta_DiffuB.jpg)
Comparison between S2S-Diffusion model (i.e., DiffuSeq) and the proposed Meta-DiffuB. The shades of color represent different amounts of noise being imposed.
Different from prior works that use a fixed noise, we introduce a novel scheduler-exploiter framework, Meta-DiffuB, which achieves trainable noise scheduling inspired by Meta Exploration. Our scheduler model schedules contextualized noise, enhancing the training and generation of the S2S-Diffusion model, resulting in state-of-the-art (SOTA) performance compared to previous S2S-Diffusion models, as detailed in Section 4.

## Getting started
Our implementation is based on Python 3.8, PyTorch 1.11 and Fairseq 0.10.2. The following command will install the dependencies and this package in a Conda environment:
```
conda install pytorch==1.11.0 -c pytorch
pip install -e .
```
According to the provided steps, after confirming the installation of fairseq, please replace the code in the installed environment with the code from the `fairseq` and `fairseq_cli` folders that we provided.

## Datasets
For the non-translation task, we follows [DiffuSeq](https://github.com/Shark-NLP/DiffuSeq/tree/main) dataset settings.
Prepare datasets and put them under the `datasets` folder. 
Take `datasets/WA/train.jsonl` as an example. We use four datasets in our paper.
|Task|Datasets|TRaiing Sample|Source|Used in Meta-DiffuB|
|:---|:---|:---|:---|:---|
|Open-domain Dialogue|Commonsense Conversation|3382k|[CCM](https://github.com/thu-coai/ccm)|[download](https://drive.google.com/drive/folders/1exENF9Qc5UtXnHlNl9fvaxP3zyyH32qp)|
|Question Generation|Quasar-T|117k|[OpenQA](https://github.com/thunlp/OpenQA)|[download](https://drive.google.com/drive/folders/122YK0IElSnGZbPMigXrduTVL1geB4wEW)|
|Text Simplification|Wiki-Auto|677k|[Wiki-auto](https://github.com/chaojiang06/wiki-auto)|[download](https://drive.google.com/drive/folders/1BlWtD1UbnL_ef06Riq-gABlL0Zb50s-d)|
|Paraphrase|Quora Question Pairs|144k|[Kaggle](https://www.kaggle.com/c/quora-question-pairs)|[download](https://drive.google.com/drive/folders/1BHGCeHRZU7MQF3rsqXBIOCU2WIC3W6fb)|

For the translation task, we follow the [instructions of Fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/translation#iwslt14-german-to-english-transformer) to preprocess the translation datasets. Then we adopt [knowledge distillation](https://github.com/facebookresearch/fairseq/tree/main/examples/nonautoregressive_translation#knowledge-distillation) using Transformer models trained on the same datasets. To binarize the distilled and tokenized datasets, run following command (take the IWSLT14 De-En dataset as an example):
```
fairseq-preprocess \
    --source-lang de --target-lang en \
    --trainpref {PATH-TO-YOUR-DATASET}/train \
    --validpref {PATH-TO-YOUR-DATASET}/valid \
    --testpref {PATH-TO-YOUR-DATASET}/test \
    --destdir data-bin/iwslt14_de_en_distill \
    --joined-dictionary \
    --workers 20
```

## Training, inference, and Evaluation
All training, inference, and evaluation scripts are located in the `{model_type}/scripts` directory. For example, to train Meta-DiffuB-Difformer on the QQP dataset, simply run:
```bash
bash scripts/qqp/train.sh
```
To run inference and evaluate Meta-DiffuB-Difformer on the QQP dataset, run:
```bash
bash scripts/qqp/evaluate.sh
```

For Meta-DiffuB-DiffuSeq, a different approach is required. Instead of bash scripts, Jupyter Notebook files are used for training, inference, and evaluation. Specifically:
- To train Meta-DiffuB-DiffuSeq, execute `scripts/Train.ipynb` in Jupyter Notebook.
- To run inference, execute `scripts/Inference.ipynb` in Jupyter Notebook.
- To evaluate the model, execute `scripts/Evaluate.ipynb` in Jupyter Notebook.

You can modify the parameters in the `.ipynb` files (such as the dataset) to fit your specific usage scenario.


## Baseline Model Reference
The other S2S-Diffusion models' code we run for experiments.
- [Diffuseq](https://github.com/Shark-NLP/DiffuSeq)
- [SeqDiffuSeq](https://github.com/Yuanhy1997/SeqDiffuSeq)
- [Dinoiser](https://github.com/yegcjs/DINOISER)
- [Difformer](https://github.com/zhjgao/difformer/tree/main)
- [BG-DiffuSeq](https://github.com/ZetangForward/Bridge_Gap_Diffusion/tree/main)
- [RDM](https://github.com/HKUNLP/reparam-discrete-diffusion/tree/main)
- [LD4LG](https://github.com/justinlovelace/latent-diffusion-for-language)

<!--
**metabeta-diffusion/metabeta-diffusion** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
