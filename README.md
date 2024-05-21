## MetaBeta-Diffusion
![Image Alt text](/img/Meta_DiffuB.jpg)
Comparison between S2S-Diffusion model (i.e., Diffuseq) and the proposed Meta-DiffuB. The shades of color represent different amounts of noise being imposed. Different from prior works that use a fixed noise, we introduce a novel scheduler-exploiter framework, Meta-DiffuB, which achieves trainable noise scheduling inspired by Meta Exploration. Our Meta-B schedules contextualized noise, enhancing the training and generation of the S2S-Diffusion model, resulting in state-of-the-art (SOTA) performance compared to previous S2S-Diffusion models, as detailed in Section 4.

## Datasets
We follows [DiffuSeq](https://github.com/Shark-NLP/DiffuSeq/tree/main) for the dataset settings.
Prepare datasets and put them under the `datasets` folder. 
Take `datasets/WA/train.jsonl` as an example. We use four datasets in our paper.
|Task|Datasets|TRaiing Sample|Source|Used in MetaBeta-Diffusion|
|:---|:---|:---|:---|:---|
|Open-domain Dialogue|Commonsense Conversation|3382k|[CCM](https://github.com/thu-coai/ccm)|[download](https://drive.google.com/drive/folders/1nDqh-bGte9QfTneCEb8NmCn3rNMJx67H?usp=sharing)|
|Question Generation|Quasar-T|117k|[OpenQA](https://github.com/thunlp/OpenQA)|[download](https://drive.google.com/drive/folders/1VrrUwd09DK9oA29yX96zBMopUUvv0KWF?usp=sharing)|
|Text Simplification|Wiki-Auto|677k|[Wiki-auto](https://github.com/chaojiang06/wiki-auto)|[download](https://drive.google.com/drive/folders/1ASIqRJru9ZwNF95e5ESPsveE7YcpZOjv?usp=sharing)|
|Paraphrase|Quora Question Pairs|144k|[Kaggle](https://www.kaggle.com/c/quora-question-pairs)|[download](https://drive.google.com/drive/folders/150SkknKILNm1H9gnwyUxwDxtrc2p84DJ?usp=sharing)|

## MetaBeta-Diffusion Training
Run `Train.ipynb` in jupyter notebook.

## MetaBeta-Diffusion Inference
Run `Inference.ipynb` in jupyter notebook.

## MetaBeta-Diffusion Evaluate
Run `Evaluate.ipynb` in jupyter notebook.

## Baseline Model Reference
The other S2S-Diffusion models' code we run for experiments.
- [Diffuseq](https://github.com/Shark-NLP/DiffuSeq)
- [SeqDiffuSeq](https://github.com/Yuanhy1997/SeqDiffuSeq)
- [Dinoiser](github.com/yegcjs/DINOISER)

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
