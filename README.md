<h2 align="center"> Awesome Efficient Diffusion Models <div align=center> </a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align="center">

   [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
   [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FNastyMarcus%2FA-Survey-of-Efficient-Diffusion-Models&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
   ![GitHub Repo stars](https://img.shields.io/github/stars/NastyMarcus/A-Survey-of-Efficient-Diffusion-Models)

</h5>

Diffusion models have kickstart a new era in the field of artificial intelligence generative content (AIGC). This repo is a curated list of papers about the latest advancements in efficient diffusion models. **This repo is being actively updated, please stay tuned!**

## 📣 Update News

`[2024-10-19]` We released the repository.

## ⚡ Contributing
We welcome feedback, suggestions, and contributions that can help improve this survey and repository so as to make them valuable resources to benefit the entire community.
We will actively maintain this repository by incorporating new research as it emerges. If you have any suggestions regarding our taxonomy, find any missed papers, or update any preprint arXiv paper that has been accepted to some venue.

If you want to add your work or model to this list, please do not hesitate to [pull requests]([https://github.com/ChaofanTao/autoregressive-vision-survey/pulls](https://github.com/ChaofanTao/autoregressive-vision-survey/pulls)).
Markdown format:

```markdown
* **[Name of Conference or Journal + Year]** Paper Name. [[Paper]](link) [[Code]](link)
```

## 📖 Table of Contents
- [📣 Update News](#-update-news)
- [⚡ Contributing](#-contributing)
- [📖 Table of Contents](#-table-of-contents)
- [⭐ Star History](#-star-history)
- [♥️ Contributors](#️-contributors)
- [👍 Acknowledgement](#-acknowledgement)
- [📑 Citation](#-citation)


### Algorithm
#### Efficient Training
  - ##### Noise Schedule
    - **[ICLR 2021]** Denoising Diffusion Implicit Models. [[Paper]](https://openreview.net/pdf?id=St1giarCHLP)
    - **[ICML 2021]** Improved Denoising Diffusion Probabilistic Models. [[Paper]](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) [[Code]](https://github.com/openai/improved-diffusion)
    - **[Arxiv 2024.07]** Improved Noise Schedule for Diffusion Training. [[Paper]](https://arxiv.org/pdf/2407.03297) 
    - **[EMNLP 2023]** A Cheaper and Better Diffusion Language Model with Soft-Masked Noise. [[Paper]](https://aclanthology.org/2023.emnlp-main.289.pdf) [[Code]](https://github.com/SALT-NLP/Masked_Diffusioin_LM)
  - ##### Score Matching
  - ##### Data-Dependent Adaptive Priors
  - ##### Rectified Flow
#### Efficient Fine-tuning
  - ##### Low Rank Adaptation
#### Efficient Sampling
  - ##### Solver
    - **[NeurIPS 2021]** Diffusion Normalizing Flow. [[Paper]](https://proceedings.neurips.cc/paper/2021/file/876f1f9954de0aa402d91bb988d12cd4-Paper.pdf)
    - **[NeurIPS 2023]** Gaussian Mixture Solvers for Diffusion Models. [[Paper]](https://papers.nips.cc/paper_files/paper/2023/file/51373b6499708b6fcc38f1e8f8f5b376-Paper-Conference.pdf) [[Code]](https://github.com/Guohanzhong/GMS)
    - **[ICML 2024]** Unifying Bayesian Flow Networks and Diffusion Models through Stochastic Differential Equations. [[Paper]](https://openreview.net/pdf/4d120b565267ca44bc866a8f372f670c5837e719.pdf) [[Code]](https://github.com/ML-GSAI/BFN-Solver)
    - **[NeurIPS 2023]** SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models. [[Paper]](https://openreview.net/pdf?id=f6a9XVFYIo)
    - **[NeurIPS 2022]** DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. [[Paper]](https://arxiv.org/pdf/2206.00927) [[Code]](https://github.com/LuChengTHU/dpm-solver)
    - **[ICLR 2023]** Fast Sampling of Diffusion Models with Exponential Integrator. [[Paper]](https://openreview.net/pdf?id=Loek7hfb46P) [[Code]](https://github.com/qsh-zh/deis)
    - **[ICML 2023]** Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs. [[Paper]](https://proceedings.mlr.press/v202/zheng23c/zheng23c.pdf) [[Code]](https://github.com/thu-ml/i-DODE)
    - **[ICML 2023]** Denoising MCMC for Accelerating Diffusion-Based Generative Models. [[Paper]](https://proceedings.mlr.press/v202/kim23z/kim23z.pdf) [[Code]](https://github.com/1202kbs/DMCMC)
  - ##### Efficient Scheduling
    - 
  - ##### Partial Sampling 
#### Compression
  - ##### Knowledge Distillation
  - ##### Quantization
  - ##### Pruning

### System
#### Optimized Hardware-Software Co-Design
#### Parallel Computing
#### Caching Technique




## Frameworks
<div align="center">

|                                                    | Efficient Training | Efficient Fine-Tuning | Efficient Inference    |
| :-------------------------------------------------------------------- | :------------------: | :---------------------: | :--: |
| Diffusers [[Code](https://github.com/huggingface/diffusers)]            | ✅                   | ✅                     | ✅   |
| DALL-E [[Code](https://github.com/openai/DALL-E)]                       | ❌                   | ❌                     | ✅   |
| OneDiff [[Code](https://github.com/siliconflow/onediff)]                | ❌                   | ❌                     | ✅   |
| LiteGen [[Code](https://github.com/Vchitect/LiteGen)]                   | ✅                   | ✅                     | ✅   |
| InvokeAI [[Code](https://github.com/invoke-ai/InvokeAI)]                | ❌                   | ✅                     | ✅   |
| ComfyUI-Docker [[Code](https://github.com/YanWenKun/ComfyUI-Docker)]    | ❌                   | ✅                     | ✅   |
| Grate [[Code](https://github.com/damian0815/grate)]                     | ❌                   | ✅                     | ✅   |
| Versatile Diffusion [[Code](https://github.com/SHI-Labs/Versatile-Diffusion)]                  | ✅                     | ✅                     | ✅   |
| UniDiffuser [[Code](https://github.com/thu-ml/unidiffuser)]             | ✅                   | ✅                     | ✅   |

</div>

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NastyMarcus/A-Survey-of-Efficient-Diffusion-Models&type=Date)](https://star-history.com/#NastyMarcus/A-Survey-of-Efficient-Diffusion-Models&Date)

## ♥️ Contributors

<a href="https://github.com/NastyMarcus/A-Survey-of-Efficient-Diffusion-Models/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=NastyMarcus/A-Survey-of-Efficient-Diffusion-Models" />
</a>


<!--
## 👍 Acknowledgement
To be continued


## 📑 Citation

Please consider citing 📑 our papers if our repository is helpful to your work, thanks sincerely!

```BibTeX
To be continued
–->
