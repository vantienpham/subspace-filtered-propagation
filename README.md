# Subspace-Filtered Error Propagation in Deep Low-Rank Compressed Networks

The paper bounds how the error of post-training low-rank compression propagates
through a deep network. A rank-$r$ layer transmits only the part of an incoming
perturbation that lies in its $r$-dimensional input subspace (the subspace
transmission $\kappa_l$), and the nonlinearity passes only part of what remains
(the activation gating $\theta_l$). The scripts here measure every quantity in the
resulting bound on VGG-16, compare calibration schemes on ResNets and on two
language models, and measure how far the propagated error is from general
position.

Every number, table and figure in the paper is generated from `exp/results/*.json`
by `exp/make_figs.py`, and `exp/verify_claims.py` re-checks the quantitative
claims of the text against those files.

## Contents

| file | purpose | paper |
|---|---|---|
| `exp/lrc.py` | rank-$r$ compressors: covariance-weighted (whitened) truncated SVD, diagonal weighting, data-agnostic SVD, and the normal-equation form used by all calibration schemes | Sec. II |
| `exp/data.py` | ImageNet validation loaders; calibration, held-out and evaluation images are disjoint | |
| `exp/e2_propagation.py` | VGG-16: $\delta_l,\gamma_l,\kappa_l,\bar\kappa_l,c_l,\theta_l,\varepsilon_l$ and the bounds, on calibration and held-out images | Fig. 1, Table II, Sec. IV-A |
| `exp/e3_imagenet.py` | ResNet-18/34/50 top-1 under one-shot, sequential and compensating calibration | Table III |
| `exp/e4_llm.py` | WikiText-2 perplexity of Qwen2.5-0.5B and Llama-3.2-1B-Instruct | Table IV |
| `exp/e5_alignment.py` | alignment of the propagated error with the retained subspace on transformer linear layers | Sec. IV-A, IV-C |
| `exp/merge_e4.py` | merges the per-model (or per-budget) E4 outputs into `results/e4.json` | |
| `exp/make_figs.py` | builds `fig_propagation.pdf`, `tab_e{2,3,4}.tex` and `numbers.tex` | all |
| `exp/verify_claims.py` | re-checks the claims of the text | |
| `exp/check_theory.py` | numerical check of Lemma 1, Proposition 2 and Eq. (2) on random instances (numpy only) | Sec. II-III |
| `exp/job.sh` | Slurm wrapper | |
| `exp/results/` | the raw results behind the paper | |

The generated `fig_propagation.pdf`, `tab_e*.tex` and `numbers.tex` are included at
the top level.

## Compressors and calibration schemes

All schemes are the rank-$r$ minimizer of a least-squares objective,
`lr_normal_eq(R, C, r)` $=B^{-1}[B^{-\top}R]_r$ with $B^\top B=C+\tau I$
(Cholesky), differing only in the statistics:

| scheme | $C$ | $R$ |
|---|---|---|
| one-shot | $X^\top X$ | $CW$ |
| sequential | $\widehat X^\top\widehat X$ | $CW$ |
| compensating | $\widehat X^\top\widehat X$ | $\widehat X^\top XW$ |

$X$ is the input of the layer in the original network and $\widehat X$ the input
it receives in the network whose earlier layers are already compressed;
$\tau$ is `ridge` times the mean diagonal entry of $C$ (default $10^{-6}$).
Convolutions are compressed as the reshaped kernel
$W\in\mathbb R^{Ck^2\times C'}$ acting on im2col patches. Language models are
compressed block by block on a cached activation stream: every linear map of
every block (the query, key, value and output projections and the three MLP
matrices) is replaced by its rank-$r$ estimate, with $r$ chosen so that a
fraction $\rho$ of its parameters is kept.

## Requirements

One GPU (an A100 40 GB or a V100 32 GB is enough), Python 3.10+, and the
packages in `requirements.txt` (tested with torch 2.5.1, torchvision 0.20.1,
transformers 5.15.0, matplotlib 3.11.1).

Data and weights:
- the ImageNet (ILSVRC-2012) validation set in class-folder layout (1000 wnid
  folders); set `IMAGENET_VAL` to its location;
- torchvision weights `IMAGENET1K_V1` for VGG-16 and ResNet-18/34/50;
- from the HuggingFace Hub: `Qwen/Qwen2.5-0.5B`, `meta-llama/Llama-3.2-1B-Instruct`
  (gated: accept the license first) and the dataset `Salesforce/wikitext`
  (`wikitext-2-raw-v1`). The scripts read the HuggingFace cache (`HF_HOME` or
  `HF_HUB_CACHE`) and run offline, so download these once beforehand, e.g.
  `huggingface-cli download Salesforce/wikitext --repo-type dataset`.

## Reproducing

From the repository root (`mkdir -p logs` first when using Slurm; without Slurm,
run the same `python -u exp/...` commands directly):

```bash
sbatch -J e2   --time=0-02:00:00 exp/job.sh exp/e2_propagation.py --rhos 0.5 0.3 0.15 --batches 16 --heldout-sets 4
sbatch -J e2c2 --time=0-02:00:00 exp/job.sh exp/e2_propagation.py --rhos 0.5 0.3 0.15 --batches 16 --heldout-sets 4 --calib-offset 128 --out results/e2_calib2.json
sbatch -J e3   --time=0-05:00:00 exp/job.sh exp/e3_imagenet.py --models resnet18 resnet34 resnet50 --rhos 0.7 0.6 0.5 0.4 --eval-per-class 10 --calib 256
sbatch -J e4q  --time=0-04:00:00 exp/job.sh exp/e4_llm.py --models Qwen/Qwen2.5-0.5B --rhos 0.9 0.8 0.6 --calib-seq 32 --out results/e4_part_qwen.json
sbatch -J e4l  --time=0-05:00:00 exp/job.sh exp/e4_llm.py --models meta-llama/Llama-3.2-1B-Instruct --rhos 0.9 0.8 0.6 --calib-seq 32 --out results/e4_part_llama.json
sbatch -J e5   --time=0-04:00:00 exp/job.sh exp/e5_alignment.py --models Qwen/Qwen2.5-0.5B meta-llama/Llama-3.2-1B-Instruct --rhos 0.9 0.8 0.6
# damping sweeps of compensating calibration
for r in 1e-4 1e-2; do
  sbatch -J rr$r --time=0-00:40:00 exp/job.sh exp/e3_imagenet.py --models resnet18 --rhos 0.4 --schemes compensated --ridge $r --out results/e3_ridge_$r.json
  sbatch -J rq$r --time=0-01:00:00 exp/job.sh exp/e4_llm.py --models Qwen/Qwen2.5-0.5B --rhos 0.9 --schemes compensated --ridge $r --out results/e4_ridge_$r.json
done
python exp/merge_e4.py
python exp/make_figs.py
python exp/verify_claims.py
```

Side runs must pass `--out`, otherwise they overwrite the main results file.
Approximate times on one A100 40 GB: E2 10 min per run, E3 40 min, E4 45 min
(Qwen) and 1.5 h (Llama; it can be split by budget with `--rhos`, which
`merge_e4.py` handles), E5 25 min.

Sanity checks: dense ResNet-18 top-1 on the evaluation split is 69.74, and dense
Qwen2.5-0.5B WikiText-2 perplexity is 13.07 at context length 2048.

## License

MIT, see `LICENSE`. ImageNet, the model weights and WikiText-2 are not included
and are subject to their own licenses.
