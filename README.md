# Estimating environmental impacts of food products

---

## Overview

This repository contains the code used to estimate the environmental impacts (greenhouse gas emissions, land use, biodiversity loss, eutrophication potential, water use) of food products sold in supermarkets. The pipeline proceeds in five main stages:

1. **Generate embeddings** — encode product names and ingredient lists using a sentence-transformer model (`all-mpnet-base-v2`)
2. **Classify food products** — train and apply Random Forest / Neural Network classifiers to assign products to 104 ingredient categories
3. **Adjust LCA data for sourcing** — build trade-weighted sourcing matrices from FAOSTAT production and trade data, and reweight per-ingredient LCA values by country of origin
4. **Estimate ingredient composition** — infer the percent composition of each ingredient using back-of-pack labelling and interpolation
5. **Calculate environmental impacts** — combine composition estimates with life-cycle assessment (LCA) data, adjusted for trade-weighted sourcing

---

## System Requirements

### Software dependencies

**Python 3.12.9** (ML pipeline and trade data processing):
- `pandas` 2.2.3, `numpy` 1.26.4, `scipy` 1.15.1
- `scikit-learn` 1.6.1, `joblib` 1.4.2
- `matplotlib` 3.10.0, `seaborn` 0.13.2
- `tensorflow` 2.14.0, `sentence-transformers` 3.3.0
- `jupyter` 1.1.1, `jupyterlab` 4.3.4 
**R 4.4.2** (composition estimation and impact calculation):
- `ggplot2`, `plotly`, `cowplot`, `tidyr`, `plyr`, `dplyr`, `readr`, `ggrepel`, `stringr`, `stringi`, `reshape2`, `matrixStats`, `dismo`, `parallel`

### Operating systems tested on
- macOS Tahoe 26.1

### Non-standard hardware
No non-standard hardware is required

---

## Installation Guide

### Python environment

```bash
# Clone the repository
git clone https://github.com/shrutijain90/food_product_impacts.git
cd food_product_impacts

# Create and activate a conda environment
conda create -n prod_imp python=3.12.9
conda activate prod_imp

# Install dependencies
conda install pandas=2.2.3 numpy=1.26.4 scipy=1.15.1 scikit-learn=1.6.1 joblib=1.4.2 matplotlib=3.10.0 seaborn=0.13.2 jupyterlab=4.3.4

# Install pip-only packages
pip install tensorflow==2.14.0 sentence-transformers==3.3.0
```

### R environment

Install required packages from within R:

```r
install.packages(c("ggplot2", "cowplot", "tidyr", "plyr", "dplyr", "readr",
                   "ggrepel", "stringr", "stringi", "reshape2", "matrixStats",
                   "dismo", "plotly"))
```

**Typical install time:** approximately 10–20 minutes on a standard desktop computer.

---

## Demo

The `demo/` folder provides data at each stage of the pipeline for a sample of 500 food products, along with required data and modelling inputs. 

| File/Folder | Description |
|-------------|-------------|
| `demo/products.csv` | Uncategorised product data (sample from Open Food Facts) |
| `demo/product_categorized.csv` | Products after ML classification |
| `demo/trained_models/` | Pre-trained Neural Network classifier models (one per food group) |
| `demo/data_inputs/` | Lookup files required by the R scripts (search word lists, LCA data, country groupings) |
| `demo/product_impacts.csv` | Expected output of Step 2, provided for reference |

### Running the demo

The two steps can be run independently using the provided input files. Before running either step, update the file names and paths in the scripts.

**Step 1 — Classify products (Python):**

Takes `demo/products.csv` and models in `demo/trained_models/` as input.
```bash
python -m product_impacts.ml_pipeline.generate_embeddings
python -m product_impacts.ml_pipeline.make_predictions
```
Expected output: a CSV with categorised products. Expected run time: ~5 minutes.

**Step 2 — Estimate composition and calculate impacts (R):**

Takes `demo/product_categorized.csv` and lookup files in `demo/data_inputs/` as input. Use the provided `product_categorized.csv` directly (rather than the output of Step 1) to ensure variable names match.
```r
source("product_impacts/impacts_cal/1.0_Estimating_Composition_2025.07.27.R")
source("product_impacts/impacts_cal/3.0_Calculating_Impacts_2025.08.08.R")
```
Expected output: a CSV with estimated environmental impacts per 100 g (see `demo/product_impacts.csv` for reference). Expected run time: ~10 minutes.

---

## Instructions for Use

### Input data

The full pipeline requires the following additional datasets (publicly available):

| Dataset | Description | Source |
|---------|-------------|--------|
| Open Food Facts | Product database with names and ingredients | [world.openfoodfacts.org/data](https://world.openfoodfacts.org/data) (CC BY-SA) |
| FAOSTAT production & trade | Crop production and trade matrices by country/year | [fao.org/faostat](https://www.fao.org/faostat/en/#data) (open access) |

### Running the full pipeline

**1.** Download the datasets listed above and update all paths and pathnames in each script.

**2.** Run in order:

```
ml_pipeline/open_food_facts.ipynb                               # translate non-English products to English
ml_pipeline/generate_embeddings.py                              # encode product text with BERT
ml_pipeline/make_predictions.py                                 # classify products using pre-trained models
↓
impacts_pipe_io/clean_ingredients_data.ipynb                    # clean ingredients text
impacts_pipe_io/get_processing_factors.py                       # calculate processing factors for oil and sugar crops
impacts_pipe_io/get_trade_data.py                               # harmonize trade data from FAOSTAT
impacts_pipe_io/trade_lca_reweight.py                           # adjust LCA values for country of origin using trade matrices
↓
impacts_cal/1.0_Estimating_Composition_2025.07.27.R             # estimate ingredient percent composition
impacts_cal/3.0_Calculating_Impacts_2025.08.08.R                # calculate per-product environmental impacts
↓
impacts_pipe_io/impacts_pipeline_results.ipynb                  # figures and summary statistics
```

**Note on model training:** The pre-trained classifier models in `demo/models/` are sufficient to run the full pipeline above. Re-training from scratch requires a set of labelled food products (code in `ml_pipeline/train_models.py` and `product_cat/`).

---

## License

This code is released under the [MIT License](LICENSE) and is fully open source — available for use, reuse, repurposing, and dissemination.

---
