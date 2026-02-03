# AutoLabeler - Few-Shot Classifier

AutoLabeler is a few-shot image classifier based on prototypes, combining meta-learning adaptability and a little bit of the concept of “Wisdom of the Crowd”. It is designed for the semi-automatic annotation of a large dataset from which the user already knows what is expected to be found. By labeling 15 to 20 images per class, the algorithm can expand those annotations to the whole dataset, improving efficiency.

It uses five pretrained CNNs (ImageNet) and a voting system based on the similarity of features and prototypes.

## Project Structure

    autolabeler/
    │
    ├── autolabeler/
    │ ├── cli.py # Main pipeline entry point
    │ ├── embedder.py # Image loading and embedding
    │ ├── models.py # CNN backbone wrappers
    │ ├── prototypes.py # Prototype computation
    │ ├── classify.py # Cosine similarity, voting, CSV, output routing
    │ ├── cache.py # Prototype cache (save/load)
    │ ├── eval.py # Evaluation against divided data to check the performance for your task
    │ └── io_utils.py # Directory and file utilities
    │
    ├── requirements.txt
    └── README.txt


## Input Data Format


Supported extensions = ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"

1) Support directory (few-shot labeled data)
    
        - support/
        - class_A/
        │   ├── img1.png
        │   └── img2.png
        ├── class_B/
        │   └── img1.png
        └── ...

- Folder name = class label
- Minimum recommended per class = 10 images

2) Query directory (unlabeled images)
        
        query/
        ├── img_001.png
        ├── img_002.png
        └── subfolder/
            └── img_003.png

- Recursive scanning by default

## Running the Auto-Labeling Pipeline

Run all commands from the repository root.

Basic run (first time or after changes):

    python -m autolabeler.cli \
      --support_dir /path/to/support \
      --query_dir /path/to/query \
      --output_dir /path/to/output \
      --recompute_prototypes

Normal run (reuses cached prototypes):
    
    python -m autolabeler.cli \
      --support_dir /path/to/support \
      --query_dir /path/to/query \
      --output_dir /path/to/output

Important parameters:
- --threshold_x : minimum to be considered valid
- --min_votes  : minimum number of models that must agree
- --image_size : input size for CNNs (default: 224)

## Output Structure

    output/
    ├── ok/
    │   ├── class_A/
    │   └── class_B/
    └── not_sure/
        ├── class_A/
        └── class_B/

- ok/       → predictions confirmed by ensemble voting according to the parameters
- not_sure/ → predictions decided by fallback

A CSV file is also generated:

results.csv

Containing:
- per-model predictions and scores
- final prediction
- decision reason (vote or fallback)

## Evaluation with Ground Truth

This function is designed for the user to evaluate if the labeler fits the task well.

Ground truth directory format:

    gabarito/
    ├── class_A/
    │   ├── img_001.png
    │   └── ...
    ├── class_B/
    │   └── ...
    └── ...

Evaluate all predictions:

    python -m autolabeler.eval \
      --gabarito_dir /path/to/gabarito \
      --results_csv results.csv

Evaluate only high-confidence ("ok") predictions:

    python -m autolabeler.eval \
      --gabarito_dir /path/to/gabarito \
      --results_csv results.csv \
      --only_ok

The evaluation reports:
- accuracy
- per-class precision / recall / F1
- confusion matrix
- coverage (when using --only_ok)


## Notes

- Prototype cache must be recomputed if:
  - support images change
  - image_size changes
  - backbone set changes

- This system performs classification, not detection.
- It is ideal for bootstrapping datasets before training
  detection or segmentation models.

## License / Usage

This project is intended for research, experimentation, and
semi-automatic annotation workflows.


