# Bandits with Preference Feedback: A Stackelberg Game Perspective
[<img src="https://img.shields.io/badge/license-Apache2.0-blue.svg">](https://github.com/luchris429/purejaxrl/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

We recommend using a virtual environment to manage dependencies such as `virtualenv` or `conda`.

## Preprocessing
If you would like to run experiments on the `Yelp` restaurant review dataset, first add your OpenAI API Key in the preprocessing script and then run it with the following command: 
```
python src/create_embeddings.py
```
This script saves the preprocessed data in the `data/yelp_aggregates` directory.
For more information on how to use the OpenAI API, please refer to the [OpenAI API documentation](https://platform.openai.com/docs/guides/embeddings).

## Training

### Logistic Feedback
To train the model(s) with logistic feedback, run:
```
python run_experiment_logistic.py --dir /path/to/experiment/directory
```
Configuration parameters are expected to be in a file named `config.yaml` in the experiment directory.
We provide the configurations used in the paper in the `data/logistic_feedback/` directory.

If you prefer to use only one of the algorithms, you can specify the algorithm by using the `--algo` flag.
For example, to run the `LGP-UCB` algorithm add `--algo LGPUCB` to the command.


### Comparison Feedback
To train the model(s) with comparison feedback, run:
```
python run_experiment_preference.py --dir /path/to/experiment/directory
```
Configuration parameters are expected to be in a file named `config.yaml` in the experiment directory.
We provide the configurations used in the paper in the `data/preference_feedback/` directory.

## Evaluation

To reproduce the figures and tables in the paper, run the `notebooks/visualize_logistic.ipynb`
and `notebooks/visualize_preference.ipynb` notebooks respectively.

# References and Contact
With any question about the code, please reach out to [Barna Pásztor](mailto:barna.pasztor@ai.ethz.ch) and,
if you find our code useful for your research, cite our work as follows:
```bibtex
@misc{pasztor2024banditspreferencefeedbackstackelberg,
      title={Bandits with Preference Feedback: A Stackelberg Game Perspective}, 
      author={Barna Pásztor and Parnian Kassraie and Andreas Krause},
      year={2024},
      eprint={2406.16745},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.16745}, 
}
```
