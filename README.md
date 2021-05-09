The open source of paper: mining trajectory patterns with point-of-interest and behavior-of-interest

Model
    Model.py: A improved transformer model we use to precdict user's age and gender, which pretrain the trajectory(check-ins) by word2vec.
    lightgbm.py and catboost.py test the embedding method by different models.

load_data
    data_loader.py: Export the word2vec model and save the embedding result.

eval
    plot the result of classifier and the log of training process. CLuster the trajectory.
    Part of result figures.

data
    The real-world dataset we use.
