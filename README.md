# Private email filtering using Homomorphic Encryption

## What is this? What does this mean?

This project does a handful of things. The first being a ML solution to filter out Spam
email with end-to-end encryption, done with Homomorphic encryption, which allows computations
to be performed correctly on encrypted data. Hence, the server can run a predictive AI model
on encrypted text and correctly classify an email, so the server never sees what the contents
of the emails are.

The 'heart' of the project is the whole pipeline managing the Data and Machine Learning. Though I plan to make additional changes:

## Scope & Future Development

Beyond what has already been done, an actual application will be deployed that uses the data and ML pipeline. Initially
I wipped up a quick rest server to 'showcase' that the private filtering actually works. That is,
encrypting data, sending to server, classifying, and sending back for the client to decrypt. Though
this is still purely 'back-end', and very minimal. The goal here will be to get a front end working,
both for a web and mobile client, so that users can actually interact with the project. The 
idea is that they can select an email from the dataset, or even paste their own one in, and the
site with visualise the process of filtering (so encrypting, predicting, sending back, etc..).

This is the scope for now. Later, if I wish to progress the project even further, I can think
of ways of integrating it into real email clients (in real time) for the thing to work, and further
add more 'data engineering-y' pipelines to have an updated and dynamic list of data for the models
to retrain on.

The front end will use React native to get an end-to-end application.

## ML info

Support Vector Machine (SVM) and Logistic Regression (LR) models are used to predict email classifications. The data is processed with both a Bag-of-words model and a Term-Frequency-Inverse-Frequency (TF-IDF) model for tokenisation of the text. Further we deploy PCA and SVC dimensionality reduction models to have 'efficient' variants for faster inference (since encrypted inference can be slow).

Zama, TenSEAL and Paillier Homomorphic encryption libraries were tested to compare results and showcase functionality. Currently, our 'efficient' Tenseal variant is 'deployed' for actual demo showcase.

Our models are ~98% accurate, and ~97% accurate (for pca/svc optimised models). Inference is much slower than plaintext. However, our fastest Tenseal model performs encrypted inference at 3.7ms per email, which theoretically allows us to perform ~184 million encrypted predictions per day on my laptop alone (when utilising all 8 cpu cores).

Real world deployment of end-to-end private spam filtering at scale is very feasable with these numbers. 6 of my laptops would be able to handle the load of 100million users. Of course this is heavily extrapolated and 'napkin math' (I don't know what constraints would come my way here), though I think this is a good indicator of the potential of this tech.

Models are trained and saved into their respective folders (`\ts_compat_models` and `\zama_models`) to be
later loaded and used.

Preprocessing is handled in the `\data_functionality` folder.

## Running the project

If you wish to download and run, you just need `python3` installed (this project used `python3.11`, but any should work). The dataset is not included in the git repo (too big), so you'd need to find your own and link it to the project. This means linking the pre-processing steps to your new data (it assumes your using json), and running the required scripts to train and save your models. These are in `\execute_scripts`, and each have their own 'functionality' with respect to the project. Running `model_setup.py` is most important here.

Download the required packages from pip (as your code throws errors). Not sure if there's a way to automatically install them all before running (prob is, not interested in doing that now).