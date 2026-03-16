# Session 4

Implement a Deep Learning App. Train a model to classify text reviews between positive and negative and deploy a web app to predict on user reviews.
Then, deploy the model using Google Cloud Run.

## Dataset

We will use the Yelp Review polarity dataset ([link](https://www.kaggle.com/datasets/irustandi/yelp-review-polarity), [direct download](https://www.kaggle.com/api/v1/datasets/download/irustandi/yelp-review-polarity)).

The dataset has two labels. "1" corresponds to negative reviews, and "2" to positive reviews.

## Installation
### With Conda
Create a conda environment by running

```bash
conda create --name aidl-session4 python=3.10
```
Then, activate the environment
```bash
conda activate aidl-session4
```
and install the dependencies
```bash
pip install -r requirements.txt
```

**Note:** it is important to create a new conda environment, and not re-use previous ones from other sessions.

### With venv
Create a virtual environment by running

```bash
python -m venv .venv
source .venv/bin/activate
```
and install the dependencies
```bash
pip install -r requirements.txt
```

## Tasks

### Task 1: Build the app

Complete the code in the `app/` folder to get the web app running locally. A dummy (untrained) checkpoint is already provided, so you can test your app right away. Predictions will be random, but you'll see the app working!

1. Complete `app/main.py`: load the model from the checkpoint and implement the prediction function. The model (`app/model.py`) is already provided.
2. Run the app:
```bash
cd session-4/app
python main.py
```
4. Go to http://localhost:8080 and try submitting a review. The predictions will be random since the model is untrained — that's expected!

### Task 2: Train the model

Now let's train a real model so the app gives meaningful predictions.

1. Download the [dataset](https://www.kaggle.com/api/v1/datasets/download/irustandi/yelp-review-polarity) and extract it at the root of the repository (you should have a `yelp_review_polarity_csv/` folder).
2. Complete the TODOs in `train.py`: implement the training loop, evaluation loop, and dataset split.
3. Run the training:
```bash
python session-4/train.py
```
4. This will save a trained checkpoint to `app/state_dict.pt`, replacing the dummy one.
5. Run the app again and see the difference — predictions should now be meaningful!

### Task 3: Deploy to Google Cloud Run (optional)

Deploy your trained app to the cloud so anyone can access it.

#### Prerequisites

Make sure you have the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and configured:

```bash
gcloud auth login
gcloud config set project <your-project-id>
```

#### One-time IAM setup

Before deploying for the first time, run this command to grant the necessary permissions to Cloud Build:

```bash
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
  --member="serviceAccount:$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.builder"
```

#### Deploy

From the `app/` directory (make sure your trained `state_dict.pt` is inside):

```bash
cd session-4/app
gcloud run deploy sentiment-analysis \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi
```

The first deploy takes ~10-15 minutes (it builds a container in the cloud). Subsequent deploys are faster.

Once deployed, you will get a URL like `https://sentiment-analysis-XXXXX.europe-west1.run.app`.

#### Cleanup

When you are done, you can delete the service to avoid any charges:

```bash
gcloud run services delete sentiment-analysis --region europe-west1
```
