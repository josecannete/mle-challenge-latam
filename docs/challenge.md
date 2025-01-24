# LATAM Challenge

## Installation

The project is structured as package and is installable like that.

To install use:
```bash
pip install .
```

Or, to install optional dependencies:
```bash
pip install ".[dev,test]"
```

## Chosen Model

The model chosen was the Logistic Regression one. Both models seems to have similar performance but Logistic Regression is simpler, easier to understand and to explain. In this conditions, I prefer it over XGBoost.

## New tests

Some new test were added. These are described in the following [PR](https://github.com/josecannete/mle-challenge-latam/pull/5).

## CI

The CI workflow was incorporated [very early](https://github.com/josecannete/mle-challenge-latam/pull/1) and evolved [later on](https://github.com/josecannete/mle-challenge-latam/blob/main/.github/workflows/cd.yml).

It is very simple for now:
- It uses ruff to lint and formatting of the code.
- It runs the model tests and api tests.

In the future I would like to:
- Better control which files require and which don't require a CI run.
- Add other kind of lints and checks, e.g: security ones.

## CD

Two different mechanisms of Continuous Delivery were implemented. 

### Vertex AI Model Registry + Endpoints

My first idea was to use Model Registry + Endpoints. It is suitable for machine learning in general due to its monitoring features, evaluation, among others.

I implemented it in a way that:
- There is a single Vertex Endpoint.
- Each time the CD Workflow runs, a new model version is uploaded to the Model Registry.
- After that, the new version of the model is deployed, sending all the traffic to it.

A problem with this implementation (for the case of this challenge) is that there is no easy way to allow unauthenticated requests. I managed to run the stress tests with little modifications to the stress test (changing the endpoint to a [particular one](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints/rawPredict) and including the auth header to it).

That makes it difficult to share with you for the purpose of this challenge. I left the cd workflow [here](https://github.com/josecannete/mle-challenge-latam/blob/main/.github/workflows/cd.yml) as I find it valuable, but implemented a simpler one with Cloud Run.


### Cloud Run

This one deploys the model to Cloud Run, it is very simple and can be seen [here](https://github.com/josecannete/mle-challenge-latam/blob/main/.github/workflows/cd-cloudrun.yml). This is the one enabled in the repo. The URL in the makefile is deployed using this mechanism.