# Uber eats ETA prediction


## Objective
This project aims to tackle datascience / machine learning problem by solving a regression task from Kaggle. To know more about Problem Statement, refer [here](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset/data).


## Public Endpoints for Deployed App

Application is deployed on streamlit cloud [here](https://uber-eta.streamlit.app/).

You can deploy it on your own easily and (possibly) free of charge on cloud. Scroll down to `Docker Playground Cloud Deployment` in `Deployment` section.


## Code Structure / Services
- `data` - Contains raw, cleaned and feature engineered data.
- `model` - contains final trained model.
- `notebooks` - Contains EDA (exploratory data analysis) and model development steps including all preprocessing and evaluation. Once preprocessing steps are defined and model is selected, final code is processed to `src`.
- `src` - Contains frontend services along with champion model training script.
    - `infer` - Load model and process input to get predictions.
    - `preprocess` - Preprocessing steps to transform the input data.
    - `train` - Champion model training script.
- `docker-compose` - Compose file which starts application.


## Deployment
- Local deployment
    - Install Docker. Instructions available [here](https://docs.docker.com/engine/install/). Make sure docker is up and running before proceeding.
    - Install Git. Instruction [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
    - Clone repo and run compose
    ```
    git clone https://github.com/uditmanav17/ML-Projects.git && cd ./ML-Projects && git switch uber-eta && cd ./uber-eta
    docker compose --profile app up
    ```
    - `--profile app` will start on `localhost:8501` and `localhost:8501` ports.

- Docker Playground Cloud Deployment
    - Navigate to [docker playground](https://labs.play-with-docker.com/).
    - Login using your docker account. Click Start. This will direct you to a new page.
    - Click `Add New Instance` on left pane. Then run following commands in terminal -
    ```
    git clone https://github.com/uditmanav17/ML-Projects.git && cd ./ML-Projects && git switch uber-eta && cd ./uber-eta
    docker compose --profile app up
    ```
    - To access application, click on port numbers next to `OPEN PORT` button to visit application.


## Future work/ Improvements
- Experiment with ensemble technique
- Experiment with model stacking