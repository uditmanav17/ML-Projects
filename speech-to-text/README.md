# Speech to text transcription


## Objective
This project aims to transcribe audio file using [openai-whisper](https://github.com/openai/whisper) model. Users can also enter a valid YouTube URL for transcription.


## Public Endpoints for Deployed App

Application is deployed on streamlit cloud [here](https://transcribe-whisper.streamlit.app/).

You can deploy it on your own easily and (possibly) free of charge on cloud. Scroll down to `Docker Playground Cloud Deployment` in `Deployment` section.


## Code Structure / Services
- `app` - Complete application code built in streamlit.
- `packages` - List of linux dependencies required to deploy code on streamlit cloud.
- `docker-compose` - Compose file which starts application.


## Deployment
- Local deployment
    - Install Docker. Instructions available [here](https://docs.docker.com/engine/install/). Make sure docker is up and running before proceeding.
    - Install Git. Instruction [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
    - Clone repo and run compose
    ```
    git clone https://github.com/uditmanav17/ML-Projects.git && cd ./ML-Projects
    git switch whisper && cd ./speech-to-text
    docker compose --profile app up
    ```
    - `--profile app` will start on `localhost:8501` and `localhost:8501` ports.

- Docker Playground Cloud Deployment
    - Navigate to [docker playground](https://labs.play-with-docker.com/).
    - Login using your docker account. Click Start. This will direct you to a new page.
    - Click `Add New Instance` on left pane. Then run following commands in terminal -
    ```
    git clone https://github.com/uditmanav17/ML-Projects.git && cd ./ML-Projects
    git switch whisper && cd ./speech-to-text
    docker compose --profile app up
    ```
    - To access application, click on port numbers next to `OPEN PORT` button to visit application.


## Future work/ Improvements
- Add download transcription as subtitles `.srt` file.
