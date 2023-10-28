# Calling Rust from Python

> A Gentle Introduction to PyO3

## Setup

```bash
git clone git@github.com:murilo-cunha/intro-to-pyo3.git
cd intro-to-pyo3
```

### Dev Containers (VSCode)

> Make sure you have [Docker installed](https://docs.docker.com/engine/install/) and the [Dev Containers VSCode extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

If you are on VSCode and don't have Rust installed and/or Python 3.12 installed, dev containers is a good option. Just open the project folder in VSCode and click on the green button at the bottom left corner of the screen. Make sure that the Docker daemon is running.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


### PDM

If you are using [PDM](https://pdm.fming.dev/latest/), make sure to include `pip` in your virtual environment.

```bash
pdm install
pdm run python -m ensurepip
```
