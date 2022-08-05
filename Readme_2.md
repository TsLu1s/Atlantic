# INFOS

## Index
0. [Initial configuration](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#0-initial-configuration)  
0.1. [Setting Git](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#01-initial-configuration)  
0.2. [Setting Docker](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#02-initial-configuration)  
0.3. [Setting pyenv](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#03-initial-configuration)  
0.4. [Setting Python](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#04-initial-configuration)  
0.5. [Setting virtualenv](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#05-initial-configuration)  
1. [Overview](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#1-overview)
2. [Cloning INFOS Github project](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#2-cloning-infos-github-project)
3. [Creating a virtual environment](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#3-creating-a-virtual-environment)
4. [Installing dependencies](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#4-installing-dependencies)  
5. [Defining environment variables](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#5-defining-environment-variables)   
5.1 [Setting the environment variables for the current shell session only](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#51-setting-the-environment-variables-for-the-current-shell-session-only)    
5.2 [Setting the environment variables permanently](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#52-setting-the-environment-variables-permanently)   
6. [Initializing containers](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#6-initializing-containers)  
7. [Running the pipelines](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#7-running-the-pipelines)    
7.1 [Visualizing the pipeline](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#71-visualizing-the-pipeline)    
7.2 [Changing pipeline parameters](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#72-changing-pipeline-parameters)    
8. [Model versions](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#8-model-versions)   
8.1 [Comparing model versions](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#81-comparing-model-versions)    
8.2 [Moving model version to production](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#82-moving-model-version-to-production)  
9. [Visualizing the monitoring dashboards](https://github.com/EPMQ/INFOS/blob/dev-amatta/README.md#9-visualizing-the-monitoring-dashboards)    

## **0. Initial configuration** 

### **0.1. Setting Git**

Git is software for tracking changes in any set of files, usually used for coordinating work among programmers collaboratively developing source code during software development.

```bash
sudo yum groupinstall "Development Tools"
sudo yum install gettext-devel openssl-devel perl-CPAN perl-devel zlib-devel libcurl-devel expat-devel
wget https://github.com/git/git/archive/v2.33.0.tar.gz -O git.tar.gz
tar -zxf git.tar.gz
cd git-*
sudo yum install asciidoc xmlto
make all doc
sudo make install install-doc install-html
cd ..
rm git.tar.gz
cd ~
```

### **0.2. Setting Docker**

Docker is an open-source software project automating the deployment of applications inside software containers.

```bash
sudo yum install -y yum-utils
sudo yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```

#### **0.2.1. Setting Docker Compose**

Docker Compose is a tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application’s services. Then, with a single command, you create and start all the services from your configuration.

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo curl \
    -L https://raw.githubusercontent.com/docker/compose/1.29.2/contrib/completion/bash/docker-compose \
    -o /etc/bash_completion.d/docker-compose
source ~/.bashrc
```
### **0.3. Setting pyenv**

`pyenv` lets you easily switch between multiple versions of Python. It's simple, unobtrusive, and follows the UNIX tradition of single-purpose tools that do one thing well.

```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src

sed -Ei -e '/^([^#]|$)/ {a \
export PYENV_ROOT="$HOME/.pyenv"
a \
export PATH="$PYENV_ROOT/bin:$PATH"
a \
' -e ':a' -e '$!{n;ba};}' ~/.bash_profile

echo 'eval "$(pyenv init --path)"' >> ~/.bash_profile

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init --path)"' >> ~/.profile

echo 'eval "$(pyenv init -)"' >> ~/.bashrc

source .bash_profile
```

### **0.4. Setting Python**

This project was developed using [Python (v3.8.10)](https://www.python.org/downloads/release/python-3810/), an interpreted high-level general-purpose programming language. Python is the most suitable programming language for this because it is easy to understand and comes with a large number of inbuilt libraries for Machine Learning and Artificial Intelligence.

```bash
sudo yum install gcc zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel tk-devel libffi-devel xz-devel
pyenv install 3.8.10
pyenv global 3.8.10
```

### **0.5. Setting virtualenv**

`virtualenv` is a tool to create isolated Python environments.
```bash
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
exec "$SHELL"
```

## **1. Overview**

This project was developed by Arthur Matta, Hugo Carvalho, and Rui Ribeiro in the context of a partnership between CCG - Centre for Computer Graphics and INFOS.

This module uses a framework called [kedro (v0.17.3)](https://kedro.readthedocs.io/en/0.17.3/). From its documentation:

```
Kedro is an open-source Python framework for creating reproducible, maintainable and modular data science code. It borrows concepts from software engineering best-practice and applies them to machine-learning code; applied concepts include modularity, separation of concerns and versioning.
```

## **2. Cloning INFOS Github project**

Pull down a full copy of all the repository data that GitHub has at that point in time, including all versions of every file and folder for the project.

```
git clone https://github.com/EPMQ/INFOS.git
````

## **3. Creating a virtual environment**

All project's dependencies will be installed in virtual environment, keeping the specific Python packages in an environment exclusive to this project, instead of system-wide.

```
cd INFOS
pyenv virtualenv 3.8.10 venv
pyenv local venv
pyenv activate venv
```

## **4. Installing dependencies**

All project's dependencies are defined in a file located in `src/requirements.txt`. This file list the required libraries and their respective versions, and can be installed using `pip`:

```
pip install -r src/requirements.txt
```

## **5. Defining environment variables**

In order to succesffuly execute the project, some environment variables (envvars) need to be defined. These are:

- **MLFLOW_TRACKING_URI** (url to the mlflow instance.)
- **MLFLOW_S3_ENDPOINT_URL** (url to the minio instance.)
- **MINIO_ROOT_USER** (username to access the minio repository.)
- **MINIO_ROOT_PASSWORD** (password to access the minio repository.)
- **AWS_ACCESS_KEY_ID** (must have the same value as MINIO_ROOT_USER)
- **AWS_SECRET_ACCESS_KEY** (must have the same value as MINIO_ROOT_PASSWORD)
- **POSTGRES_USER** (username for the postgres database.)
- **POSTGRES_PASSWORD** (password for the psotgres database.)
- **PROJ_VERSION** (Optional: version of the project. Defaults to 1.0)
- **PROJ_NAME** (Optional: name of the project. Defaults to "predict")
- **PROJ_DESCRIPTION** (Optional: description of the project. Defaults to "Predictive module")
- **H2O_URL** (url to the h2o instance.)
- **FLASK_ENV** (environment for the flask app. Possible values are: "development", "testing", "production")
- **FLASK_HOST** (ip to serve the flask app)
- **FLASK_PORT** (port to serve the flask app)
- **GF_SMTP_ENABLED** (whether to enable SMTP support. Required for inviting multiple users to Grafana)
- **GF_SMTP_HOST** (smtp url and port)
- **GF_SMTP_USER** (smtp username)
- **GF_SMTP_PASSWORD** (smtp password)

These variables can be defined directly into your environment by editing you environment variables:

### **5.1 Setting the environment variables for the current shell session only**

#### **5.1.1 Using a .env file**
- Create a text file and copy the content bellow replacing `value` by the respective value.

```bash
export MINIO_CONSOLE_PORT=9001
export MINIO_PORT=9000
export MINIO_ROOT_USER="minio"
export MINIO_ROOT_PASSWORD="minio123"
export AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER}
export AWS_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD}
export MLFLOW_PORT=5000
export MLFLOW_TRACKING_URI="http://127.0.0.1:${MLFLOW_PORT}"
export MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:${MINIO_PORT}"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="postgres"
export MAX_CONNECTIONS=100
export SHARED_BUFFERS="1GB"
export PROJ_VERSION="1.0"
export PROJ_NAME="infos_ccg"
export PROJ_DESCRIPTION="infos_ccg ml project"
export H2O_RELEASE="rel-zipf"
export H2O_VERSION="3.32.1.7"
export H2O_CLUSTER_NAME="H2O_INFOS"
export H2O_IP="127.0.0.1"
export H2O_PORT=54321
export FLASK_ENV="production"
export FLASK_HOST="0.0.0.0"
export FLASK_PORT="5005"
export GF_SMTP_ENABLED="false"
export GF_SMTP_HOST=""
export GF_SMTP_USER=""
export GF_SMTP_PASSWORD=""
```

- Save the file as `.env`.

- Open a shell session and run

```bash
source .env
```

#### **5.1.2 Defining directly in the shell**

- Open a shell session and export each of the envvars listed above, replacing `value` by the respective value:

```bash
$ export MLFLOW_TRACKING_URI=value
```

### **5.2 Setting the environment variables permanently**

#### **5.2.1 For a single user**

- To set the environment variables for a single user, edit the `.bashrc` file

```bash
sudo nano ~/.bashrc
```

- Copy each envvar listed above, replacing `value` by the respective value.

- Save and exit the file. The changes are applied after you restart the shell. To apply the changes during the current session, use the source command

```bash
source ~/.bashrc
```

#### **5.2.1 For all users**

- To set the environment variables for all user, create a `.sh` file in the `/etc/profile.d` folder:

```bash
sudo nano /etc/profile.d/[filename].sh
```

- Copy each envvar listed above, replacing `value` by the respective value.

- Save and exit the file. The changes are applied at the next logging in

## **6. Initializing containers**

This project uses several containers to monitor the pipelines and manage the models. To initialize the containers, it´s required to have [docker](https://docs.docker.com/engine/install/#server) and [docker-compose](https://docs.docker.com/compose/install/) installed. Run the following command to initialize all containers at once:

```
docker-compose up
```

There are two containers that are considered most important. They are Grafana and MLFlow. The former provides dashboards to monitor the pipeline execution. The latter provides functionalities to version and manage models, metrics and parameters.

**NOTE:** The default credentials for Grafana are as follow:

```yaml
username: admin
password: admin
```

## **7. Running the pipelines**

To train/refresh the models, we use a framework called kedro. To run a specific pipeline, execute the command bellow, replacing `target-name` by the desired target to refresh.

```
kedro run --pipeline target-name
```

`target-name` can assume one of the values `days`, `leadtime`, `prodtime` (Production time), `proddelay` (Production delay), `prodwaste` (Raw material waste)

### **7.1 Visualizing the pipeline**

This project includes a plugin for kedro, called kedro-viz, that provides a visual and interative representation of the pipelines by executing the command:

```
kedro viz
```

A window should open on your default browser with the pipelines.

![Kedro Viz UI](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/Kedro_viz_main.PNG)

To access detailed information about pipeline functions click on it:

![Kedro Viz Detailed](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/Kedro_viz_detailed.PNG)

### **7.2 Changing pipeline parameters**

**IMPORTANT**: It´s highly recommended only to edit the parameters under the `default_target`, `monitoring`, and `api` tags. The other tags are configuration parameters that can crash the application if changed.

#### **7.2.1 Through `.yaml` file**

All parameters used by the pipelines are defined in a file located in `conf/base/globals.yml`. Edit this file to change the parameters' values to the desired value.

#### **7.2.2 Through CLI**

To change a parameter through command line interface, use the option `--params param_key_1:param_value_1,param_key_2:param_value_2`, for example:

```
kedro run --pipeline days --params h2o.port:8500
```

The command above will execute the `days` pipeline with the parameter `h2o.port` set to `8500`.

If any of the parameters key and/or value contains spaces, wrap the whole option contents in quotes:

```
kedro run --pipeline target --params "key1:value with spaces,key2:value2"
```

Since key-value pairs are split on the first colon, values can contain colons, but keys cannot. This is a valid CLI command:

```
kedro run --pipeline days --params api.url:http://161.97.167.164:81/restBasic/rest/CCGservice
```

## **8. Model versions**

The MLFLow instance implemented in this project allows to version the models trained/refreshed and log the parameters, metrics, and artifacts associated with it. Open the MLFlow UI by acessing its address (specified by the MLFLOW_TRACKING_URI environment variable defined on topic 3.). The `Experiments` tab displays all the runs of the pipeline and the `Models` tab displays a list of the named models generated by those runs.

### **8.1 Comparing model versions**

To compare multiple model versions, go to the `Models` tab and click on the named model you want to compare, e.g., `days`, as show in the image bellow:

![Select target](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/MLflow_select_target.PNG)

A new window will be opened listing all the versions of the selected model. Select the versions to compare and click the `Compare` button:

![Compare versions](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/MLflow_select_to_compare.PNG)

A detailed comparison of the selected versions will be displayed, including the parameters used to generate each version, their metrics, and schemas.

![Different versions](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/MLflow_compare_models.PNG)

   
### **8.2 Moving model version to production**

The models generated by the pipelines are automatically assigned the stage `Staging`. This allows you to compare the new model version and decide whether to substitute the version currently in `Production`.

**Note:** If the **MAE (Mean Average Error)** value of the new model version is greater than the older value it is necessary to take into account:
- If the difference in the amount of data is <u>**insignificant**</u>, it is <u>**not recommended**</u> to transit model to production;
- If the difference in the amount of data is <u>**significant**</u>, even if the MAE slightly increases, it is <u>**recommended**</u> to transit this specific model to production;
- In the event of an increase in the amount of data <u>**equal or greater than 25%**</u>, it is <u>**recommended**</u> to transit the specific model to production.

To change the stage of a model version, go to the `Models` tab

![MLFlow UI](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/MLflow_UI.PNG)

Select the version you want to transit. The latest version in `Staging` will be displayed in the list of named models:

![Version](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/MLflow_select_version.PNG)

However, if you want to transit another version, select the named model and the respective version from the list of versions. Click on the Stage option and select `Transit to -> Production`.

![Production](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/MLflow_select_transition.PNG)

A pop-up window will appear asking for confirmation. Check the option to archite the existing `Production` model version and click `OK` to archive the current production model version and transit the selected model to `Production`.

![Accept](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/MLflow_accept_transition.PNG)

## **9. Visualizing the monitoring dashboards**

The Grafana application, supported by several other applications such as Prometheus and Loki, scrape metrics and logs reported by the pipeline and display them in user-friendly dashboards.

To visualize the dashboards, access the Grafana UI and log in (credentials indicated in topic 4.):

![Grafana log in](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/Grafana_login.PNG)

On the side menu, select `Dashboards` and then select `Manage`

![Dashboards and Manage](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/Grafana_manage_services.png)

Select the `Services` folder. Four dashboards should be displayed: `Cadvisor exporter`, `Logging`, `Pipeline metrics`, and `Web App`. The first display system-level metrics, e.g. cpu and memory usage by each container. The second display pipeline logs. The thrid displays pipeline metrics, e.g. node execution times and memory usage by dataset. The fourth presents metrics related to requests made to the predict / optimization module.

![Services](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/Grafana_services.png)

Selecting the `Pipeline metrics` should display the dashboards as bellow:

![Grafana Time](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/Grafana_time.PNG)

![Grafana Memory](https://github.com/EPMQ/INFOS/blob/dev-amatta/images/Grafana_memory.PNG)
