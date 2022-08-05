<br>
<p align="center">
  <a href="https://www.uminho.pt" target="_blank"><img src="https://i.imgur.com/FXQo8OL.png" alt="Universidade do Minho"></a>
  <a href="https://www.eng.uminho.pt" target="_blank"><img src="https://i.imgur.com/WABo4st.png" alt="Escola de Engenharia"></a>
  <br><a href="http://www.dsi.uminho.pt" target="_blank"><strong>Departamento de Sistemas de Informação</strong></a>
  
  <h2 align="center">Projeto Prático SBC - MIEGSI 2020/2021</h2>
  <br>
  
## Índice de conteúdos

- [Introdução](#intro)
- [Tarefa A - Aconselhamento para compra de uma refeição](#ta)
  - [Parte 1 - Aquisição manual de conhecimento](#ta1)
  - [Parte 2 - Aquisição automática de conhecimento](#ta2)
  - [Pré-requisitos](#pre1)
  - [Getting started](#getting1)
    - [Quick-start](#quick1)
    - [Installation](#install1)
    - [Usage](#usage1)
- [Tarefa B - Aconselhamento de trajeto para entrega de uma refeição](#tb)
  - [Parte 1 - Resolução via Procura ](#tb1)
  - [Parte 2 - Otimização do lucro, tempo ou ambos ](#tb2)
  - [Pré-requisitos](#pre2)
  - [Getting started](#getting2)
    - [Quick-start](#quick2)
    - [Usage](#usage2)
- [Ferramentas](#built)
- [Licença](#license)
- [Contactos](#contact)
- [Reconhecimentos](#ack)
- [Referências](#refer)

## Introdução <a name = "intro"></a>

No âmbito da unidade curricular de Sistemas Baseados em Conhecimento, foi-nos proposto a conceção de um SBC implementado na linguagem Prolog, estando a mesma dividida em 2 tarefas com 2 partes cada uma, tendo por base o conceito de food delivery, tão em voga no último ano decorrente da situação pandémica que vivenciamos.


## Tarefa A - Aconselhamento para compra de uma refeição <a name = "ta"></a>
Dentro do conceito de fooddelivery, take away & drive-in, pretende-se elaborar um SBC para aconselhar sobre a escolha e compra de uma refeição(com entrega em casa ou take away). Através de uma interface desenvolvida em python com auxílio do interpretador de Prolog [Pyswip](https://pypi.org/project/pyswip), é possível fazer pesquisas na nossa base de conhecimento Prolog usando o WhatsApp.

### Parte 1 - Aquisição manual de conhecimento  <a name = "ta1"></a>
Nesta fase foram usadas técnicas de aquisição de conhecimento manual (Pesquisa e entrevistas) para a conceção de [regras de produção manuais](/Tarefa_A//server/prolog/baseconhecimento.pl).

### Parte 2 - Aquisição automática de conhecimento <a name = "ta2"></a>
Nesta segunda fase foi desenvolvido e partilhado um formulário usando o [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSeNAQaEAmYyQwujrFkVR1biIBEo4PDDMI2MmAVUZw43gGIujA/viewform). A respostas foram depois descarregas para o ficheiro [pratos.csv](/Tarefa_A/data_mining/pratos.csv). Foi depois escrita uma [pequena script em R](/Tarefa_A/data_mining/pratos_script.R) que gerou [regras de produção](/Tarefa_A/data_mining/regras.txt) de forma automática, que pudessem ser usadas pelo [Prolog](/Tarefa_A/server/prolog/baseconhecimentoautomatica.pl).

### Pré-requisitos <a name = "pre1"></a>
* [SWI-Prolog 8.2.4](https://www.swi-prolog.org/download/stable)
* [Python 3.9](https://www.python.org/downloads)
* [Twilio Sandbox for WhatsApp](https://www.twilio.com/docs/whatsapp/sandbox)
* [npm](https://www.npmjs.com/get-npm)
* [R 4.0](https://cran.r-project.org/mirrors.html)
* [RStudio](https://www.rstudio.com/products/rstudio/download)

## Getting started <a name = "getting1"></a>

### Quick-start <a name = "quick1"></a>

#### Start your server
  ```sh
  python app.py
  ```
#### Expose your server to the world
  ```sh
  npx localtunnel --port 5000
  ```
#### Configure your Twilio Sandbox
![Twilio Sandbox](https://i.imgur.com/0tFXC6P.png)

#### Join the Twilio WhatsApp Sandbox
Send a WhatsApp message to **Your sandbox WhatsApp number** with the correct **Sandbox join code**

![Join the Twilio WhatsApp Sandbox](https://i.imgur.com/8tKLhgM.png)

### Installation  <a name = "install1"></a>
#### Add SWI-Prolog to the PATH environment variable
#### Clone the repo
  ```sh
  git clone https://github.com/nonvegan/trabalho-sbc.git
  ```
#### Install all the dependecies
  ```sh
  pip install flask pyswip twilio python-dotenv
  ```
#### Configure your .env file
  ```python
  TWILIO_USER=twiliouser
  TWILIO_TOKEN=twiliotoken
  TWILIO_PHONE=whatsapp:+123456789
  PHONE=whatsapp:+351123456789
  ```
### Usage  <a name = "usage1"></a>
* Type a message with the keyword **!dish**.
* Answer the quick survey.
* Wait for your dish suggestion.
* Message either **!manual** or **!automatica** to switch the type of the knowledge base.
* Type **!dish** to start over.

![Bot](https://i.imgur.com/uquLinP.png)

## Tarefa B - Aconselhamento de trajeto para entrega de uma refeição <a name = "tb"></a>
Desenvolver um SBC para um estafeta que usa uma scooter como meio de transporte que trabalha para um sistema de entrega de um restaurante. O SBC deve aconselhar que encomendas o estafeta deve pegar no restaurante e qual o caminho a seguir para proceder às entregas. Optamos por desenvolver uma webapp com o auxílio do interpretador Prolog em JavaScript [Tau prolog](http://tau-prolog.org) como interface para o nosso SBC.

### Parte 1 - Resolução via Procura <a name = "tb1"></a>
Nesta parte foram desenvolvidas as funcionalidades de procura para o objetivo 1 (a scooter só pode levar uma encomenda de cada vez) e 2 (a scooter pode levar uma ou duas encomendas de cada vez), podendo o utilizador escolher depth-first, iterative-deepening e breath-first como métodos de procura.
Publicamos o nosso SBC na web através da plataforma Netlify no endereço https://projeto-sbc-g53-parte2-miegsi-2021.netlify.app

### Parte 2 - Otimização do lucro, tempo ou ambos <a name = "tb2"></a>
Nesta parte foram desenvolvidas as funcionalidades de optimização usando o método de hillclimbing para o objetivo A (maximizar o lucro), B (minimizar o tempo do percurso) e C (maximizar 0.8*lucro+0.2*(20-tempo)).


### Pré-requisitos <a name = "pre2"></a>
* [serve](https://www.npmjs.com/package/serve)

## Getting started <a name = "getting2"></a>

### Quick-start <a name = "quick2"></a>

#### Clone the repo
  ```sh
  git clone https://github.com/nonvegan/trabalho-sbc.git
  ```

#### Serve the static files
  ```sh
  serve webapp -l 80
  ```
### Usage <a name = "usage2"></a>

* Select the running mode( 1 or 2 deliveries) 
* Select the searching mode
* Click search

![Procura](https://i.imgur.com/A5CepUX.png)
  
## Ferramentas <a name = "built"></a>
* [SWI-Prolog](https://www.swi-prolog.org)
* [Python](https://www.python.org)
* [Pyswip](https://pypi.org/project/pyswip)
* [Twilio](https://www.twilio.com)
* [R](https://www.r-project.org)
* [JavaScript](https://www.javascript.com)
* [Tau Prolog](http://tau-prolog.org)

## Licença <a name = "license"></a>

Distributed under the MIT License. See `LICENSE` for more information.

## Contactos <a name = "contact"></a>

* [Pedro Magalhães](mailto:pedromagalhaes_2000@hotmail.com)
* [Álvaro Ferreira](mailto:alvarobahrain@gmail.com)
* [André Gomes](mailto:andrede@live.com.pt)
* [José Carvalho](mailto:zemmcarvalho@gmail.com)

## Reconhecimentos <a name = "ack"></a>
* [Paulo Cortez](http://www3.dsi.uminho.pt/pcortez/Home.html)
* [André Pilastri](https://pilastri.github.io/andrepilastri.github.io/#about)

## Referências <a name = "refer"></a>
* Cortez, P. (2018). Exercícios Resolvidos em Prolog sobre Sistemas Baseados em Conhecimento: Regras de Produção, Extração de Conhecimento, Procura e Otimização. Teaching Report, University of Minho, Guimarães, Portugal.
* Cortez, P. (2015). A tutorial on the rminer R package for data mining tasks. Teaching Report, University of Minho, Guimarães, Portugal.
* Bratko, I. (2012). Programming in Prolog for Artificial intelligence. Pearson Education, 4thedition, Harlow, England.
* Wielemaker, J., De Koninck, L., Fruehwirth, T., Triska, M., & Uneson, M. (2014). SWI Prolog Reference Manual 7.1.

