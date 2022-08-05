
<br>
<p align="center">
  <h2 align="center"> Atlantic - Automated PreProcessing Framework for Supervised Machine Learning
  <br>
  

## Technological Contextualization <a name = "ta"></a>

This project constitutes an comprehensive and objective approach to automate data processing through the integration and objectively validated application of various processing mechanisms, ranging from feature engineering, automated feature selection, different encoding versions and null imputation methods.  The optimization methodology of this framework follows a evaluation structured in tree-based models by the implemention of Random Forest and Extra Trees ensembles.

#### Main Development Tools <a name = "pre1"></a>
    
* [Python](https://www.python.org/downloads)
* [H2O.ai](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
* [Sklearn](https://scikit-learn.org/stable/)

    
## Framework Architecture <a name = "ta"></a>

![Atlantic Architecture](https://github.com/TsLu1s/Atlantic/blob/main/img/ATL%20Architecture.PNG)
    
    ---
<img src="https://github.com/TsLu1s/Atlantic/blob/main/img/ATL%20Architecture.PNG" width="500" height="500" />


    
    
### Preprocessing Available Methods  <a name = "ta1"></a>

    
    

As versões instaladas devem ser as mais recentes referentes aos links abaixo indicados:

* [PBIDesktopSetup_x64.exe - https://www.microsoft.com/pt-PT/download/details.aspx?id=58494]([https://www.swi-prolog.org/download/stable](https://www.microsoft.com/pt-PT/download/details.aspx?id=58494))
* [Anaconda Distributions - https://www.anaconda.com/products/distribution]([https://www.python.org/downloads](https://www.anaconda.com/products/distribution))

### Greenshoes Github Project  <a name = "ta2"></a>

Instalação da cópia completa dos dados existentes no repositório GitHub, incuindo todo o software produzido associado á elaboração do protótipo .


  ```sh
  https://github.com/EPMQ/GreenShoes.git
  ```

## Configuração Anaconda  <a name = "tb"></a>

A Anaconda's open-source Distribution será utilizada para integrar a codigo python desenvolvido. Serão instalados através da mesma a linguagem python e a bibliotecas utilizadas na execução do projeto. Após instalado a ferramenta "Anaconda Navigator" prossegue-se á instalação das dependencias associadas a elaboração do codigo pyhton.

### Instalação e Customização Anaconda <a name = "tb1"></a>

### Criação Ambiente de Desenvolvimento <a name = "tb2"></a>

Nesta secção é criado o ambiente onde será instalada a versão python (3.8.12) e todas as dependencias associadas ao projeto.

  ```sh
  Anaconda.Navigator > Environments > Create > Create New Environment -> Name = Greenshoes
  ```
  
![Criar_Ambiente](https://github.com/EPMQ/GreenShoes/blob/dev/images/Criar_Ambiente.PNG)

### Customização do Ambiente <a name = "tb3"></a>

Criado o ambiente segue-se a instalação das bibliotecas implementadas no desenvolvimento do projeto.

  ```sh
Anaconda prompt > conda activate Greenshoes  
pip install numpy==1.22.3
pip install pandas==1.1.0
pip install sklearn==1.0.2
pip install pmdarima==1.8.4
pip install matplotlib
  ```

#### Configuração Powerbi  <a name = "td"></a>

Nesta secção é demostrada a conexão do ambiente Anaconda criado e a sua compatilibilização e integração com o interface tecnológico PowerBi desenvolvido. 

### Acesso ao Protótipo Powerbi Desenvolvido <a name = "td1"></a>

Aceder ao repositório Github Greenshoes e abrir a aplicação 'Greenshoes - Gestão de Encomendas.pbix'

  ```sh
  GreenShoes/PowerBi/Greenshoes - Gestão de Encomendas.pbix
  ```

#### Conectar PowerBi ao Ambiente Python<a name = "td2"></a>

Ligação direta do protótipo desenvolvido ao ambiente criado conectando desta forma as depêndencias necessarias ao seu funcionamento.

  ```sh
  Ficheiro > Opções e Definições > Opções > Scripts de Python
  ```

![Integrar_Python](https://github.com/EPMQ/GreenShoes/blob/dev/images/Integracao.PNG)

### Permitir Interligação de Fonte Externa <a name = "td3"></a>

Aumento de Performance relativa á dimensão de performance associada de conjunto de dados integrados e permissão relativa ao acesso do codigo Python produzido no desenvolvimento do projeto.

  ```sh
  Ficheiro > Opções e Definições > Opções > Privacidade > 2ª Opção
  ```

#### Atualizar Dados <a name = "td4"></a>

Uma vez concluida a conexão do ambiente de desenvolvimento anaconda (Greenshoes) especificado no ponto anterior, atualizam-se os dados e o deploy dá-se assim por concluido. 

  ```sh
  Base| Consultas -> Atualizar
  ```

