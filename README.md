## Um estudo de caso em um sistema conversacional baseado em *Deep Learning*
O estudo de caso foi realizado sobre um modelo de *Deep Learning* aplicado a diálogos multi turnos baseado em recuperação.

### Visão Geral
Os códigos utilizados foram produzidos neste [*hands-on*](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow), os quais implementam o modelo Dual Encoder LSTM do *paper* [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](http://arxiv.org/abs/1506.08909), além de um algoritmo heurístico clássico da área de recuperação de informação chamado de modelo de espaço vetorial, que nesta pesquisa foi utilizado como *baseline*.

### Configuração

Os códigos utilizam Python 3. Clone o repositório e instale todos os pacotes necessários:
> **Nota:** Os scripts de geração da base de dados utilizam Python2.
```
1. install tensorflow (version 0.11 and above wokr correctly, version 0.10 not tested)
2. (opcional) install cuda + cudnn (recomendo para suporte gpu)
2. pip3 install -U pip
3. pip3 install -r requirements.txt
```

### Base de diálogos


Os experimentos foram realizados utilizando o Ubuntu Dialogue Corpus versão 2.0 apresentado no *paper* [Training End-to-End Dialogue Systems with the Ubuntu Dialogue Corpus](http://www.cs.toronto.edu/~lcharlin/papers/ubuntu_dialogue_dd17.pdf), cujo script de geração está disponível neste [repositório](https://github.com/rkadlec/ubuntu-ranking-dataset-creator). 
Entretanto, como o objetivo da pesquisa era entender o impacto do tamanho do *context* na previsão da próxima *utterance*, tornou-se necessário a modificação do script de geração para obter conjuntos de treinamento com a quantidade de turnos que fosse informada por argumento. Desta forma, o script modificado encontra-se na pasta scripts com nome [create_ubuntu_dataset_modificado.py](https://github.com/amycardoso/retrieval-based-chatbot/blob/master/scripts/create_ubuntu_dataset_modificado.py).

#### Geração do conjunto de treino com *script* modificado
Para geração dos conjuntos de treinamento siga os passos descritos neste [repositório](https://github.com/rkadlec/ubuntu-ranking-dataset-creator), exceto pela adição de mais um sub argumento, ao parser de treino, para determinar a quantidade de turnos desejada. 
#### Subparser:

`train`: gerador do conjunto de treino

-   `-t`: quantidade de turnos desejado

Exemplo para geração de um conjunto composto de *contexts* com 2 turnos:
```
python2 create_ubuntu_dataset_modificado.py --data_root ./dados -o 'train.csv' -t -s -l train -t 2
```
Execute a geração dos conjuntos de treino com o *script* modificado e para os conjuntos de validação e teste utilize o *script* original, ou faça download de todos os conjuntos necessários [aqui](https://drive.google.com/open?id=1--1LbkFMUIx6J3hqHFMrVtdPTkp5K9FY). Por fim, mova todos para a pasta `./data`.

#### Pré-Processamento
Antes de partir para o treinamento do modelo de *Deep Learning*, os conjuntos precisam ser transformados de CSV para TFRecord.
```
cd scripts
python3 prepare-data.py
```
### Modelo Dual Encoder LSTM
#### Treinamento

```
python3 udc_train.py
```

#### Avaliação

```
python3 udc_test.py --model_dir=...
```


**Exemplo:**
```
python3 udc_test.py --model_dir=./runs/1481183770/
```

#### Predição

```
python3 udc_predict.py --model_dir=...
```

**Exemplo:**
```
python3 udc_predict.py --model_dir=./runs/1481183770/
```

### Modelo de Espaço Vetorial

Como *baseline* utilizamos o modelo de espaço vetorial, disponível na pasta [notebooks](https://github.com/amycardoso/retrieval-based-chatbot/tree/master/notebooks).
