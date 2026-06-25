# Trabalho de Comunicação de Dados

Repositório contendo os scripts em Python, áudios de teste e o relatório final em LaTeX referente ao trabalho da disciplina de Comunicação de Dados (UTFPR - Prof. Dr. Luiz Fernando Carvalho).

## Estrutura do Projeto

- **`src/`**: Contém os códigos fonte em Python (`main.py`, `multiplexador.py`, `demultiplexador.py` e o script bônus unificado `exercicio11_bonus.py`).
- **`assets/audio/`**: Contém os arquivos de áudio originais (piano, violão, bateria) e os áudios resultantes da multiplexação e demultiplexação.
- **`docs/`**: Contém o código fonte do relatório em LaTeX, as imagens geradas pelos scripts e o PDF final compilado (`Relatorio_Final.pdf`).

## Como Executar

Para rodar os scripts, é necessário ter o Python instalado com as bibliotecas `numpy`, `soundfile`, `matplotlib`, `scipy` e `librosa`.

Você pode executar o script principal acessando a pasta `src`:

```bash
cd src
python exercicio11_bonus.py
```

Isso irá carregar os áudios da pasta `assets/audio/`, aplicar a multiplexação e demultiplexação AM DSB-SC, salvar os arquivos de áudio resultantes e plotar os espectros em imagens salvas na pasta `docs/`.