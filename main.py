from multiplexador import Multiplexador
from demultiplexador import Demultiplexador

if __name__ == "__main__":
    mult = Multiplexador()
    mult.multiplexacao() # Chama a multiplexação

    demux = Demultiplexador() # Chama a demultiplexação
    demux.plota_espectro_multiplexador() # Plota o espectro do sinal multiplexado
    demux.compara_espectro()
    demux.plota_erros()