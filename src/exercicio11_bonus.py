import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter
import librosa
import os

class SistemaComunicao:
    def __init__(self):
        # Taxa de amostragem padronizada para todos os áudios
        self.fs = 44100 
        # Frequências das portadoras escolhidas
        self.fc_A = 4000   # Canal A (ex: piano)
        self.fc_B = 10000  # Canal B (ex: bateria)
        self.fc_C = 14000  # Canal C (ex: violao)
        
        # Filtro de corte da banda base. 
        # Áudios reais têm espectro largo. Limitamos a 3kHz para não sobrepor na multiplexação.
        self.cutoff_baseband = 3000 
        self.order = 6 # Ordem dos filtros Butterworth

    # === Funções Auxiliares de Filtros ===
    def lowpass_filter(self, signal, cutoff, fs, order):
        nyq = 0.5 * fs
        b, a = butter(order, cutoff/nyq, btype='low')
        return lfilter(b, a, signal)

    def bandpass_filter(self, signal, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        return lfilter(b, a, signal)

    def normalize(self, signal):
        return signal / np.max(np.abs(signal))

    # === Etapa 1: Preparação do Sinal ===
    def carrega_audios(self):
        # Utilizando librosa para forçar o áudio para Mono (mono=True) e na mesma taxa (sr=fs)
        print("Carregando arquivos de áudio (isso pode demorar alguns segundos)...")
        a, _ = librosa.load("../assets/audio/piano.wav", sr=self.fs, mono=True)
        b, _ = librosa.load("../assets/audio/bateria.wav", sr=self.fs, mono=True)
        c, _ = librosa.load("../assets/audio/violao.wav", sr=self.fs, mono=True)

        # Truncar todos os áudios para o menor tamanho encontrado para podermos somar os arrays
        min_len = min(len(a), len(b), len(c))
        a = a[:min_len]
        b = b[:min_len]
        c = c[:min_len]

        # Pré-filtragem: aplica um Passa-Baixa em 3000 Hz antes da modulação
        # Isso garante que não haverá interferência (overlapping) quando somarmos no meio do canal
        a = self.lowpass_filter(a, self.cutoff_baseband, self.fs, self.order)
        b = self.lowpass_filter(b, self.cutoff_baseband, self.fs, self.order)
        c = self.lowpass_filter(c, self.cutoff_baseband, self.fs, self.order)

        # Salva os arquivos de referência filtrados (banda base real)
        sf.write("../assets/audio/base_A.wav", self.normalize(a), self.fs)
        sf.write("../assets/audio/base_B.wav", self.normalize(b), self.fs)
        sf.write("../assets/audio/base_C.wav", self.normalize(c), self.fs)

        return a, b, c, min_len

    # === Etapa 2: Modulação e Multiplexação ===
    def multiplexacao(self):
        a, b, c, length = self.carrega_audios()
        t = np.arange(length) / self.fs

        # Item (a): Modulação individual de cada sinal com sua respectiva portadora
        a_mod = a * np.cos(2 * np.pi * self.fc_A * t)
        b_mod = b * np.cos(2 * np.pi * self.fc_B * t)
        c_mod = c * np.cos(2 * np.pi * self.fc_C * t)

        # Mostra o espectro dos sinais modulados separadamente (Item a do Bônus)
        self.plota_espectros_individuais(a_mod, b_mod, c_mod, length)

        # Soma os sinais modulados para criar o sinal multiplexado FDM
        muxed = a_mod + b_mod + c_mod
        muxed = self.normalize(muxed)

        # Salva o arquivo derivado da multiplexação (Item c do Bônus)
        sf.write("../assets/audio/muxed_audio.wav", muxed, self.fs)
        print("Sinal multiplexado salvo como '../assets/audio/muxed_audio.wav'.")

        # Mostra o espectro do sinal multiplexado (Item b do Bônus)
        self.plota_espectro_multiplexado(muxed, length)

        return muxed, length

    # === Etapa 3: Demultiplexação e Recuperação ===
    def demultiplexacao(self, muxed, length):
        t = np.arange(length) / self.fs
        carriers = {'A': self.fc_A, 'B': self.fc_B, 'C': self.fc_C}

        print("Iniciando Demultiplexação...")
        for label, fc in carriers.items():
            # 1. Filtro Passa-Banda para isolar a região do canal
            band = self.bandpass_filter(muxed, fc - self.cutoff_baseband, fc + self.cutoff_baseband, self.fs, self.order)
            
            # 2. Demodulação multiplicando pela mesma portadora
            demod = band * np.cos(2 * np.pi * fc * t)
            
            # 3. Filtro Passa-Baixa para recuperar o áudio original isolado
            baseband = self.lowpass_filter(demod, self.cutoff_baseband, self.fs, self.order)
            
            # 4. Normaliza e Salva (Item f: os audios para comparação estarão salvos)
            baseband_norm = self.normalize(baseband)
            filename = f"../assets/audio/demux_channel_{label}.wav"
            sf.write(filename, baseband_norm, self.fs)
            print(f"Canal {label} (Portadora {fc}Hz) demultiplexado salvo como {filename}.")

    # === Funções Gráficas ===
    def plota_espectros_individuais(self, a_mod, b_mod, c_mod, length):
        frequencias = fftfreq(length, 1/self.fs)[:length//2]
        
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        plt.plot(frequencias, np.abs(fft(a_mod))[:length//2], color='blue')
        plt.title('Espectro - Sinal A Modulado (4000 Hz)')
        plt.xlim(0, 18000)

        plt.subplot(3, 1, 2)
        plt.plot(frequencias, np.abs(fft(b_mod))[:length//2], color='orange')
        plt.title('Espectro - Sinal B Modulado (10000 Hz)')
        plt.xlim(0, 18000)

        plt.subplot(3, 1, 3)
        plt.plot(frequencias, np.abs(fft(c_mod))[:length//2], color='green')
        plt.title('Espectro - Sinal C Modulado (14000 Hz)')
        plt.xlim(0, 18000)

        plt.tight_layout()
        plt.savefig("../docs/espectros_individuais.png")
        plt.close()

    def plota_espectro_multiplexado(self, muxed, length):
        frequencias = fftfreq(length, 1/self.fs)[:length//2]
        espectro = np.abs(fft(muxed))[:length//2]

        plt.figure(figsize=(10, 4))
        plt.plot(frequencias, espectro, color='purple')
        plt.title("Espectro do Sinal Multiplexado Completo")
        plt.xlabel("Frequência (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.xlim(0, 18000)
        plt.savefig("../docs/espectro_multiplexado.png")
        plt.close()

if __name__ == "__main__":
    sistema = SistemaComunicao()
    
    # Executa todo o pipeline do trabalho bônus
    sinal_multiplexado, min_length = sistema.multiplexacao()
    sistema.demultiplexacao(sinal_multiplexado, min_length)
    
    print("Processo concluído! Os arquivos originais (ajustados) e demultiplexados estão na pasta.")
