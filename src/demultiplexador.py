import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, spectrogram

class Demultiplexador:
    def __init__(self):
        self.__fs = 44100
        self.__duration = 3
        self.__cutoff_baseband = 3000  # Frequência de corte para filtro LP. Largura de banda do sinal antes da modulação
                        # Se for muito grande, pode pegar interferência de outros canais. Se for muito estreito, pode cortar parte do sinal.
        self.__order = 6 #controle o quão "abrupto" é o filtro. Order = 1 filtro simples, mais leve. Order grande, mais "afiado" ou "forte" o filtro

        # Frequências das portadoras
        self.__carriers = {'A': 4000, 'B': 10000,'C': 14000}
        self.__muxed, self.__fs = self.ler_sinal_multiplexado()
        self.demultiplexar()

    # === Filtros ===
    def bandpass_filter(self, signal, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        return lfilter(b, a, signal)

    def lowpass_filter(self, signal, cutoff, fs, order):
        nyq = 0.5 * fs
        b, a = butter(order, cutoff/nyq, btype='low')
        return lfilter(b, a, signal)

    def demodulate(self, signal, carrier_freq):
        t = np.arange(len(signal)) / self.__fs
        return signal * np.cos(2 * np.pi * carrier_freq * t)

    def normalize(self, signal):
        return signal / np.max(np.abs(signal))

    def plota_spectrograma(self, signal, fs, title, ax):
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=1024, noverlap=512)
        ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        ax.set_title(title)
        ax.set_ylabel('Frequência (Hz)')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylim(0, 3000)  # Limitar para banda de áudio
        ax.grid(True)

        # === Carregar o sinal multiplexado ===
    def ler_sinal_multiplexado(self):
        muxed, fs_read = sf.read("../assets/audio/muxed_audio.wav")
        assert fs_read == self.__fs, "Frequência de amostragem incorreta"
        if muxed.ndim > 1:
            muxed = muxed[:, 0]
        return muxed, fs_read
    
    def demultiplexar(self):
        for label, fc in self.__carriers.items():
            # 1. Filtrar a banda do canal
            band = self.bandpass_filter(self.__muxed, fc - self.__cutoff_baseband, fc + self.__cutoff_baseband, self.__fs, self.__order)
    
            # 2. Demodular para baseband
            demod = self.demodulate(band, fc)
    
            # 3. Filtrar passa-baixa para recuperar o áudio
            baseband = self.lowpass_filter(demod, self.__cutoff_baseband, self.__fs, self.__order)
    
            # 4. Normalizar e salvar
            baseband_norm = self.normalize(baseband)
            sf.write(f"../assets/audio/demux_channel_{label}.wav", baseband_norm, self.__fs)
            
            print(f"Canal {label} demultiplexado e salvo.")

        print("Todos os canais extraídos com sucesso.")
    
    def plota_espectro_multiplexador(self):
        N = len(self.__muxed)
        frequencias = fftfreq(N, 1/self.__fs)[:N//2]
        espectro = np.abs(fft(self.__muxed))[:N//2]

        plt.figure(figsize=(10, 4))
        plt.plot(frequencias, espectro)
        plt.title("Espectro do sinal multiplexado")
        plt.xlabel("Frequência (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.xlim(0, 16000)
        plt.savefig("../docs/figura4_espectro.png")
        plt.close()

    def ler_pares(self, base_path, demux_path):
        base, fs1 = sf.read(base_path)
        demux, fs2 = sf.read(demux_path)
        assert fs1 == fs2, "Taxas de amostragem não coincidem"
        min_len = min(len(base), len(demux))
        return base[:min_len], demux[:min_len], fs1

    # Dicionário de pares: base e extraído
    def compara_espectro(self):
        canais = {'A': ("../assets/audio/audio_A_base.wav", "../assets/audio/demux_channel_A.wav"),
                'B': ("../assets/audio/audio_B_base.wav", "../assets/audio/demux_channel_B.wav"),
                'C': ("../assets/audio/audio_C_base.wav", "../assets/audio/demux_channel_C.wav")
                }
        
        # Gerar espectrogramas para cada canal
        for label, (base_path, demux_path) in canais.items():
            base, demux, fs = self.ler_pares(base_path, demux_path)

            fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
            self.plota_spectrograma(base, fs, f"Original - Canal {label}", axs[0])
            self.plota_spectrograma(demux, fs, f"Demux - Canal {label}", axs[1])
            plt.tight_layout()
            plt.savefig(f"../docs/espectrograma_canal_{label}.png")
            plt.close()

    def plota_erros(self):
        #Plotar e comparar cada par
        canais = {'A': ("../assets/audio/audio_A_base.wav", "../assets/audio/demux_channel_A.wav"),
                'B': ("../assets/audio/audio_B_base.wav", "../assets/audio/demux_channel_B.wav"),
                'C': ("../assets/audio/audio_C_base.wav", "../assets/audio/demux_channel_C.wav")
                }
        for label, (base_path, demux_path) in canais.items():
            base, demux, fs = self.ler_pares(base_path, demux_path)
            
            # Calcular erro médio absoluto (MAE)
            mae = np.mean(np.abs(base - demux))
            
            # Plotar primeiros 1000 samples (~23 ms)
            plt.figure(figsize=(10, 3))
            plt.plot(base[:1000], label=f"Sinal original {label}", linewidth=1.5)
            plt.plot(demux[:1000], label=f"Demux {label}", linestyle='dashed', linewidth=1.2)
            plt.title(f"Comparação do Canal {label} - Erro Médio Absoluto: {mae:.6f}")
            plt.xlabel("Amostras")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

