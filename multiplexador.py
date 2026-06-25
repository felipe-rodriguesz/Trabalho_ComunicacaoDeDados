import numpy as np
import soundfile as sf

class Multiplexador:
    def __init__(self): # Construtor
        self.__fs = 44100    # Frequência de amostragem (Hz)
        self.__duration = 3  # Duração do sinal (segundos)
        self.__fc_A = 4000   # Portadora do sinal A        
        self.__fc_B = 10000  # Portadora do sinal B
        self.__fc_C = 14000  # Portadora do sinal C
        self.__t = self.amostragem()
        self.__muxed = self.multiplexacao()

    def amostragem(self):
       t = np.linspace(0, self.__duration, int(self.__fs * self.__duration), endpoint=False)
       return t
    
    def geraSinais(self): #Sinais de áudio base
        a = np.sin(2 * np.pi * 440 * self.__t)    # Sinal A: 440 Hz
        b = np.sin(2 * np.pi * 880 * self.__t)    # Sinal B: 880 Hz
        c = np.sin(2 * np.pi * 1760 * self.__t)   # Sinal C: 1760 Hz
        return a, b, c
    
    def modulacao(self, a, b, c): #  Modulação em amplitude (DSB)
        a_mod = a * np.cos(2 * np.pi * self.__fc_A * self.__t)
        b_mod = b * np.cos(2 * np.pi * self.__fc_B * self.__t)
        c_mod = c * np.cos(2 * np.pi * self.__fc_C * self.__t)
        return a_mod, b_mod, c_mod

    def salvaSinais(self, a, b, c, muxed):
        sf.write("muxed_audio.wav", muxed, self.__fs)
        sf.write("audio_A_base.wav", a, self.__fs)
        sf.write("audio_B_base.wav", b, self.__fs)
        sf.write("audio_C_base.wav", c, self.__fs)
        print("Multiplexação concluída e saiva no arquivo muxed_audio.wav.")

    def multiplexacao(self):
        a, b, c = self.geraSinais()
        a_mod, b_mod, c_mod = self.modulacao(a, b, c)
        muxed = a_mod + b_mod + c_mod # Multiplexação
        muxed /= np.max(np.abs(muxed)) #Normalizar para evitar clipping
        self.salvaSinais(a, b, c, muxed)
        return muxed