import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from mel_spectrogram import mel_spectrogram, stft, istft
from mertics import print_metrics, calculate_metrics

def add_gaussian_noise(audio, snr_db=20):
    signal_power = np.mean(audio**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
    noisy_audio = audio + noise
    return noisy_audio

def spectral_subtraction(noisy_audio, N_fft=1024, H=256, L=800, window_func=None, 
                         subtraction_factor=2.0):
    if window_func is None:
        window_func = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_fft) / (N_fft - 1))

    stft_noisy = stft(noisy_audio, N_fft, H, L, window_func)
    n_frames = stft_noisy.shape[1]
    frame_energies = np.sum(np.abs(stft_noisy)**2, axis=0) # вычисляем энергию каждого кадра
    energy_threshold = np.percentile(frame_energies, 40) # берем порог как 40ой перцентиль энергии
    noise_frames_indices = np.where(frame_energies < energy_threshold)[0] # выбираем кадры с энергией ниже порога как шумовые
    if len(noise_frames_indices) > 0:
        noise_spectrum = np.mean(np.abs(stft_noisy[:, noise_frames_indices])**2, axis=1) # усредняем шумовые кадры

    stft_clean = np.zeros_like(stft_noisy)
    for i in range(n_frames):
        magnitude_squared = np.abs(stft_noisy[:, i])**2
        clean_magnitude_squared = np.maximum(magnitude_squared - subtraction_factor * noise_spectrum, 1e-8)
        clean_magnitude = np.sqrt(clean_magnitude_squared)
        phase = np.angle(stft_noisy[:, i])
        stft_clean[:, i] = clean_magnitude * np.exp(1j * phase)
    
    return stft_clean

def plot_mel_spectrograms(mel_orig, mel_noisy, mel_denoised, fs, H):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Original", "Noisy", "Denoised"]
    mel_list = [mel_orig, mel_noisy, mel_denoised]

    for ax, mel_spec, title in zip(axs, mel_list, titles):
        img = ax.imshow(
            mel_spec,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[0, mel_spec.shape[1] * H / fs, 0, mel_spec.shape[0]]
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mel bands")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig('mel_spec')
    plt.show()

def main():
    audio, fs = sf.read("speech.mp3")
    noisy_audio = add_gaussian_noise(audio, snr_db=15)
    sf.write("speech_with_noise.mp3", noisy_audio, fs)

    N_fft = 1024
    H = 256
    L = 800
    window_func = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_fft) / (N_fft - 1))
    stft_clean = spectral_subtraction(
            noisy_audio, 
            N_fft=N_fft, 
            H=H, 
            L=L, 
            window_func=window_func,
            subtraction_factor=2.0
        )
    clean_audio = istft(stft_clean, N_fft, H, window_func, len(noisy_audio))
    sf.write("denoised_speech.mp3", clean_audio, fs)


    mel_spec_orig, fs = mel_spectrogram("speech.mp3")
    mel_spec_noisy, _ = mel_spectrogram("speech_with_noise.mp3")
    mel_spec_denoised, _ = mel_spectrogram("denoised_speech.mp3")
    plot_mel_spectrograms(mel_spec_orig, mel_spec_noisy, mel_spec_denoised, fs, H)

    denoised_audio, _ = sf.read("denoised_speech.mp3")
    clean_audio = clean_audio / np.max(np.abs(clean_audio))
    noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
    denoised_audio = denoised_audio / np.max(np.abs(denoised_audio))

    metrics = calculate_metrics(clean_audio, noisy_audio, denoised_audio)
    print_metrics(metrics)


if __name__ == "__main__":
    main()