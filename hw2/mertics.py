import numpy as np


def calculate_snr(clean_signal, processed_signal):
    noise = clean_signal - processed_signal
    signal_power = np.mean(clean_signal**2)
    noise_power = np.mean(noise**2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_si_sdr(clean_signal, processed_signal):
    s = clean_signal
    s_hat = processed_signal
    alpha = np.dot(s_hat, s) / np.dot(s, s)
    s_target = alpha * s
    e = s_hat - s_target
    target_power = np.dot(s_target, s_target)
    error_power = np.dot(e, e)

    if error_power == 0:
        return float('inf')

    si_sdr = 10 * np.log10(target_power / error_power)
    return si_sdr

def calculate_metrics(clean_signal, noisy_signal, denoised_signal):
    metrics = {}
    metrics['snr_noisy'] = calculate_snr(clean_signal, noisy_signal)
    metrics['snr_denoised'] = calculate_snr(clean_signal, denoised_signal)
    metrics['si_sdr_noisy'] = calculate_si_sdr(clean_signal, noisy_signal)
    metrics['si_sdr_denoised'] = calculate_si_sdr(clean_signal, denoised_signal)
    metrics['snr_improvement'] = metrics['snr_denoised'] - metrics['snr_noisy']
    metrics['si_sdr_improvement'] = metrics['si_sdr_denoised'] - metrics['si_sdr_noisy']
    
    return metrics

def print_metrics(metrics):
    print("=" * 50)
    print("МЕТРИКИ КАЧЕСТВА ШУМОПОДАВЛЕНИЯ")
    print("=" * 50)
    print(f"SNR зашумленного сигнала:    {metrics['snr_noisy']:7.2f} dB")
    print(f"SNR очищенного сигнала:      {metrics['snr_denoised']:7.2f} dB")
    print(f"Улучшение SNR:               {metrics['snr_improvement']:7.2f} dB")
    print("-" * 50)
    print(f"SI-SDR зашумленного сигнала: {metrics['si_sdr_noisy']:7.2f} dB")
    print(f"SI-SDR очищенного сигнала:   {metrics['si_sdr_denoised']:7.2f} dB")
    print(f"Улучшение SI-SDR:            {metrics['si_sdr_improvement']:7.2f} dB")
    print("=" * 50)