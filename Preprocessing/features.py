# 추출된 feature들을 모두 Dataframe 형태로 반환
# 이후에 이들을 모두 한 Dataframe에 concat하고, 이 데이터를 학습시킴
# csv로 굳이 만들 필요는 없음 - 애초에 csv를 불러오는 목적이 Dataframe을 로드하기 위함이니까
# 합친 후, 바로 Model로 넘길 것

import librosa
import numpy as np
import pandas as pd
import pyworld as pw


# MFCC - 검증 완료 (n_mfcc는 추출할 feature vector의 개수, 보통 12, 13, 20 등을 사용)
def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512, fmin=0, fmax=22050)
    mfcc_df = pd.DataFrame(mfcc.T, columns=[f'MFCC_{i + 1}' for i in range(mfcc.shape[0])]) # DataFrame으로 변환
    return mfcc_df

# Pitch - 검증 완료
def extract_pitch(y, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    pitches, magnitudes = librosa.core.piptrack(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Get the maximum pitch for each frame
    pitch = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch_value = pitches[index, t]
        pitch.append(pitch_value)

    pitch = np.array(pitch)
    pitch = np.nan_to_num(pitch)
    # pitch = pad_to_length(pitch, target_data_rows)  # Ensure the row count matches
    pitch_df = pd.DataFrame(pitch, columns=['Pitch'])
    return pitch_df

def extract_f0_pyworld(y, sr, frame_period=512 / 44100 * 1000):
    y = y.astype(np.float64)

    _f0, t = pw.dio(y, sr, frame_period=frame_period)  # Extract the initial fundamental frequency using the DIO algorithm
    f0 = pw.stonemask(y, _f0, t, sr)  # Refine the F0 estimation using the StoneMask algorithm

    # Convert to a DataFrame and ensure the row count matches
    # f0 = pad_to_length(f0, target_data_rows)
    # times = pad_to_length(t, target_data_rows)
    f0_df = pd.DataFrame(f0, columns=['F0'])
    return f0_df

# Spectral Flux - 검증 완료
def extract_spectral_flux(y, sr):
    S = np.abs(librosa.stft(y))  # STFT 계산
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))  # 스펙트럼 간의 차이 계산
    flux = np.append(flux, flux[-1])  # row 수 맞추기 위해 마지막 값 복제
    flux_df = pd.DataFrame(flux, columns=['Spectral_Flux'])
    return flux_df

# Spectral Entropy - 검증 완료
def extract_spectral_entropy(y, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    sum_S = np.sum(S, axis=0)
    # Check for zeros in sum_S and set them to a small value to avoid division by zero
    sum_S[sum_S == 0] = 1e-10  # or any small value to prevent division by zero

    ps = S / sum_S
    spectral_entropy = -np.sum(ps * np.log(ps + 1e-10), axis=0)
    spectral_entropy_df = pd.DataFrame(spectral_entropy, columns=['Spectral_Entropy'])
    return spectral_entropy_df





"""
# Fundamental Frequency (f0) - 검증 완료, 그러나 실시간 처리를 위한 최적화 필요
def extract_f0(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_f0 = np.nan_to_num(f0)
    pitch_f0_df = pd.DataFrame(pitch_f0, columns=['F0'])
    return pitch_f0_df
"""


"""
# low level로 구현한 formant 추출 함수 - 검증 완료, 그러나 너무 느림 (Python 기반)
# TODO : 속도 개선
def extract_formants_for_frames(audio_file, frame_length=0.025, hop_length=0.01):
    sample_rate, signal = wavfile.read(audio_file)
    if len(signal.shape) == 2:
        signal = signal[:, 0]

    # Frame parameters
    n_frame = int(frame_length * sample_rate)
    n_hop = int(hop_length * sample_rate)
    n_frames = 1 + int((len(signal) - n_frame) / n_hop)

    # Buffer to hold formant data
    formants_buffer = []

    for i in range(n_frames):
        start_sample = i * n_hop
        frame = signal[start_sample:start_sample + n_frame]
        windowed = frame * hamming(n_frame)
        lpc_order = 2 + sample_rate // 1000
        A = librosa.lpc(windowed, order=lpc_order)
        rts = np.roots(A)
        rts = rts[np.imag(rts) >= 0]
        angz = np.arctan2(np.imag(rts), np.real(rts))
        formants = angz * (sample_rate / (2 * np.pi))
        formants_sorted = np.sort(formants)

        # Handling variable number of formants
        formants_to_use = formants_sorted[:5]  # Take the first 5 formants
        if len(formants_to_use) < 5:
            formants_to_use = np.pad(formants_to_use, (0, 5 - len(formants_to_use)), 'constant',
                                     constant_values=(np.nan,))

        formants_buffer.append(formants_to_use)

    return np.array(formants_buffer)
"""


"""
# Formants using Parselmouth
def extract_formants_praat(audio_file):
    try:
        # 오디오 파일을 Sound 객체로 변환
        Sound = parselmouth.Sound(audio_file)

        # Formant 추출
        formant = Sound.to_formant_burg(time_step=0.1)

        # Pitch 추출
        pitch = Sound.to_pitch()

        # Formant 추출에 사용된 시간을 데이터프레임으로 만들기
        df = pd.DataFrame({"times": formant.ts()})

        # F1부터 F5까지의 값을 각 시간에 대해 추출
        for idx in range(1, 6):  # F1 to F5
            df[f'F{idx}'] = df['times'].apply(lambda x: formant.get_value_at_time(formant_number=idx, time=x))

        # Pitch값 추출 (F0)
        df['F0(pitch)'] = df['times'].apply(lambda x: pitch.get_value_at_time(time=x))
        # 파일 이름 추가
        df['filename'] = os.path.basename(audio_file)

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # 오류가 발생하면 빈 데이터프레임 반환
"""