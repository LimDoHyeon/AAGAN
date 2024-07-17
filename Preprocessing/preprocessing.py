import librosa
import numpy as np
import pandas as pd
import Preprocessing.features as features  # features.py 파일을 import
import torch
from torch.utils.data import Dataset, DataLoader


class WAVDataset(Dataset):
    def __init__(self, wav_files, max_length):
        """
        초기화 메서드
        :param wav_files: 처리할 WAV 파일의 경로 리스트
        :param max_length: 각 샘플의 최대 길이
        """
        self.wav_files = wav_files
        self.max_length = max_length

    def __len__(self):
        """
        데이터셋의 길이 반환
        :return: 데이터셋에 있는 WAV 파일의 수
        """
        return len(self.wav_files)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 데이터 반환
        :param idx: 인덱스
        ====================================================
        - Preprocessing (features.py 이용)
        - 피처 병합 (merge_features 메서드 이용)
        - 피처 데이터(X)와 라벨(y) 분리
        ====================================================
        :return: 피처 데이터(X)와 라벨(y)을 텐서 형태로 반환
        """
        wav_path = self.wav_files[idx]
        y, sr = librosa.load(wav_path, sr=44100)

        # Preprocessing 과정
        mfcc = features.extract_mfcc(y, sr)
        pitch = features.extract_pitch(y, sr)
        f0_pyworld = features.extract_f0_pyworld(y, sr)
        spectral_flux = features.extract_spectral_flux(y, sr)
        spectral_entropy = features.extract_spectral_entropy(y, sr)

        # 추출된 feature 병합한 dataframe을 concated_df로 선언 후, return
        # 피처 병합
        features_dict = {
            'mfcc': mfcc,
            'pitch': pitch,
            'f0_pyworld': f0_pyworld,
            'spectral_flux': spectral_flux,
            'spectral_entropy': spectral_entropy
        }
        concated_df = self.merge_features(features_dict)

        # 패딩 또는 자르기 적용
        X = self.pad_or_truncate(concated_df.values)

        # 라벨과 나머지 데이터 분리
        y = X[:, -1]  # 마지막 열이 라벨
        X = X[:, :-1]  # 나머지가 피처 데이터

        # X와 y를 텐서로 변환
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y

    def merge_features(self, features_dict):
        # 피처들을 하나의 데이터프레임으로 병합
        df_list = []
        for key, df in features_dict.items():
            # 각 DataFrame의 행 수를 통일
            df_list.append(df)

        # 열 방향으로 병합
        concated_df = pd.concat(df_list, axis=1)
        return concated_df

    def pad_or_truncate(self, features):
        length, feature_dim = features.shape
        if length > self.max_length:
            return features[:self.max_length]
        elif length < self.max_length:
            pad_width = self.max_length - length
            padding = np.zeros((pad_width, feature_dim))
            return np.vstack((features, padding))
        return features


def create_dataloader(wav_files, max_length, batch_size, shuffle=True):
    dataset = WAVDataset(wav_files, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
