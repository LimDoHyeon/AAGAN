import os
import torch
import wavio
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Preprocessing.preprocessing import create_dataloader
from audio_processing import DTW_align, perform_STT
from utils import data_denorm


def train(args, train_loader, models, criterions, optimizers, epoch, trainValid=True, inference=False):
    pass


def train_G(args, input, target, voice, labels, models, criterions, optimizer_g, data_info, trainValid):
    (model_g, model_d, model_STT, decoder_STT) = models
    (criterion_recon, criterion_ctc, criterion_adv, _, CER) = criterions

    if trainValid:
        model_g.train()
        model_d.train()
        model_STT.train()
    else:
        model_g.eval()
        model_d.eval()
        model_STT.eval()

    # Adversarial ground truths 1:real, 0: fake
    valid = torch.ones((len(input), 1), dtype=torch.float32).cuda()

    ###############################
    # Train Generator
    ###############################

    if trainValid:
        for p in model_g.parameters():
            p.requires_grad_(True)  # unfreeze G
        for p in model_d.parameters():
            p.requires_grad_(False)  # freeze D
        for p in model_STT.parameters():
            p.requires_grad_(False)  # freeze model_STT

        # set zero grad
        optimizer_g.zero_grad()

        # Run Generator
        output = model_g(input)
    else:
        with torch.no_grad():
            # run generator
            output = model_g(input)

    # DTW
    # TODO: Mel이 아니라 waveform 기준으로 코드 수정 -> waveform에 DTW가 최선인지 고민
    waveform_out = output.clone()
    waveform_out = DTW_align(waveform_out, target)

    # Run Discriminator
    g_valid, _ = model_d(waveform_out)

    # generator loss
    loss_recon = criterion_recon(waveform_out, target)

    # GAN loss (Discriminator loss로 추정)
    loss_valid = criterion_adv(g_valid, valid)

    # accuracy    args.l_g = h_g.l_g
    acc_g_valid = (g_valid.round() == valid).float().mean()

    ###############################
    # Loss from Vocoder - STT
    ###############################
    # out_DTW
    target_denorm = data_denorm(target, data_info[0], data_info[1])
    output_denorm = data_denorm(waveform_out, data_info[0], data_info[1])

    gt_label = []  # Generator_label
    gt_label_idx = []  # Generator_label_index
    gt_length = []
    for j in range(len(target)):
        gt_label.append(args.word_label[labels[j].item()])
        gt_label_idx.append(args.word_index[labels[j].item()])
        gt_length.append(args.word_length[labels[j].item()])
    gt_label_idx = torch.tensor(np.array(gt_label_idx), dtype=torch.int64)
    gt_length = torch.tensor(gt_length, dtype=torch.int64)

    # target
    wav_target = target_denorm
    wav_target = torch.reshape(wav_target, (len(wav_target), wav_target.shape[-1]))

    # resampling
    wav_target = torchaudio.functional.resample(wav_target, args.sample_rate_mel, args.sample_rate_STT)
    if wav_target.shape[1] != voice.shape[1]:
        p = voice.shape[1] - wav_target.shape[1]
        p_s = p // 2
        p_e = p - p_s
        wav_target = F.pad(wav_target, (p_s, p_e))

    # recon
    wav_generated = output_denorm
    wav_generated = torch.reshape(wav_generated, (len(wav_generated), wav_generated.shape[-1]))

    # resampling
    wav_generated = torchaudio.functional.resample(wav_generated, args.sample_rate_mel, args.sample_rate_STT)
    if wav_generated.shape[1] != voice.shape[1]:
        p = voice.shape[1] - wav_generated.shape[1]
        p_s = p // 2
        p_e = p - p_s
        wav_generated = F.pad(wav_generated, (p_s, p_e))

    # STT Wav2Vec 2.0
    emission_gt, _ = model_STT(voice)
    emission_recon, _ = model_STT(wav_generated)

    # CTC loss
    input_lengths = torch.full(size=(emission_gt.size(dim=0),), fill_value=emission_gt.size(dim=1), dtype=torch.long)
    emission_recon_ = emission_recon.log_softmax(2)
    loss_ctc = criterion_ctc(emission_recon_.transpose(0, 1), gt_label_idx, input_lengths, gt_length)

    # Total generator loss
    loss_g = args.l_g[0] * loss_recon + args.l_g[1] * loss_valid + args.l_g[2] * loss_ctc

    # decoder STT
    # TODO: decoder STT가 무슨 역할인지 알아야 함
    transcript_gt = []
    transcript_recon = []

    for j in range(len(voice)):
        transcript = decoder_STT(emission_gt[j])  # decoder_STT : from models
        transcript_gt.append(transcript)

        transcript = decoder_STT(emission_recon[j])
        transcript_recon.append(transcript)

    cer_gt = CER(transcript_gt, gt_label)  # CER : from criterions
    cer_recon = CER(transcript_recon, gt_label)

    if trainValid:
        loss_g.backward()
        optimizer_g.step()

    e_loss_g = (loss_g.item(), loss_recon.item(), loss_valid.item(), loss_ctc.item())
    e_acc_g = (acc_g_valid.item(), cer_gt.item(), cer_recon.item())

    return waveform_out, e_loss_g, e_acc_g


def train_D(args, waveform_out, target, target_cl, labels, models, criterions, optimizer_d, trainValid):
    (_, model_d, _, _, _) = models
    (_, _, criterion_adv, criterion_cl, _) = criterions

    if trainValid:
        model_d.train()
    else:
        model_d.eval()

    # Adversarial ground truths 1:real, 0: fake
    valid = torch.ones((len(waveform_out), 1), dtype=torch.float32).cuda()
    fake = torch.zeros((len(waveform_out), 1), dtype=torch.float32).cuda()

    ###############################
    # Train Discriminator
    ###############################

    if trainValid:
        if args.pretrain and args.prefreeze:
            for total_ct, _ in enumerate(model_d.children()):
                ct = 0
            for ct, child in enumerate(model_d.children()):
                if ct > total_ct - 1:  # unfreeze classifier
                    for param in child.parameters():
                        param.requires_grad = True  # unfreeze D
        else:
            for p in model_d.parameters():
                p.requires_grad_(True)  # unfreeze D

        # set zero grad
        optimizer_d.zero_grad()

    # run model cl
    real_valid, real_cl = model_d(target)
    fake_valid, fake_cl = model_d(waveform_out.detach())

    loss_d_real_valid = criterion_adv(real_valid, valid)
    loss_d_fake_valid = criterion_adv(fake_valid, fake)
    loss_d_real_cl = criterion_cl(real_cl, target_cl)

    loss_d_valid = 0.5 * (loss_d_real_valid + loss_d_fake_valid)
    loss_d_cl = loss_d_real_cl

    loss_d = args.l_d[0] * loss_d_cl + args.l_d[1] * loss_d_valid

    # accuracy
    acc_d_real = (real_valid.round() == valid).float().mean()
    acc_d_fake = (fake_valid.round() == fake).float().mean()
    preds_real = torch.argmax(real_cl, dim=1)
    acc_cl_real = (preds_real == labels).float().mean()
    preds_fake = torch.argmax(fake_cl, dim=1)
    acc_cl_fake = (preds_fake == labels).float().mean()

    if trainValid:
        loss_d.backward()
        optimizer_d.step()

    e_loss_d = (loss_d.item(), loss_d_valid.item(), loss_d_cl.item())
    e_acc_d = (acc_d_real.item(), acc_d_fake.item(), acc_cl_real.item(), acc_cl_fake.item())

    return e_loss_d, e_acc_d


def saveData(args, test_loader, models, epoch, losses):
    model_g = models[0].eval()
    # model_d = models[1].eval()
    model_STT = models[3].eval()
    decoder_STT = models[4]

    input, target, target_cl, voice, data_info = next(iter(test_loader))

    input = input.cuda()
    target = target.cuda()
    voice = torch.squeeze(voice, dim=-1).cuda()
    labels = torch.argmax(target_cl, dim=1)

    with torch.no_grad():
        # run the mdoel
        output = model_g(input)

    mel_out = DTW_align(output, target)
    wav_generated = data_denorm(mel_out, data_info[0], data_info[1])
    wav_generated = torch.reshape(wav_generated, (len(wav_generated), wav_generated.shape[-1]))

    wav_generated = torchaudio.functional.resample(wav_generated, args.sample_rate_mel, args.sample_rate_STT)
    if wav_generated.shape[1] != voice.shape[1]:
        p = voice.shape[1] - wav_generated.shape[1]
        p_s = p // 2
        p_e = p - p_s
        wav_generated = F.pad(wav_generated, (p_s, p_e))

    ##### STT Wav2Vec 2.0
    gt_label = args.word_label[labels[0].item()]

    transcript_recon = perform_STT(wav_generated, model_STT, decoder_STT, gt_label, 1)

    # save
    wav_generated = np.squeeze(wav_generated.cpu().detach().numpy())

    str_tar = args.word_label[labels[0].item()].replace("|", ",")
    str_tar = str_tar.replace(" ", ",")

    str_pred = transcript_recon[0].replace("|", ",")
    str_pred = str_pred.replace(" ", ",")

    title = "Tar_{}-Pred_{}".format(str_tar, str_pred)
    wavio.write(args.savevoice + '/e{}_{}.wav'.format(str(str(epoch)), title), wav_generated, args.sample_rate_STT,
                sampwidth=1)


def main():
    # Parameters
    train_dir = 'G:/공유 드라이브/4N_PKNU/BXAI/BMI/Mel-ResNet/Voice/Train'
    train_wav = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.endswith('.wav')]
    test_dir = 'G:/공유 드라이브/4N_PKNU/BXAI/BMI/Mel-ResNet/Voice/Test'
    test_wav = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith('.wav')]
    val_dir = 'G:/공유 드라이브/4N_PKNU/BXAI/BMI/Mel-ResNet/Voice/Val'
    val_wav = [os.path.join(val_dir, file) for file in os.listdir(val_dir) if file.endswith('.wav')]

    max_length = 10000  # example
    batch_size = 32  # example

    train_dataloader = create_dataloader(train_wav, max_length, batch_size, shuffle=True)
    test_dataloader = create_dataloader(test_wav, max_length, batch_size, shuffle=True)
    val_dataloader = create_dataloader(val_wav, max_length, batch_size, shuffle=True)

