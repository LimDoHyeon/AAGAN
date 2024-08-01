import os
import torch
import wavio
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Preprocessing.preprocessing import create_dataloader
from utils import data_denorm
from utils_audio import DTW_align


def train(args, train_loader, models, criterions, optimizers, epoch, trainValid=True, inference=False):
    (optimizer_g, optimizer_d) = optimizers

    # Switch to train mode
    assert type(models) == tuple, "More than two models should be inputted (generator and discriminator)"

    epoch_loss_g = []; epoch_loss_d = []
    epoch_acc_g = []; epoch_acc_d = []

    total_batches = len(train_loader)

    for i, (input_data, target, target_cl, voice, data_info) in enumerate(train_loader):

        print("\rBatch [%5d / %5d]" % (i, total_batches), sep=' ', end='', flush=True)

        input_data = input_data.cuda()
        target = target.cuda()
        target_cl = target_cl.cuda()
        voice = torch.squeeze(voice, dim=-1).cuda()
        labels = torch.argmax(target_cl, dim=1)

        idx = []
        for j in range(len(labels)):
            idx.append(j)

        # Data initialization
        input_data = input_data[idx]
        target = target[idx]; target_cl = target_cl[idx]
        voice = voice[idx]
        labels = labels[idx]
        data_info = [data_info[0][idx], data_info[1][idx]]

        # Training process
        if len(input_data) != 0:
            # Train generator
            mel_out, e_loss_g, e_acc_g = train_g(args,
                                                 input_data, target, voice, labels,
                                                 models, criterions, optimizer_g,
                                                 data_info,
                                                 trainValid)
            epoch_loss_g.append(e_loss_g)
            epoch_acc_g.append(e_acc_g)

            # Train discriminator
            e_loss_d, e_acc_d = train_d(args,
                                        mel_out, target, target_cl, labels,
                                        models, criterions, optimizer_d,
                                        trainValid)
            epoch_loss_d.append(e_loss_d)
            epoch_acc_d.append(e_acc_d)

    epoch_loss_g = np.array(epoch_loss_g); epoch_acc_g = np.array(epoch_acc_g)
    epoch_loss_d = np.array(epoch_loss_d); epoch_acc_d = np.array(epoch_acc_d)

    # Average loss and accuracy
    args.loss_g = sum(epoch_loss_g[:, 0]) / len(epoch_loss_g[:, 0])
    args.loss_g_recon = sum(epoch_loss_g[:, 1]) / len(epoch_loss_g[:, 1])
    args.loss_g_valid = sum(epoch_loss_g[:, 2]) / len(epoch_loss_g[:, 2])

    args.acc_g_valid = sum(epoch_acc_g[:, 0]) / len(epoch_acc_g[:, 0])

    args.loss_d = sum(epoch_loss_d[:, 0]) / len(epoch_loss_d[:, 0])
    args.loss_d_valid = sum(epoch_loss_d[:, 1]) / len(epoch_loss_d[:, 1])
    args.loss_d_cl = sum(epoch_loss_d[:, 2]) / len(epoch_loss_d[:, 2])

    args.acc_d_real = sum(epoch_acc_d[:, 0]) / len(epoch_acc_d[:, 0])
    args.acc_d_fake = sum(epoch_acc_d[:, 1]) / len(epoch_acc_d[:, 1])
    args.acc_cl_real = sum(epoch_acc_d[:, 2]) / len(epoch_acc_d[:, 2])
    args.acc_cl_fake = sum(epoch_acc_d[:, 3]) / len(epoch_acc_d[:, 3])

    # Tensorboard
    if trainValid:
        tag = 'train'
    else:
        tag = 'valid'

    if not inference:
        args.writer.add_scalar("Loss_G/{}".format(tag), args.loss_g, epoch)
        args.writer.add_scalar("Loss_G_recon/{}".format(tag), args.loss_g_recon, epoch)
        args.writer.add_scalar("Loss_G_valid/{}".format(tag), args.loss_g_valid, epoch)

        args.writer.add_scalar("ACC_D_real/{}".format(tag), args.acc_d_real, epoch)
        args.writer.add_scalar("ACC_D_fake/{}".format(tag), args.acc_d_fake, epoch)

    print(
        '\n[%3d/%3d] ACC_R: %.4f ACC_F: %.4f / g-RMSE: %.4f g-lossValid: %.4f'
        % (i, total_batches,
           args.acc_d_real, args.acc_d_fake,
           args.loss_g_recon, args.loss_g_valid))

    return (
        args.loss_g, args.loss_g_recon, args.loss_g_valid, args.acc_g_valid,
        args.loss_d, args.acc_d_real, args.acc_d_fake)


def train_g(args, input_data, target_data, voice, labels, models, criterions, optimizer_g, data_info, trainValid):
    (model_g, model_d) = models
    (criterion_recon, criterion_adv, _) = criterions

    if trainValid:
        model_g.train()
        model_d.train()
    else:
        model_g.eval()
        model_d.eval()

    # Adversarial ground truths 1: real, 0: fake
    valid = torch.ones((len(input_data), 1), dtype=torch.float32).cuda()

    ###############################
    # Train Generator
    ###############################

    if trainValid:
        for p in model_g.parameters():
            p.requires_grad_(True)  # unfreeze G
        for p in model_d.parameters():
            p.requires_grad_(False)  # freeze D

        # set zero grad
        optimizer_g.zero_grad()

        # Run Generator
        output = model_g(input_data)
    else:
        with torch.no_grad():
            # run generator
            output = model_g(input_data)

    # DTW
    # TODO: Mel이 아니라 waveform 기준으로 코드 수정 -> waveform에 DTW가 최선인지 고민
    waveform_out = output.clone()
    waveform_out = DTW_align(waveform_out, target_data)

    # Run Discriminator
    g_valid, _ = model_d(waveform_out)

    # Generator loss
    loss_recon = criterion_recon(waveform_out, target_data)

    # Discriminator loss
    loss_valid = criterion_adv(g_valid, valid)

    # accuracy    args.l_g = h_g.l_g
    acc_g_valid = (g_valid.round() == valid).float().mean()

    # Total generator loss
    loss_g = args.l_g[0] * loss_recon + args.l_g[1] * loss_valid

    if trainValid:
        loss_g.backward()
        optimizer_g.step()

    e_loss_g = (loss_g.item(), loss_recon.item(), loss_valid.item())
    e_acc_g = acc_g_valid.item()

    return waveform_out, e_loss_g, e_acc_g


def train_d(args, waveform_out, target_data, target_data_cl, labels, models, criterions, optimizer_d, trainValid):
    (_, model_d, _, _, _) = models
    (_, _, criterion_adv, criterion_cl, _) = criterions

    if trainValid:
        model_d.train()
    else:
        model_d.eval()

    # Adversarial ground truths 1:real, 0: fake
    valid = torch.ones((len(waveform_out), 1), dtype=torch.float32).cuda()
    fake = torch.zeros((len(waveform_out), 1), dtype=torch.float32).cuda()

    # Train Discriminator
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
    real_valid, real_cl = model_d(target_data)
    fake_valid, fake_cl = model_d(waveform_out.detach())

    loss_d_real_valid = criterion_adv(real_valid, valid)
    loss_d_fake_valid = criterion_adv(fake_valid, fake)
    loss_d_real_cl = criterion_cl(real_cl, target_data_cl)

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


def saveData(args, test_loader, models, epoch, losses):  # TODO: test_loader 출처 확인
    model_g = models[0].eval()
    # model_d = models[1].eval()  # annotation from original code

    input_data, target_data, target_data_cl, voice, data_info = next(iter(test_loader))

    input_data = input_data.cuda()
    target_data = target_data.cuda()
    voice = torch.squeeze(voice, dim=-1).cuda()
    labels = torch.argmax(target_data_cl, dim=1)

    with torch.no_grad():
        # run the mdoel
        output = model_g(input_data)  # generator에 넣어 학습 시작

    mel_out = DTW_align(output, target_data)
    wav_generated = data_denorm(mel_out, data_info[0], data_info[1])
    wav_generated = torch.reshape(wav_generated, (len(wav_generated), wav_generated.shape[-1]))

    wav_generated = torchaudio.functional.resample(wav_generated, args.sample_rate_mel)
    # 차원 불일치 에러 핸들링
    if wav_generated.shape[1] != voice.shape[1]:
        p = voice.shape[1] - wav_generated.shape[1]
        p_s = p // 2
        p_e = p - p_s
        wav_generated = F.pad(wav_generated, (p_s, p_e))

    # Save (Squeeze & Naming)
    wav_generated = np.squeeze(wav_generated.cpu().detach().numpy())

    str_tar = args.word_label[labels[0].item()].replace("|", ",")
    str_tar = str_tar.replace(" ", ",")

    title = "Tar_{}".format(str_tar)
    wavio.write(args.savevoice + '/e{}_{}.wav'.format(str(str(epoch)), title), wav_generated, 44100, sampwidth=1)


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

