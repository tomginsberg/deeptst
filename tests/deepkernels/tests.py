import torch
import numpy


def deep_kernel_tests(K=10):
    adversarial_loss = torch.nn.CrossEntropyLoss()

    # Repeat experiments K times (K = 10) and report average test power (rejection rate)
    for kk in range(K):
        torch.manual_seed(kk * 19 + N1)
        torch.cuda.manual_seed(kk * 19 + N1)
        np.random.seed(seed=1102 * (kk + 10) + N1)
        # Initialize deep networks for MMD-D (called featurizer), C2ST-S and C2ST-L (called discriminator)
        featurizer = Featurizer()
        discriminator = Discriminator()
        # Initialize parameters
        epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
        sigma0OPT.requires_grad = True
        print(epsilonOPT.item())
        if cuda:
            featurizer = featurizer.cuda()
            discriminator = discriminator.cuda()
            adversarial_loss = adversarial_loss.cuda()

        # Collect CIFAR10 images
        Ind_tr = np.random.choice(len(data_all), N1, replace=False)
        Ind_te = np.delete(Ind_all, Ind_tr)
        train_data = []
        for i in Ind_tr:
            train_data.append([data_all[i], label_all[i]])

        dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=opt.batch_size,
            shuffle=True,
        )

        # Collect CIFAR10.1 images
        np.random.seed(seed=819 * (kk + 9) + N1)
        Ind_tr_v4 = np.random.choice(len(data_T), N1, replace=False)
        Ind_te_v4 = np.delete(Ind_v4_all, Ind_tr_v4)
        New_CIFAR_tr = data_trans[Ind_tr_v4]
        New_CIFAR_te = data_trans[Ind_te_v4]

        # Initialize optimizers
        optimizer_F = torch.optim.Adam(list(featurizer.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT],
                                       lr=0.0002)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # ----------------------------------------------------------------------------------------------------
        #  Training deep networks for MMD-D (called featurizer), C2ST-S and C2ST-L (called discriminator)
        # ----------------------------------------------------------------------------------------------------
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        for epoch in range(opt.n_epochs):
            for i, (imgs, _) in enumerate(dataloader):
                if True:
                    ind = np.random.choice(N1, imgs.shape[0], replace=False)
                    Fake_imgs = New_CIFAR_tr[ind]
                    # Adversarial ground truths
                    valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                    fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                    # Configure input
                    real_imgs = Variable(imgs.type(Tensor))
                    Fake_imgs = Variable(Fake_imgs.type(Tensor))
                    X = torch.cat([real_imgs, Fake_imgs], 0)
                    Y = torch.cat([valid, fake], 0).squeeze().long()

                    # ------------------------------
                    #  Train deep network for MMD-D
                    # ------------------------------
                    # Initialize optimizer
                    optimizer_F.zero_grad()
                    # Compute output of deep network
                    modelu_output = featurizer(X)
                    # Compute epsilon, sigma and sigma_0
                    ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
                    sigma = sigmaOPT ** 2
                    sigma0_u = sigma0OPT ** 2
                    # Compute Compute J (STAT_u)
                    TEMP = MMDu(modelu_output, imgs.shape[0], X.view(X.shape[0], -1), sigma, sigma0_u, ep)
                    mmd_value_temp = -1 * (TEMP[0])
                    mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                    STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                    # Compute gradient
                    STAT_u.backward()
                    # Update weights using gradient descent
                    optimizer_F.step()

                    # ------------------------------------------
                    #  Train deep network for C2ST-S and C2ST-L
                    # ------------------------------------------
                    # Initialize optimizer
                    optimizer_D.zero_grad()
                    # Compute Cross-Entropy (loss_C) loss between two samples
                    loss_C = adversarial_loss(discriminator(X), Y)
                    # Compute gradient
                    loss_C.backward()
                    # Update weights using gradient descent
                    optimizer_D.step()
                    if (epoch + 1) % 100 == 0:
                        print(
                            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [Stat: %f]"
                            % (epoch, opt.n_epochs, i, len(dataloader), loss_C.item(), -STAT_u.item())
                        )
                    batches_done = epoch * len(dataloader) + i
                else:
                    break

        # Run two-sample_data test on the training set
        # Fetch training data
        s1 = data_all[Ind_tr]
        s2 = data_trans[Ind_tr_v4]
        S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
        Sv = S.view(2 * N1, -1)
        # Run two-sample_data test (MMD-D) on the training set
        h_u, threshold_u, mmd_value_u = TST_MMD_u(featurizer(S), N_per, N1, Sv, sigma, sigma0_u, ep, alpha, device,
                                                  dtype)
        # Run two-sample_data test (C2STs) on the training set
        h_C2ST_S, threshold_C2ST_S, s_C2ST_S = TST_C2ST_D(S, N1, N_per, alpha, discriminator, device, dtype)
        h_C2ST_L, threshold_C2ST_L, s_C2ST_L = TST_LCE_D(S, N1, N_per, alpha, discriminator, device, dtype)

        # Train MMD-O
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        Dxy = Pdist2(Sv[:N1, :], Sv[N1:, :])
        sigma0 = Dxy.median()
        sigma0.requires_grad = True
        optimizer_sigma0 = torch.optim.Adam([sigma0], lr=0.0005)
        for t in range(opt.n_epochs):
            TEMPa = MMDu(Sv, N1, Sv, sigma, sigma0, is_smooth=False)
            mmd_value_tempa = -1 * (TEMPa[0] + 10 ** (-8))
            mmd_std_tempa = torch.sqrt(TEMPa[1] + 10 ** (-8))
            STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
            optimizer_sigma0.zero_grad()
            STAT_adaptive.backward(retain_graph=True)
            optimizer_sigma0.step()
            if t % 100 == 0:
                print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ",
                      -1 * STAT_adaptive.item())
        h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(Sv, N_per, N1, Sv, sigma,
                                                                                        sigma0,
                                                                                        alpha,
                                                                                        device, dtype)
        print("h:", h_adaptive, "Threshold:", threshold_adaptive, "MMD_value:", mmd_value_adaptive)

        # Train ME
        np.random.seed(seed=1102)
        test_locs_ME, gwidth_ME = TST_ME(Sv, N1, alpha, is_train=True, test_locs=1, gwidth=1, J=5, seed=15)
        h_ME = TST_ME(Sv, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=5, seed=15)

        # Train SCF
        np.random.seed(seed=1102)
        test_freqs_SCF, gwidth_SCF = TST_SCF(Sv, N1, alpha, is_train=True, test_freqs=1, gwidth=1, J=5, seed=15)
        h_SCF = TST_SCF(Sv, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=5, seed=15)

        # Record best epsilon, sigma and sigma_0
        ep_OPT[kk] = ep.item()
        s_OPT[kk] = sigma.item()
        s0_OPT[kk] = sigma0_u.item()

        # Compute test power of MMD-D and baselines
        H_u = np.zeros(N)
        T_u = np.zeros(N)
        M_u = np.zeros(N)
        H_adaptive = np.zeros(N)
        T_adaptive = np.zeros(N)
        M_adaptive = np.zeros(N)
        H_ME = np.zeros(N)
        H_SCF = np.zeros(N)
        H_C2ST_S = np.zeros(N)
        Tu_C2ST_S = np.zeros(N)
        S_C2ST_S = np.zeros(N)
        H_C2ST_L = np.zeros(N)
        Tu_C2ST_L = np.zeros(N)
        S_C2ST_L = np.zeros(N)
        np.random.seed(1102)
        count_u = 0
        count_adp = 0
        count_ME = 0
        count_SCF = 0
        count_C2ST_S = 0
        count_C2ST_L = 0
        for k in range(N):
            # Fetch test data
            np.random.seed(seed=1102 * (k + 1) + N1)
            data_all_te = data_all[Ind_te]
            N_te = len(data_trans) - N1
            Ind_N_te = np.random.choice(len(Ind_te), N_te, replace=False)
            s1 = data_all_te[Ind_N_te]
            s2 = data_trans[Ind_te_v4]
            S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
            Sv = S.view(2 * N_te, -1)
            # MMD-D
            h_u, threshold_u, mmd_value_u = TST_MMD_u(featurizer(S), N_per, N_te, Sv, sigma, sigma0_u, ep, alpha,
                                                      device,
                                                      dtype)
            # MMD-O
            h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(Sv, N_per, N_te, Sv, sigma,
                                                                                            sigma0, alpha, device,
                                                                                            dtype)
            # ME
            h_ME = TST_ME(Sv, N_te, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=10, seed=15)
            # SCF
            h_SCF = TST_SCF(Sv, N_te, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=10,
                            seed=15)
            # C2ST-S
            H_C2ST_S[k], Tu_C2ST_S[k], S_C2ST_S[k] = TST_C2ST_D(S, N1, N_per, alpha, discriminator, device, dtype)
            # C2ST-L
            H_C2ST_L[k], Tu_C2ST_L[k], S_C2ST_L[k] = TST_LCE_D(S, N1, N_per, alpha, discriminator, device, dtype)

            # Gather results
            count_u = count_u + h_u
            count_adp = count_adp + h_adaptive
            count_ME = count_ME + h_ME
            count_SCF = count_SCF + h_SCF
            count_C2ST_S = count_C2ST_S + int(H_C2ST_S[k])
            count_C2ST_L = count_C2ST_L + int(H_C2ST_L[k])
            print("MMD-DK:", count_u, "MMD-OPT:", count_adp, "MMD-ME:", count_ME, "SCF:", count_SCF, "C2ST_S: ",
                  count_C2ST_S, "C2ST_L: ", count_C2ST_L)
            H_u[k] = h_u
            T_u[k] = threshold_u
            M_u[k] = mmd_value_u
            H_adaptive[k] = h_adaptive
            T_adaptive[k] = threshold_adaptive
            M_adaptive[k] = mmd_value_adaptive
            H_ME[k] = h_ME
            H_SCF[k] = h_SCF

        # Print test power of MMD-D and baselines
        print("Reject rate_u: ", H_u.sum() / N_f, "Reject rate_C2ST-L: ", H_C2ST_L.sum() / N_f, "Reject rate_C2ST-S: ",
              H_C2ST_S.sum() / N_f, "Reject rate_adaptive: ",
              H_adaptive.sum() / N_f, "Reject rate_ME: ", H_ME.sum() / N_f, "Reject rate_SCF: ", H_SCF.sum() / N_f)
        Results[0, kk] = H_u.sum() / N_f
        Results[1, kk] = H_C2ST_L.sum() / N_f
        Results[2, kk] = H_C2ST_S.sum() / N_f
        Results[3, kk] = H_adaptive.sum() / N_f
        Results[4, kk] = H_ME.sum() / N_f
        Results[5, kk] = H_SCF.sum() / N_f
        print("Test Power of Baselines (K times): ")
        print(Results)
        print("Average Test Power of Baselines (K times): ")
        print("MMD-D: ", (Results.sum(1) / (kk + 1))[0], "C2ST-L: ", (Results.sum(1) / (kk + 1))[1],
              "C2ST-S: ", (Results.sum(1) / (kk + 1))[2], "MMD-O: ", (Results.sum(1) / (kk + 1))[3])
