import torch
import torch.nn as nn
import numpy as np


def sample_brownian_bridge(start, stop, n_step):
    """
    start:
        [1, C, H, W]
    stop:
        [1, C, H, W]
    n_step:

    example
    ```
    d1 = 10
    d2 = 20
    B = make_brownian_bridge(np.random.randn(d1,d2), np.random.randn(d1,d2), 10)
    plt.plot(B.reshape(10,-1))
    plt.axhline(0, ls='--', c='k')
    ```
    """
    assert start.shape == stop.shape
    d = start.numel()
    dt = torch.tensor(1/(n_step - 1))
    tt = torch.linspace(0, 1, n_step)[:, None]

    dW = torch.sqrt(dt) * torch.randn(n_step, d)
    dW[0] = start.flatten()
    W = torch.cumsum(dW, axis=0)

    B = W - tt * (W[-1] - stop.flatten())[None, :]

    # print(torch.norm(((W - tt * W[-1][None, :])[-1])) < 1e-6)
    # print(torch.norm((W - W[-1:])[-1]) < 1e-6)
    # print(torch.norm(B[-1] - stop.flatten())) # why less exact?
    B = torch.reshape(B, (n_step, *start.shape[1:]))

    assert torch.norm(B[0] - start) < 1e-6
    assert torch.norm(B[-1] - stop) < 1e-5 # why less exact?
    return B


def make_straight_line(start, stop, n_step):
    """
    start:
        [1, C, H, W]
    stop:
        [1, C, H, W]
    n_step:

    """
    assert start.shape == stop.shape
    d = start.numel()
    tt = torch.linspace(0, 1, n_step).unsqueeze(1).unsqueeze(1).unsqueeze(1)

    return  (1 - tt) * start + tt * stop
    # x = torch.Tensor(self.n_steps-2, *self.image_size[1:])
    # for i in range(self.n_steps-2):
    #     t = (i+1)/(self.n_steps-1)
    #     x[i].copy_(self.xA[0] * (1 - t)+(t * self.xB[0]))


def get_angles_dist_accel(x):
    """
    under dev


    in:  B  T X Y
    out: theta B T-2
         dist  B T-1
         accel B T-2 X Y

    """
    B, T, X, Y = x.shape

    x = x.view((B, T, -1))
    v = x[:,0:T-1] - x[:,1:T]
    d = torch.norm(v, dim = 2, keepdim=True)
    v_hat = torch.div(v, d)

    theta = torch.empty((B,T-2))
    accel = torch.empty((B,T-2, X * Y))

    for t in range(T-2):
        theta[:,t] = torch.acos(
                     torch.bmm(v_hat[:, t].view((B,1,X * Y)),
                               v_hat[:, t+1].view((B,X * Y,1))).squeeze()
                                ) / torch.tensor(np.pi)
        accel[:,t] = v_hat[:, t] - torch.bmm(v_hat[:, t].view((B,1,-1)),
                                   v_hat[:, t+1].view((B,-1,1)))[...,0] * v_hat[:, t]

    accel_hat = torch.div(accel, torch.norm(accel, dim = 2, keepdim=True))
    accel_hat = accel_hat.view((B, T-2, X, Y))

    return theta, d, accel_hat


def Haar_1d(x, n_scales=None):
    """
    tool for multiscale geodesic

    in: B 1 T X Y
    Haar decomposition along T axis

    todo:
    work with [T,C,H,W]
    use functionals, to avoid parameters
    """

    if n_scales is None:
        n_scales = int(np.log2(x.shape[-3]))

    diff = nn.Conv3d(1,1,(2,1,1), bias=False)
    blur = nn.Conv3d(1,1,(2,1,1), bias=False, stride=(2,1,1))

    diff.weight = nn.Parameter(torch.ones_like(diff.weight))
    diff.weight.select(2,0).mul_(-1) # padding = 1 and pop ?
    blur.weight = nn.Parameter(torch.ones_like(diff.weight))

    y = []
    for s in range(n_scales):
#         print(s, x.shape)
        y.append(diff(x))
        x = blur(x)

    return y



# def get_distance_angle_accel(z):
#     '''
#     z     : array [P, N]
#         location in feature space, P dimensional sequence of length N
#     ---
#     d     : distance b/w adjacent frames, N-1 step lengths
#     theta : curvature, N-2 angles, between 0 and 1
#     accel : acceleration, N-2 P dimensional unit vectors
#     '''
#
#     P,N   = np.shape(z)
#     v     = np.diff(z)
#
#     d     = np.linalg.norm(v, axis=0)
#     v_hat = v/d
#
#     theta = np.zeros(N-2)
#     accel = np.zeros((P,N-2))
#
#     for t in range(N - 2):
#
#         theta[t] = np.arccos(v_hat[:,t].dot(v_hat[:,t+1])) / np.pi
#         # multiply this number by 180 to get the angle in degrees
#
#         accel[:,t] = v_hat[:,t+1] - v_hat[:,t+1].dot(v_hat[:,t])* v_hat[:,t]
#
#     # TODO
#     accel /= np.linalg.norm(accel, axis=0)
#
#     return d, theta, accel
#
# def make_trajectory(d, theta, accel, z0=None, v0=None):
#
#     P,N         = np.shape(accel)
#     N          += 2
#
#     theta_hat   = theta * np.pi
#
#     z           = np.zeros((P,N))
#     v_hat       = np.zeros((P,N-1))
#
#     if v0 is not None:
#         v_hat[:,0]  = v0
#
#     for t in range(N-2):
#
#         v_hat[:,t+1] = np.cos(theta_hat[t]) * v_hat[:,t] \
#                      + np.sin(theta_hat[t]) * accel[:,t]
#
#     v = v_hat * d[None,:]
#
#     if z0 is not None:
#         z[:,0] = z0
#
#     z = np.cumsum(np.hstack((z0[:,None], v)),axis=1)
#
#     return z

# def test_curvature():
    #
    # from PIL import Image
    # import numpy
    #
    # savedir = 'fig/test_curvature'
    #
    # from utilities import saveImg, make_disk
    #
    # import matplotlib
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    #
    # #### Pixel Domain
    #
    # vid = makeGroundtruth()
    # # print('vid shape')
    # # print(vid.shape)
    #
    # Z = vid.numpy().reshape((params.imgSize**2,params.nsmpl))
    # d_pix,theta_pix,accel_pix = get_distance_angle_accel(Z)
    # # print('\n')
    # # print('mean pixel curvature')
    # # print(theta_pix.mean())
    #
    # ################
    # # Scattering 1 #
    # ################
    #
    # network = SteerableScattering( imgSize=params.imgSize,
    #                                 N=params.N, K=params.K, O=1,
    #                                 pooling=False)
    #
    # x = network( vid.cuda() )
    # # print('\n')
    # # print('output shape for a batch, scatt 2')
    # # for i in range( len(x) ):
    # #     print( x[i].size() )
    #
    # # flatten
    # y = torch.cat([x[i].view(x[i].size(0), -1) for i in range(len(x))], 1)
    # # print(y.shape)
    #
    #
    # # if pooling = True, SteerableScattering output has shape
    # # [nsmpl, nbands = N * K + 2, imgSize = 2 ** (imgScale - N)]
    # # for example:
    # # [11   ,         14        ,     16      ,        16      ]
    #
    # Z = y.data.cpu().numpy().reshape((params.nsmpl, -1)).T
    # d_scatt,theta_scatt,accel_scatt = get_distance_angle_accel(Z)
    # # print('\n')
    # # print('mean scatt1 curvature')
    # # print(theta_scatt.mean())
    #
    # # further descriptive plots
    # img    = y[5].data # to look at example frame
    # curv   = []          # to store mean curvature in each band
    # energy = np.zeros((y.size(0), y.size(1)))  # to store energy in each band
    #
    # for i in range(y.size(1)):
    #     # saveImg(img[i],'scatt1_band' + str(i) + str(params.imgName))
    #
    #     for t in range(params.nsmpl):
    #         energy[t,i] = (y[t][i].data ** 2).sum()
    #
    #     Z = y.select(1,i).data.cpu().numpy().reshape((-1, params.nsmpl))
    #     _,theta_band,_ = get_distance_angle_accel(Z)
    #     curv.append(theta_band.mean())
    #
    # fig, ax = plt.subplots()
    # plt.semilogy(energy.mean(0), 'o')
    # ax.spines['top'  ].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.xlabel('band')
    # plt.ylabel('mean energy')
    # plt.savefig(savedir + 'scatt1_energy_bands_'+str(params.imgName)+'.png')
    #
    # # print('\n')
    # # print('Check preserved energy')
    # # for t in range(params.nsmpl):
    # #     inp = ( (y[t].data ** 2).sum() / (params.imgSize ** 2))
    # #     out = ( energy[t,:].sum()      / (params.imgSize ** 2))
    # #     print(np.allclose(inp,out))
    #
    # fig, ax = plt.subplots()
    # plt.plot( curv, 'o')
    # plt.ylim([0,1])
    # ax.spines['top'  ].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.ylabel('mean curvature (time $\pi$)')
    # plt.xlabel('band')
    # plt.savefig(savedir + 'scatt1_curvature_bands_'+str(params.imgName)+'.png')
    #
    # ################
    # # Scattering 2 #
    # ################
    #
    # network = SteerableScattering( imgSize=params.imgSize,
    #                                 N=params.N, K=params.K, O=2,
    #                                 pooling=False)
    #
    # x = network( Variable(vid.cuda()) )
    # # print('\n')
    # # print('output shape for a batch, scatt 2')
    # # for i in range( len(x) ):
    # #     print( x[i].size() )
    # y = torch.cat([x[i].view(x[i].size(0), -1) for i in range(len(x))], 1)
    # # print(y.shape)
    #
    # Z = y.data.cpu().numpy().reshape((params.nsmpl, -1)).T
    # d_scatt2,theta_scatt2,accel_scatt2 = get_distance_angle_accel(Z)
    # # print('\n')
    # # print('mean scatt2 curvature')
    # # print(theta_scatt2.mean())
    #
    #
    # # further descriptive plots
    # img    = y[5].data # to look at example frame
    # curv   = []          # to store mean curvature in each band
    # energy = np.zeros((y.size(0), y.size(1)))  # to store energy in each band
    #
    # for i in range(y.size(1)):
    #     # saveImg(img[i],'scatt2_band' + str(i) + str(params.imgName))
    #
    #     for t in range(params.nsmpl):
    #         energy[t,i] = (y[t][i].data ** 2).sum()
    #
    #     Z = y.select(1,i).data.cpu().numpy().reshape((-1, params.nsmpl))
    #     _,theta_band,_ = get_distance_angle_accel(Z)
    #     curv.append(theta_band.mean())
    #
    # fig, ax = plt.subplots()
    # plt.semilogy(energy.mean(0), 'o')
    # ax.spines['top'  ].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.xlabel('band')
    # plt.ylabel('mean energy')
    # plt.savefig(savedir + 'scatt2_energy_bands_'+str(params.imgName)+'.png')
    #
    # # print('\n')
    # # print('Check preserved energy')
    # # for t in range(params.nsmpl):
    # #     inp = ( (y[t].data ** 2).sum() / (params.imgSize ** 2))
    # #     out = ( energy[t,:].sum()      / (params.imgSize ** 2))
    # #     print(np.allclose(inp,out))
    #
    # fig, ax = plt.subplots()
    # plt.plot( curv, 'o')
    # plt.ylim([0,1])
    # ax.spines['top'  ].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.ylabel('mean curvature (time $\pi$)')
    # plt.xlabel('band')
    # plt.savefig(savedir + 'scatt2_curvature_bands_'+str(params.imgName)+'.png')
    #
    #
    # ############
    # # PLOTTING #
    # ############
    #
    # fig, ax = plt.subplots()
    # plt.plot( range(1, params.nsmpl - 1), theta_pix    , label = 'pix'   )
    # plt.plot( range(1, params.nsmpl - 1), theta_scatt  , label = 'scatt1')
    # plt.plot( range(1, params.nsmpl - 1), theta_scatt2 , label = 'scatt2')
    # plt.ylim([0,1])
    # plt.ylabel('curvature (time $\pi$)')
    # plt.xlabel('frame index')
    # plt.legend(loc = 'best')
    # ax.spines['top'  ].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # fig.savefig(savedir + 'curvatures_'+str(params.imgName)+'.png' )
