class DefaultConfig(object):
    train_root = '/root/autodl-tmp/trainout4.h5'
    validation_root = '/root/autodl-tmp/evalout4.h5'
    lr = 0.0001
    batch_size = 16
    num_workers = 16
    epoch = 400
    outputs_dir = '/root/autodl-tmp/'
    cuda = True
    # opts1 = {
    #     'title': 'train_loss',
    #     'xlabel': 'epoch',
    #     'ylabel': 'loss',
    #     'width': 300,
    #     'height': 300,
    # }

    # opts2 = {
    #     'title': 'eval_psnr',
    #     'xlabel': 'epoch',
    #     'ylabel': 'psnr',
    #     'width': 300,
    #     'height': 300,
    # }

opt = DefaultConfig()