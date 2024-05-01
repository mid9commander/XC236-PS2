import argparse
import numpy as np
import torch
import tqdm
import importlib.util

use_submission = importlib.util.find_spec('submission') is not None
if use_submission:
    from submission import utils as ut
    from submission.models.vae import VAE
    from submission.models.gmvae import GMVAE
    from submission.models.ssvae import SSVAE
    from submission.models.fsvae import FSVAE
    from submission.train import train
else:
    from solution import utils as ut
    from solution.models.vae import VAE
    from solution.models.gmvae import GMVAE
    from solution.models.ssvae import SSVAE
    from solution.models.fsvae import FSVAE
    from solution.train import train

import matplotlib.pyplot as plt

SAMPLES = 200

def generate_mnist_samples(model, model_name, num_samples=SAMPLES):
    sampled_images = model.sample_x(num_samples).detach().cpu().numpy()
    grid_shape = (10, num_samples // 10)
    img_dim = 28 # height and width are the same for MNIST
    sampled_images_reshaped = sampled_images.reshape(*grid_shape, img_dim, img_dim)
    tiled_image = np.swapaxes(sampled_images_reshaped, 1, 2)
    tiled_image = tiled_image.reshape(grid_shape[0] * img_dim, grid_shape[1] * img_dim)
    plt.imshow(tiled_image, cmap='gray')
    plt.axis('off')
    plt.savefig(f'{model_name}.png')
    plt.show()

def generate_svhn_samples(model, model_name, num_samples=SAMPLES):
    latent_zs = model.sample_z(num_samples // 10)
    images = []

    for i in range(10):
        for z in latent_zs:
            y = torch.zeros(10).to(z.device) # one hot vector for class i
            y[i] = 1
            image = model.compute_mean_given(z.unsqueeze(0), y.unsqueeze(0))
            images.append(image.squeeze().detach().cpu().numpy())
    num_rows, num_cols = 10, num_samples // 10
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    for img,ax in zip(images, axes.ravel()):
        # reshape image to 3 * 32 * 32 and then transpose to 32 * 32 *3 for visualization
        img_reshaped = img.reshape(3, 32, 32).transpose(1, 2, 0)
        ax.imshow(np.clip(img_reshaped, 0, 1))
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{model_name}.png')
    plt.show()

def main(args):
    if args.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device: ", device)

    # VAE
    if args.model == "vae":
        layout = [
            ('model={:s}',  args.model),
            ('z={:02d}',  args.z),
            ('run={:04d}', args.run)
        ]
        model_name = '_'.join([t.format(v) for (t, v) in layout])
        print('Model name:', model_name)

        # Scale the #iterations proportional to the batch size
        iter_max = args.iter_max if args.batch_size == 100 else args.iter_max // (args.batch_size // 100)
        iter_save = args.iter_save if args.batch_size == 100 else args.iter_save // (args.batch_size // 100)

        train_loader, labeled_subset, _ = ut.get_mnist_data(device, args.batch_size, use_test_subset=True)
        vae = VAE(z_dim=args.z, name=model_name).to(device)
        if args.train:
            writer = ut.prepare_writer(model_name, overwrite_existing=args.overwrite)
            train(model=vae,
                train_loader=train_loader,
                labeled_subset=labeled_subset,
                device=device,
                tqdm=tqdm.tqdm,
                writer=writer,
                iter_max=iter_max,
                iter_save=iter_save)
        else:
            ut.load_model_by_name(vae, global_step=iter_max, device=device)
            generate_mnist_samples(vae, model_name, SAMPLES)
        ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=args.iwae)
    # GMVAE
    elif args.model == "gmvae":
        layout = [
            ('model={:s}',  args.model),
            ('z={:02d}',  args.z),
            ('k={:03d}',  args.k),
            ('run={:04d}', args.run)
        ]
        model_name = '_'.join([t.format(v) for (t, v) in layout])
        print('Model name:', model_name)

        # Scale the #iterations proportional to the batch size
        iter_max = args.iter_max if args.batch_size == 100 else args.iter_max // (args.batch_size // 100)
        iter_save = args.iter_save if args.batch_size == 100 else args.iter_save // (args.batch_size // 100)
        
        train_loader, labeled_subset, _ = ut.get_mnist_data(device, args.batch_size, use_test_subset=True)
        gmvae = GMVAE(z_dim=args.z, k=args.k, name=model_name).to(device)

        if args.train:
            writer = ut.prepare_writer(model_name, overwrite_existing=args.overwrite)
            train(model=gmvae,
                train_loader=train_loader,
                labeled_subset=labeled_subset,
                device=device,
                tqdm=tqdm.tqdm,
                writer=writer,
                iter_max=iter_max,
                iter_save=iter_save)
        else:
            ut.load_model_by_name(gmvae, global_step=iter_max, device=device)
            generate_mnist_samples(gmvae, model_name, SAMPLES)
        ut.evaluate_lower_bound(gmvae, labeled_subset, run_iwae=args.iwae)
    # SSVAE
    elif args.model == "ssvae":
        layout = [
            ('model={:s}',  args.model),
            ('gw={:03d}', args.gw),
            ('cw={:03d}', args.cw),
            ('run={:04d}', args.run)
        ]
        model_name = '_'.join([t.format(v) for (t, v) in layout])
        print('Model name:', model_name)

        # Logic to change max_iter
        iter_max = 30000 if args.iter_max == 20000 else args.iter_max

        # Scale the #iterations proportional to the batch size
        iter_max = iter_max if args.batch_size == 100 else iter_max // (args.batch_size // 100)
        iter_save = args.iter_save if args.batch_size == 100 else args.iter_save // (args.batch_size // 100)
        

        train_loader, labeled_subset, test_set = ut.get_mnist_data(device, args.batch_size, use_test_subset=False)
        ssvae = SSVAE(gen_weight=args.gw,
                    class_weight=args.cw,
                    name=model_name).to(device)
        
        if args.train:
            writer = ut.prepare_writer(model_name, overwrite_existing=args.overwrite)
            train(model=ssvae,
                train_loader=train_loader,
                labeled_subset=labeled_subset,
                device=device,
                y_status='semisup',
                tqdm=tqdm.tqdm,
                writer=writer,
                iter_max=iter_max,
                iter_save=iter_save)
        else:
            ut.load_model_by_name(ssvae, iter_max, device=device)
        ut.evaluate_classifier(ssvae, args.gw, test_set)
    # FSVAE
    elif args.model == "fsvae":
        layout = [
            ('model={:s}',  'fsvae'),
            ('run={:04d}', args.run)
        ]
        model_name = '_'.join([t.format(v) for (t, v) in layout])
        print('Model name:', model_name)

        # Logic to change max_iter
        iter_max = 1000000 if args.iter_max == 20000 else args.iter_max

        # Scale the #iterations proportional to the batch size
        iter_max = iter_max if args.batch_size == 100 else iter_max // (args.batch_size // 100)
        iter_save = args.iter_save if args.batch_size == 100 else args.iter_save // (args.batch_size // 100)

        train_loader, labeled_subset, test_set = ut.get_svhn_data(args.batch_size)
        fsvae = FSVAE(name=model_name).to(device)

        if args.train:
            writer = ut.prepare_writer(model_name, overwrite_existing=args.overwrite)
            train(model=fsvae,
                train_loader=train_loader,
                labeled_subset=labeled_subset,
                device=device,
                y_status='fullsup',
                tqdm=tqdm.tqdm,
                writer=writer,
                iter_max=iter_max,
                iter_save=iter_save)
        else:
            # NOTE: using midway point of training to generate samples
            # feel free to use another savepoint to generate samples
            global_step = iter_save
            ut.load_model_by_name(fsvae, global_step, device=device)
            generate_svhn_samples(fsvae, model_name, SAMPLES)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="vae", choices=['vae', 'ssvae', 'gmvae', 'fsvae'], help="Model to run")   
    parser.add_argument('--overwrite', type=int, default=1, help="Flag for overwriting")
    parser.add_argument('--run', type=int, default=0, help="Run ID. In case you want to run replicates")
    parser.add_argument('--train', action='store_true', help="Flag needed to start training")
    parser.add_argument('--cache', action='store_true', help="Cache MNIST and SVHN data to avoid redownloading")
    parser.add_argument('--iwae', action='store_true', help="Adds IWAE")
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'gpu'], help="GPU or CPU acceleration")
    parser.add_argument('--batch_size', type=int, default=100, help='number of data points to be processed per batch')

    # NOTE: iter_max for SSVAE will have a default of 30000
    # NOTE: iter_max for FSVAE will have a default of 1000000
    parser.add_argument('--z', type=int, default=10, help="Number of latent dimensions")
    parser.add_argument('--iter_max', type=int, default=20000, help="Number of training iterations")
    parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")

    # Args needed for GMVAE
    parser.add_argument('--k', type=int, default=500, help="Number mixture components in MoG prior")

    # Args needed for SSVAE
    parser.add_argument('--gw', type=int, default=1, help="Weight on the generative terms")
    parser.add_argument('--cw', type=int, default=100, help="Weight on the class term")

    args = parser.parse_args()

    if args.cache == True:
        _, _, _ = ut.get_mnist_data(torch.device("cpu"), args.batch_size, use_test_subset=True)
        _, _, _ = ut.get_svhn_data(args.batch_size)
    else:
        main(args)

