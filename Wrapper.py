import os
from typing import Optional
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange
import json
import cv2
torch.set_default_dtype(torch.float32)
from skimage.metrics import structural_similarity as ssim
from nerf_utils import cumprod_exclusive, get_minibatches, positional_encoding

TOY = 'lego'
# TOY = 'ship'

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def make_all_dirs():
    dirs_list = [
    './saved_models',
    './saved_models/lego',
    './saved_models/ship',
    './Dataset'
    ]

    for dir in dirs_list:
        create_directory(dir)


def test(img_save_path='./saved_images'):

    coarse_model = TinyNerf(num_encoding_functions=10)
    fine_model = TinyNerf(num_encoding_functions=10)

    coarse_model_path = f'./saved_models/lego positional_unencoding/coarse_model.pth'

    coarse_model.load_state_dict(torch.load(coarse_model_path))
    

    fine_model_path = f'./saved_models/lego positional_unencoding/fine_model.pth'
    fine_model.load_state_dict(torch.load(fine_model_path))

    coarse_model.to(dtype=torch.float32, device='cuda')
    fine_model.to(dtype=torch.float32, device='cuda')

    coarse_model.eval()
    fine_model.eval()


    test_images, _, focal_length, poses = load_train_test_val_data(data_path=f'./Dataset/{TOY}', data_type='test')
    focal_length = torch.tensor(focal_length)
    test_images, _, focal_length, poses = test_images.to('cuda'), _, focal_length.to('cuda'), poses.to('cuda')

    all_PSNRs = []
    all_SSIMs = []
    all_SSIMs2 = []

    print('test image size', len(test_images))
    
    for i in range(200):

        gt_test_img = test_images[i]
        rays_o, rays_d = get_rays(focal_length, poses[i])
        pts, z_vals = sample_points_on_rays(rays_o, rays_d, near=2, far=6, num_samples=32)
        
        flattened_pts = pts.reshape((-1, 3))
        
        positional_encodingd_points = positional_encoding(flattened_pts, num_encoding_functions=10)
        batches = get_minibatches(positional_encodingd_points, chunksize=4096)
        
        predictions = []
        for batch in batches:
            predictions.append(coarse_model(batch))
        radiance_field_flattened = torch.cat(predictions, dim=0)

        unflattened_shape = list(pts.shape[:-1]) + [4]
        radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)
        
        coarse_pred, _, _, coarse_weights = render(radiance_field, z_vals, rays_d)

        pts, z_vals, _ = sample_hierarchical(rays_o, rays_d, z_vals, coarse_weights, n_samples=64)

        flattened_pts = pts.reshape((-1, 3))
        
        positional_encodingd_points = positional_encoding(flattened_pts, num_encoding_functions=10)
        
        batches = get_minibatches(positional_encodingd_points, chunksize=4096)
        
        predictions = []
        for batch in batches:
            predictions.append(fine_model(batch))
        radiance_field_flattened = torch.cat(predictions, dim=0)


        unflattened_shape = list(pts.shape[:-1]) + [4]
        radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)
        
        fine_pred, _, _, _ = render(radiance_field, z_vals, rays_d)

        plt.imshow(fine_pred.detach().cpu().numpy())
        plt.savefig(os.path.join(img_save_path, str(i).zfill(6) + ".png"))

        fine_pred = fine_pred.detach().to(device='cpu', dtype=torch.float32).numpy()
        gt_test_img = gt_test_img.detach().to(device='cpu', dtype=torch.float32).numpy()
        PSNR_value = cv2.PSNR(gt_test_img, fine_pred)


        coarse_pred = coarse_pred.to('cpu')
        coarse_pred = coarse_pred.detach().numpy()
       

        coarse_loss = np.mean((gt_test_img - coarse_pred)**2)
        fine_loss = np.mean((gt_test_img - fine_pred)**2)
        loss = fine_loss + coarse_loss
        PSNR_value = -10.0 * np.log10(loss)

        predicted_test_gray = cv2.cvtColor(fine_pred, cv2.COLOR_BGR2GRAY)

        gt_test_gray_na = cv2.cvtColor(gt_test_img[:,:,:3], cv2.COLOR_BGR2GRAY)
        gt_test_gray = cv2.cvtColor(gt_test_img, cv2.COLOR_BGR2GRAY)

        SSIM_value, _ = ssim(gt_test_gray, predicted_test_gray, full=True, data_range=1.0)
        SSIM_value2, _ = ssim(gt_test_gray_na, predicted_test_gray, full=True, data_range=1.0)
        if SSIM_value != SSIM_value2:
            print("Values Differ")
            print(SSIM_value, SSIM_value2)
            exit()

        all_PSNRs.append(PSNR_value)
        all_SSIMs.append(SSIM_value)
        all_SSIMs2.append(SSIM_value2)

    print("all psnrs mean", np.mean(np.array(all_PSNRs)))
    print("all ssim mean", np.mean(np.array(all_SSIMs)))
    print("all ssim2 mean", np.mean(np.array(all_SSIMs2)))

def get_focal_length(cam_angle_x, img_size=(100, 100)):
    H, W = img_size
    return 0.5 * W / np.tan(0.5 * cam_angle_x)

def load_train_test_val_data(data_path=f"./Dataset/{TOY}", data_type='train',target_size=(100, 100)):
    
    json_file = open(f'{data_path}/transforms_{data_type}.json')
    json_data = json.load(json_file)

    frames = json_data['frames']

    camera_angle_x = float(json_data['camera_angle_x'])
    focal_length = get_focal_length(camera_angle_x)
    rotations = []
    poses = []

    for dict_ in frames:
        rotations.append(float(dict_['rotation']))
        poses.append(torch.tensor(dict_['transform_matrix'], dtype=torch.float32))
    poses = torch.stack(poses)
    path = f'{data_path}/{data_type}'
    img_list = os.listdir(path)
    img_list.sort(key=lambda x: int(x.rstrip('.png').split('_')[1]))

    # load images
    data_images = []
    if data_type == 'test':
        depth_images = []
        for i in range(0, len(img_list), 2):
            img_path = f'{path}/{img_list[i]}'
            depth_img_path = f'{path}/{img_list[i]}'
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = img.astype(float) / 255.0
            depth_img = cv2.imread(depth_img_path)
            depth_img = cv2.resize(depth_img, target_size)

            data_images.append(torch.from_numpy(img).to(torch.float32))
            depth_images.append(torch.from_numpy(depth_img).to(torch.float32))
        data_images = torch.stack(data_images)
        depth_images = torch.stack(depth_images)
        return data_images, depth_images, focal_length, poses

    else:
        for name in img_list:
            img_path = f'{path}/{name}'
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = img.astype(float) / 255.0
            data_images.append(torch.from_numpy(img).to(torch.float32))
        data_images = torch.stack(data_images)
        return data_images, focal_length, poses

def get_rays(focal_length, cam_pose, img_size=(100, 100)):

    cam_pose = cam_pose.to(focal_length)
    H, W = img_size
    i, j = torch.meshgrid(torch.arange(W, dtype=focal_length.dtype, device=focal_length.device).to(focal_length), torch.arange(H, dtype=focal_length.dtype, device=focal_length.device).to(focal_length))
    i = i.transpose(-1, -2)
    j = j.transpose(-1, -2)

    dirs = torch.stack([(i - 0.5 * W) / focal_length, -(j - 0.5 * H) / focal_length, -torch.ones_like(i)], dim=-1)
    
    dirs = dirs.unsqueeze(2)
    
    rays_d = torch.sum(dirs * cam_pose[:3, :3], dim=-1)

    rays_o = cam_pose[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d

def render(nerf_output, z_vals, rays_d, raw_noise_std=0.0):

    dist_between_samples = z_vals[..., 1:] - z_vals[..., :-1]
    dist_between_samples = torch.cat([dist_between_samples, 1e10 * torch.ones_like(dist_between_samples[..., :1])],
                                     dim=-1)

    dist_between_samples = dist_between_samples * torch.norm(rays_d[..., None, :], dim=-1)

    noise = 0
    if raw_noise_std > 0:
        noise = torch.randn(nerf_output[..., 3].shape) * raw_noise_std

    alpha = 1.0 - torch.exp(-nn.functional.relu(nerf_output[..., 3] + noise) * dist_between_samples)

    weights = alpha * cumprod_exclusive(1 - alpha + 1e-10)

    rgb = torch.sigmoid(nerf_output[..., :3])
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

    depth_map = torch.sum(weights * z_vals, dim=-1)

    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map, weights

## sample_heirarchical() and sample_pdf() have been referred from: 
## https://colab.research.google.com/drive/1TppdSsLz8uKoNwqJqDGg8se8BHQcvg_K?usp=sharing#scrollTo=kU4qRGMhNNHu 


def sample_hierarchical(rays_o, rays_d, z_vals, weights, n_samples, det=False):
    z_vals_centre = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_centre, weights[..., 1:-1], n_samples, det=det)
    new_z_samples = new_z_samples.detach()

    all_z_vals, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * all_z_vals[..., :, None]
    return pts, all_z_vals, new_z_samples

def sample_pdf(bins, weights, n_samples, det=False):

  pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True)

  cdf = torch.cumsum(pdf, dim=-1) 
  cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

  if not det:
    u = torch.linspace(0., 1., n_samples, device=cdf.device)
    u = u.expand(list(cdf.shape[:-1]) + [n_samples]) 
  else:
    u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)

  u = u.contiguous() 
  inds = torch.searchsorted(cdf, u, right=True) 

  below = torch.clamp(inds - 1, min=0)
  above = torch.clamp(inds, max=cdf.shape[-1] - 1)
  inds_g = torch.stack([below, above], dim=-1)

  matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
  cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                       index=inds_g)
  bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                        index=inds_g)

  denom = (cdf_g[..., 1] - cdf_g[..., 0])
  denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
  t = (u - cdf_g[..., 0]) / denom
  samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

  return samples


def sample_points_on_rays(
    ray_o,
    rays_d,
    near,
    far,
    num_samples,
    randomize=True,
):
    z_vals = torch.linspace(near, far, num_samples).to(ray_o)
    if randomize is True:
        noise_shape = list(ray_o.shape[:-1]) + [num_samples]
        z_vals = (
            z_vals
            + torch.rand(noise_shape).to(ray_o)
            * (far - near)
            / num_samples
        )
    pts = (
        ray_o[..., None, :]
        + rays_d[..., None, :] * z_vals[..., :, None]
    )
    return pts, z_vals

def run_model_once(
    focal_length,
    c2w,
    near,
    far,
    z_vals_per_ray,
    encoding_function,
    get_minibatches_function,
    chunksize,
    model,
    num_frequencies,
    model_type,
    weights=None,
    z_vals=None,
    n_samples_hierarchical=None
    
):

    rays_o, rays_d = get_rays(
        focal_length, c2w, img_size=(100, 100))

    if model_type != 'fine':
        pts, z_vals = sample_points_on_rays(
            rays_o, rays_d, near, far, z_vals_per_ray
        )

    else:
        pts, z_vals, _ = sample_hierarchical(
            rays_o, rays_d, z_vals, weights, n_samples_hierarchical)
    flattened_pts = pts.reshape((-1, 3))

    positional_encodingd_points = encoding_function(flattened_pts, num_frequencies)

    batches = get_minibatches_function(positional_encodingd_points, chunksize=chunksize)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0)

    radiance_field = torch.reshape(radiance_field_flattened, list(pts.shape[:-1]) + [4])

    rgb_predicted, _, _, weights = render(radiance_field, z_vals, rays_d)

    return rgb_predicted, weights, z_vals


class TinyNerf(torch.nn.Module):

    def __init__(self, filter_size=128, num_encoding_functions=10):
        super(TinyNerf, self).__init__()
        self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        self.relu = torch.nn.functional.relu
        

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        x = x.to(torch.float32)
        return x

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "log")
    os.makedirs(logdir, exist_ok=True)

    images, focal_length, c2w = load_train_test_val_data()
    focal_length = torch.tensor(np.array([focal_length])).to(device=device, dtype=torch.float32)
    print("F", type(focal_length), focal_length)

    height, width = images.shape[1:3]
    print("Loaded im size", height, width)

    near = 2.0
    far = 6.0

    testimg, testpose = images[99], c2w[99]
    testimg = testimg.to(device)
    
    images = images[:99, ..., :3].to(device)
    print("Imgs", type(images), images[0])

    num_encoding_functions = 10
    
    z_vals_per_ray = 32

    chunksize = 4096

    lr = 5e-3
    num_iters = 4100

    display_every = 100  

    coarse_model = TinyNerf(num_encoding_functions=num_encoding_functions)
    coarse_model.to(dtype=torch.float32, device=device)

    fine_model = TinyNerf(num_encoding_functions=num_encoding_functions)
    fine_model.to(dtype=torch.float32, device=device)

    model_params = list(coarse_model.parameters()) + list(fine_model.parameters())
    optimizer = torch.optim.Adam(model_params, lr=lr)

    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    psnrs = []
    iternums = []

    print('-------------------training begins==================')


    for i in trange(num_iters):

        
        coarse_model.train()
        fine_model.train()
        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx].to(device)
        target_c2w = c2w[target_img_idx].to(device)

        coarse_rgb_predicted, coarse_weights, coarse_depth_values = run_model_once(
            focal_length,
            target_c2w,
            near,
            far,
            z_vals_per_ray,
            positional_encoding,
            get_minibatches,
            chunksize,
            coarse_model,
            num_encoding_functions,
            model_type='coarse'
        )

        coarse_loss = torch.nn.functional.mse_loss(coarse_rgb_predicted, target_img)
        
        fine_rgb_predicted, _, _ = run_model_once(
            focal_length,
            target_c2w,
            near,
            far,
            z_vals_per_ray,
            positional_encoding,
            get_minibatches,
            chunksize,
            fine_model,
            num_encoding_functions,
            weights=coarse_weights,
            model_type='fine',
            z_vals=coarse_depth_values,
            n_samples_hierarchical=64
        )

        fine_loss = torch.nn.functional.mse_loss(fine_rgb_predicted, target_img)

        loss = coarse_loss + fine_loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % display_every == 0 or i == num_iters - 1:
            coarse_model.eval()
            fine_model.eval()
            coarse_rgb_predicted, coarse_weights, coarse_depth_values = run_model_once(
                focal_length,
                testpose,
                near,
                far,
                z_vals_per_ray,
                positional_encoding,
                get_minibatches,
                chunksize,
                coarse_model,
                num_encoding_functions,
                model_type='coarse'
            )

            fine_rgb_predicted, _, _ = run_model_once(
                focal_length,
                testpose,
                near,
                far,
                z_vals_per_ray,
                positional_encoding,
                get_minibatches,
                chunksize,
                fine_model,
                num_encoding_functions,
                weights=coarse_weights,
                model_type='fine',
                z_vals=coarse_depth_values,
                n_samples_hierarchical=64
            )

            coarse_loss = torch.nn.functional.mse_loss(coarse_rgb_predicted, testimg)
            fine_loss = torch.nn.functional.mse_loss(fine_rgb_predicted, testimg)
            loss = fine_loss + coarse_loss
            tqdm.write("Loss: " + str(loss.item()))
            psnr = -10.0 * torch.log10(loss)

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.imshow(coarse_rgb_predicted.detach().cpu().numpy())
            plt.savefig(os.path.join(logdir, str(i).zfill(6) + ".png"))
            plt.close("all")

            if i == num_iters - 1:
                plt.plot(iternums, psnrs)
                plt.savefig(os.path.join(logdir, "psnr.png"))
                plt.close("all")


    coarse_model_path = f'./saved_models/{TOY}/coarse_model.pth'
    fine_model_path = f'./saved_models/{TOY}/fine_model.pth'
    torch.save(coarse_model.state_dict(), coarse_model_path)

    torch.save(fine_model.state_dict(), fine_model_path)



if __name__ == "__main__":
    
    make_all_dirs()

    main()  # To train

    print('-------------------testing begins==================')
    test()  # To test