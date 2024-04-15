import os
import cv2
import argparse

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PositionalSharding
from jax.sharding import PartitionSpec as P
import optax

import multiprocessing as mp
import numpy as np
import time

# import latentvideodiffusion as lvd
# import latentvideodiffusion.models.frame_vae as frame_vae
# import latentvideodiffusion.frame_extractor as fe
# import latentvideodiffusion.frame_transcode as ft

from . import utils, frame_extractor as fe, frame_transcode as ft
from .models import frame_vae 

# import utils, frame_extractor as fe
# from models import frame_vae

kl_a = 0.0001 #TODO fix loading of this value

#Gaussian VAE primitives
def gaussian_kl_divergence(p, q):
    p_mean, p_log_var = p
    q_mean, q_log_var = q

    kl_div = (q_log_var-p_log_var + (jnp.exp(p_log_var)+(p_mean-q_mean)**2)/jnp.exp(q_log_var)-1)/2
    return kl_div

def gaussian_log_probabilty(p, x):
    p_mean, p_log_var = p
    log_p = (-1/2)*((x-p_mean)**2/jnp.exp(p_log_var))-p_log_var/2-jnp.log(jnp.sqrt(2*jnp.pi))
    return log_p

def sample_gaussian(p, key):
    p_mean, p_log_var = p
    samples = jax.random.normal(key,shape=p_mean.shape)*jnp.exp(p_log_var/2)+p_mean
    return samples

def concat_probabilties(p_a, p_b):
    mean = jnp.concatenate([p_a[0],p_b[0]], axis=1)
    log_var = jnp.concatenate([p_a[1],p_b[1]], axis=1)
    return (mean, log_var)

@jax.jit
def vae_loss(vae, data, key):

    encoder, decoder = vae
    #Generate latent q distributions in z space
    q = jax.vmap(encoder)(data)

    #Sample Z values
    z = sample_gaussian(q, key)

    #Compute kl_loss terms
    z_prior = (0,0)
    kl = kl_a*gaussian_kl_divergence(q, z_prior)

    #Ground truth predictions
    p = jax.vmap(decoder)(z)

    #Compute the probablity of the data given the latent sample
    log_p = gaussian_log_probabilty(p, data)

    #Maximise p assigned to data, minimize KL div
    loss = sum(map(jnp.sum,[-log_p, kl]))/(data.size)

    return loss

def make_vae(n_latent, input_size, size_multipier, key):

    enc_key, dec_key = jax.random.split(key)

    e = frame_vae.VAEEncoder(n_latent, input_size, size_multipier, enc_key)
    d = frame_vae.VAEDecoder(n_latent, input_size, size_multipier, dec_key)
    
    vae = e,d
    return vae

def sample_vae(n_latent, n_samples, vae, key):
    z_key, x_key = jax.random.split(key)
    decoder = vae[1]
    p_z = (jnp.zeros((n_samples,n_latent)),)*2
    z = sample_gaussian(p_z, z_key)
    p_x = jax.vmap(decoder)(z)
    x = sample_gaussian(p_x, x_key)
    return x

def reconstruct_vae(n_samples, data_dir, vae, key):
    z_key, x_key = jax.random.split(key)
    encoder = vae[0]
    decoder = vae[1]
    print("Created Encoder and Decoder")
    extracted_frames = fe.extract_frames(data_dir, n_samples, x_key)
    print("Extracted Video Frames")

    encoded_frames = []
    data = jnp.array(extracted_frames, dtype=jnp.float32)
    mean, _ = jax.vmap(encoder)(data)
    encoded_frames.extend(mean)
    print("Encoded Frames")

    # z = sample_gaussian(encoded_frames, z_key)
    # p_x = jax.vmap(decoder)(z)
    # decoded_frames = sample_gaussian(p_x, x_key)
    decoded_frames = []
    for encoded_frame in encoded_frames:
        mean, _ = decoder(encoded_frame)
        frame = jax.lax.clamp(0., mean, 255.)
        decoded_frames.append(frame)
    print("Decoded Frames")
    return decoded_frames

def show_samples(samples):
    y = jax.lax.clamp(0., samples ,255.)
    frame = jnp.array(y.transpose(2,1,0),dtype=jnp.uint8)
    cv2.imshow('Random Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE model.')
    subparsers = parser.add_subparsers()
    
    #Training arguments
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('--checkpoint', type=int, default=None,
                        help='Checkpoint iteration to load state from.')
    
    #Sampling arguments
    sample_parser = subparsers.add_parser('sample')
    sample_parser.set_defaults(func=sample)
    sample_parser.add_argument('--checkpoint', type=int,
                        help='Checkpoint iteration to load state from.')
    
    sample_parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory')
    
    args = parser.parse_args()
    return args

def sample(args, cfg):
    n_samples = cfg["vae"]["sample"]["n_sample"]
    n_latent = cfg["lvm"]["n_latent"]

    state = utils.load_checkpoint(args.checkpoint)
    trained_vae = state[0]

    key = jax.random.PRNGKey(cfg["seed"])
    samples = sample_vae(n_latent, n_samples, trained_vae, key)
    utils.show_samples(samples, args.name)

def reconstruct(args, cfg):
    n_samples = cfg["vae"]["reconstruct"]["n_sample"]
    video_path = cfg["vae"]["reconstruct"]["video_file"]
    generation_path = cfg["vae"]["reconstruct"]["generation_path"]
    state = utils.load_checkpoint(args.checkpoint)
    trained_vae = state[0]

    key = jax.random.PRNGKey(cfg["seed"])
    samples = reconstruct_vae(n_samples, video_path, trained_vae, key)
    utils.show_samples(samples, generation_path ,args.name)

def data_loader(queue, frame_extractor):
    print('Producer: Running', flush=True)
    # generate work
    try:
        while True:
            # generate a value
            # _, key = jax.random.split(key)
            data = next(frame_extractor)
            # block
            # time.sleep(1)
            # add to the queue
            queue.put(data)
        # all done
        queue.put(None)
        print('Producer: Done', flush=True)
    except KeyboardInterrupt:
        print("Producer received KeyboardInterrupt, terminating...")

# consume work
def update_model(args, cfg, train_queue, val_queue):
    print('Consumer: Running', flush=True)
    print("Entered VAE Training Function")
    ckpt_dir = cfg["vae"]["train"]["ckpt_dir"]
    lr = cfg["vae"]["train"]["lr"]
    ckpt_interval = cfg["vae"]["train"]["ckpt_interval"]
    clip_norm = cfg["vae"]["train"]["clip_norm"]
    metrics_path = cfg["vae"]["train"]["metrics_path"]
    kl_a = cfg["vae"]["train"]["kl_alpha"]

    adam_optimizer = optax.adam(lr)
    optimizer = optax.chain(adam_optimizer, optax.zero_nans(), optax.clip_by_global_norm(clip_norm))
    
    ckpt_params = {
        "ckpt_type" : "vae",
        "ckpt_dir"  : cfg["vae"]["train"]["ckpt_dir"],
        "max_ckpts" : cfg["vae"]["train"]["max_ckpts"],
        "ckpt_interval" : cfg["vae"]["train"]["ckpt_interval"]
    }

    if args.checkpoint is None:
        key = jax.random.PRNGKey(cfg["seed"])
        init_key, state_key = jax.random.split(key)
        vae = make_vae(cfg["lvm"]["n_latent"], cfg["transcode"]["target_size"],cfg["vae"]["size_multiplier"], init_key)
        opt_state = optimizer.init(vae)
        i = 0
        state = vae, opt_state, state_key, i
        checkpoint_state = utils.create_checkpoint_state(**ckpt_params)
    else:
        checkpoint_path = args.checkpoint
        state, checkpoint_state = utils.load_checkpoint_state(checkpoint_path, **ckpt_params)
    print("Created VAE")

    dir_name = os.path.dirname(metrics_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    start_time = time.time()
    for _ in utils.tqdm_inf():
        with open(metrics_path, "a") as f:
            train_data = train_queue.get()
            val_data = val_queue.get()
            # check for stop
            if train_data is None or val_data is None:
                break

            try:
                train_data = jnp.array(train_data)
                val_data = jnp.array(val_data)
                train_loss, state = utils.update_state(state, train_data, optimizer, vae_loss)
                val_loss, _ = utils.update_state(state, val_data, optimizer, vae_loss)
                checkpoint_state = utils.update_checkpoint_state(state, checkpoint_state)

                current_time = time.time() - start_time
                f.write(f"{train_loss}\t{val_loss}\n")
                f.flush()
            except Exception as e:
                print(e)
  
def train(args, cfg):
    video_paths_train = cfg["vae"]["train"]["data_dir_train"]
    video_paths_val = cfg["vae"]["train"]["data_dir_val"]
    batch_size = cfg["vae"]["train"]["bs"]
    train_extractor = fe.FrameExtractor(video_paths_train, batch_size)
    val_extractor = fe.FrameExtractor(video_paths_val, batch_size)

    mp_ctx = mp.get_context('fork')
    try:
        train_queue = mp_ctx.Queue()
        val_queue = mp_ctx.Queue()

        # start the producer
        train_producer_process = mp_ctx.Process(target=data_loader, args=(train_queue, train_extractor))
        train_producer_process.start()
        val_producer_process = mp_ctx.Process(target=data_loader, args=(val_queue, val_extractor))
        val_producer_process.start() 
        # start the consumer
        consumer_process = mp_ctx.Process(target=update_model, args=(args, cfg, train_queue, val_queue))
        consumer_process.start()

        # wait for all processes to finish
        train_producer_process.join()
        val_producer_process.join()
        consumer_process.join()
        # Terminate processes if they didn't finish within the timeout
        if train_producer_process.is_alive():
            print("Train Producer process didn't finish in time, terminating...")
            train_producer_process.terminate()
        if val_producer_process.is_alive():
            print("Val Producer process didn't finish in time, terminating...")
            val_producer_process.terminate()     
        if consumer_process.is_alive():
            print("Consumer process didn't finish in time, terminating...")
            consumer_process.terminate()

        print("All tasks are done!")
    except KeyboardInterrupt:
        print("Main process received KeyboardInterrupt, terminating...")
