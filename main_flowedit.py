
import argparse
from email.mime import image
import os
import sys
import calendar
import time
import copy
import gc

import numpy as np
from PIL import Image

from utils.utils import image_grid, dataset_from_json, dataset_from_yaml

import torch
from torch import autocast, inference_mode

from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler

from inversion.ddim_inversion import ddim_inversion
from inversion.ddpm_inversion import inversion_forward_process_ddpm

from inversion.p2p_baselines import nmg_p2p, ef_or_pnp_inv_w_p2p, ef_wo_p2p
from inversion.p2p_h_edit import h_Edit_p2p_flowedit_w_guide

from p2p.ptp_classes import AttentionStore, load_512
from p2p.ptp_utils import register_attention_control
from p2p.ptp_controller_utils import make_controller, preprocessing
import types
import copy
import gc
import nltk
#import logger
from diffusers.utils import (
    #USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
)
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from einops import einsum
from torch.autograd.functional import jacobian

nltk.download('punkt')
nltk.download('punkt_tab')


def local_encoder_pullback_zt(
    self, sample, timesteps=None, context=None, y=None, 
    injected_features=None, op=None, block_idx=None,
    cross_attention_kwargs=None, added_cond_kwargs=None,
    pca_rank=50, chunk_size=25, min_iter=10, max_iter=1, convergence_threshold=1e-3,
):
    # get h samples
    time_s = time.time()

    # necessary variables
    
    h_shape = self.get_h(
        sample, timesteps, context, y,
        cross_attention_kwargs=cross_attention_kwargs,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False
    )[0].shape
    #print(h_shape)
    c_i, w_i, h_i = sample.size(1), sample.size(2), sample.size(3)
    c_o, w_o, h_o = h_shape[1], h_shape[2], h_shape[3]

    a = torch.tensor(0., device=sample.device, dtype=sample.dtype)

    # Algorithm 1
    vT = torch.randn(c_i*w_i*h_i, pca_rank, device=sample.device, dtype=torch.float)
    vT, _ = torch.linalg.qr(vT)
    v = vT.T
    v = v.view(-1, c_i, w_i, h_i)

    time_s = time.time()
    for i in range(max_iter):

        v = v.to(device=sample.device, dtype=sample.dtype)
        v_prev = v.detach().cpu().clone()
        
        u = []
        if v.size(0) // chunk_size != 0:
            v_buffer = list(v.chunk(v.size(0) // chunk_size))
        else:
            v_buffer = [v]

        import functorch
        for vi in v_buffer:
            #self = self.cpu() #to(sample.device, sample.dtype)
            g = lambda a : self.get_h(
                torch.add(sample, a * vi).clone(),#.cpu(), 
                timesteps,#.cpu(), 
                context.clone(), #.cpu(), 
                #y, 
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
                )[0].clone()

            ui = functorch.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
            #ui = jacobian(g, a)
            u.append(ui.detach().cpu().clone())
        u = torch.cat(u, dim=0)
        u = u.to(sample.device, sample.dtype).detach()
        self = self.to(device=sample.device, dtype=sample.dtype)
        
        g = lambda sample : einsum(
            u, self.get_h(
                sample, timesteps, context, y, cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )[0], 'b c w h, i c w h -> b'
        )
        v_ = torch.autograd.functional.jacobian(g, sample)
        v_ = v_.view(-1, c_i*w_i*h_i).detach()

        _, s, v = torch.linalg.svd(v_.float(), full_matrices=False)
        v = v.view(-1, c_i, w_i, h_i).detach().half()
        u = u.view(-1, c_o, w_o, h_o).detach().half()
        
        convergence = torch.dist(v_prev.float(), v.detach().cpu().float()).item()
        #print(f'power method : {i}-th step convergence : ', convergence)
        
        if torch.allclose(v_prev.float(), v.detach().cpu().float(), atol=convergence_threshold) and (i > min_iter):
            #print('reach convergence threshold : ', convergence)
            break
        
    u, s, vT = u.reshape(-1, c_o*w_o*h_o).T.detach(), s.sqrt().detach().half(), v.reshape(-1, c_i, w_i, h_i).detach()

    time_e = time.time()
    #print('power method runt ime ==', time_e - time_s)

    return u, s, vT

def get_h(
    self,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> Union[UNet2DConditionOutput, Tuple]:
    r"""
    The [`UNet2DConditionModel`] forward method.

    Args:
        sample (`torch.FloatTensor`):
            The noisy input tensor with the following shape `(batch, channel, height, width)`.
        timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
        encoder_hidden_states (`torch.FloatTensor`):
            The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
        encoder_attention_mask (`torch.Tensor`):
            A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
            `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
            which adds large negative values to the attention scores corresponding to "discard" tokens.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
            tuple.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].
        added_cond_kwargs: (`dict`, *optional*):
            A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
            are passed along to the UNet blocks.

    Returns:
        [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            If `return_dict` is True, an [`~models.unet_2d_condition.UNet2DConditionOutput`] is returned, otherwise
            a `tuple` is returned where the first element is the sample tensor.
    """
    # By default samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
        logger.info("Forward upsample size to force interpolation output size.")
        forward_upsample_size = True

    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])

    t_emb = self.time_proj(timesteps)

    # `Timesteps` does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=sample.dtype)

    emb = self.time_embedding(t_emb, timestep_cond)
    aug_emb = None

    if self.class_embedding is not None:
        if class_labels is None:
            raise ValueError("class_labels should be provided when num_class_embeds > 0")

        if self.config.class_embed_type == "timestep":
            class_labels = self.time_proj(class_labels)

            # `Timesteps` does not contain any weights and will always return f32 tensors
            # there might be better ways to encapsulate this.
            class_labels = class_labels.to(dtype=sample.dtype)

        class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    if self.config.addition_embed_type == "text":
        aug_emb = self.add_embedding(encoder_hidden_states)
    elif self.config.addition_embed_type == "text_image":
        # Kandinsky 2.1 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
            )

        image_embs = added_cond_kwargs.get("image_embeds")
        text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
        aug_emb = self.add_embedding(text_embs, image_embs)
    elif self.config.addition_embed_type == "text_time":
        if "text_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
            )
        text_embeds = added_cond_kwargs.get("text_embeds")
        if "time_ids" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
            )
        time_ids = added_cond_kwargs.get("time_ids")
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)
    elif self.config.addition_embed_type == "image":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
            )
        image_embs = added_cond_kwargs.get("image_embeds")
        aug_emb = self.add_embedding(image_embs)
    elif self.config.addition_embed_type == "image_hint":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
            )
        image_embs = added_cond_kwargs.get("image_embeds")
        hint = added_cond_kwargs.get("hint")
        aug_emb, hint = self.add_embedding(image_embs, hint)
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
    elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
        # Kadinsky 2.1 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
            )

        image_embeds = added_cond_kwargs.get("image_embeds")
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
    elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
            )
        image_embeds = added_cond_kwargs.get("image_embeds")
        encoder_hidden_states = self.encoder_hid_proj(image_embeds)
    # 2. pre-process
    sample = self.conv_in(sample)

    # 3. down
    down_block_res_samples = (sample,)
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        down_block_res_samples += res_samples

    if down_block_additional_residuals is not None:
        new_down_block_res_samples = ()

        for down_block_res_sample, down_block_additional_residual in zip(
            down_block_res_samples, down_block_additional_residuals
        ):
            down_block_res_sample = down_block_res_sample + down_block_additional_residual
            new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = new_down_block_res_samples

    # 4. mid
    if self.mid_block is not None:
        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            encoder_attention_mask=encoder_attention_mask,
        )

    if mid_block_additional_residual is not None:
        sample = sample + mid_block_additional_residual


    return UNet2DConditionOutput(sample=sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Device to run
    parser.add_argument("--device_num", type=int, default=0)
    
    # Data and output path
    parser.add_argument('--data_path', type=str, default="./assets/demo")
    parser.add_argument('--output_path', type=str, default="./results/flowedit_guide/")

    # Choose methods and editing categories
    parser.add_argument("--mode",  default="h_edit_R_p2p", help="modes: h_edit_R, ef, h_edit_D_p2p, nmg_p2p, pnp_inv_p2p, h_edit_R_p2p, ef_p2p")

    # Sampling and skipping steps
    parser.add_argument("--num_diffusion_steps", type=int, default=50) 
    parser.add_argument("--skip",  type=int, default=0) 

    # Random or Deterministic Sampling
    parser.add_argument("--eta", type=float, default=1.0) 

    # For guidance strength
    parser.add_argument("--cfg_src", type=float, default=1.0)
    parser.add_argument("--cfg_src_edit", type=float, default=5.0) #This is hat{w}^orig in our paper
    parser.add_argument("--cfg_tar", type=float, default=7.5)
    
    # Only for h-Edit
    parser.add_argument("--implicit", action='store_true', help="Use implicit form of h-Edit")
    parser.add_argument("--optimization_steps", type=int, default=1)
    parser.add_argument("--weight_reconstruction", type=float, default=0.1)

    # For P2P
    parser.add_argument("--xa", type=float, default=0.4) #cross attn control
    parser.add_argument("--sa", type=float, default=0.35) #self attn control: 0.6 for h-edit-D and 0.35 for h-edit-R

    args = parser.parse_args()


    print(f'Arguments: {args}')

    # 1. Declare some global vars
    data_path=args.data_path
    output_path=args.output_path

    full_data = dataset_from_yaml(data_path + "/demo.yaml")
    device = f"cuda:{args.device_num}"

    cfg_scale_src = args.cfg_src
    cfg_scale_src_edit = args.cfg_src_edit
    cfg_scale_tar_edit = args.cfg_tar
    eta = args.eta

    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    # 2. Load Diffusion Models - can be any off-the-shelf diffusion model, v-1.5, or your local models
    model_id = "CompVis/stable-diffusion-v1-4" # model_id = "stable_diff_local"

    xa_sa_string = f'_xa_{args.xa}_sa{args.sa}_' if args.mode in ['h_edit_D_p2p', 'nmg_p2p', 'pnp_inv_p2p', 'h_edit_R_p2p', 'ef_p2p'] else '_'
    weight_string = f'implicit_{args.implicit}_eta_{args.eta}_src_orig_{cfg_scale_src}_src_edit_{cfg_scale_src_edit}_tar_scale_{cfg_scale_tar_edit}_w_rec_{args.weight_reconstruction}_n_opts_{args.optimization_steps}_time_{time_stamp}'

    # 4. Load/Reload model:
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_id)
    ldm_stable.unet.get_h = types.MethodType(get_h, ldm_stable.unet) # replace get_h function
    ldm_stable.unet.local_encoder_pullback_zt = types.MethodType(local_encoder_pullback_zt, ldm_stable.unet) # replace get_h function
    
    # 5. Editing LOOP over all samples of the dataset

    for i in range(len(full_data)):
        current_image_data = full_data[i]

        # 5.1. Define DDIM or DDPM Inversion (deterministic or random)
        eta = args.eta
        is_ddim_inversion = True if eta == 0 else False 

        # 5.2. Clone a model to avoid attention masks tracking from previous samples
        ldm_stable_each_query = copy.deepcopy(ldm_stable).to(device)

        # 5.3 + 5.4. Define path to the image, editing_instruction, blended_word for P2P/ Load prompts
        image_path = "sampled_celebahq_500" + current_image_data['image']

        original_prompt = current_image_data.get('source_prompt', "") # default empty string
        editing_prompt = current_image_data.get('target_prompt', "")
        
        original_prompt = original_prompt.replace("[", "").replace("]", "")
        editing_prompt = editing_prompt.replace("[", "").replace("]", "")

        editing_instruction = current_image_data["editing_instruction"]

        blended_word = current_image_data["blended_word"].split(" ") if current_image_data["blended_word"] != "" else []

        # 5.5. Finalize the output path

        sub_string_save = args.mode + '_total_steps_' + str(args.num_diffusion_steps) + '_skip_' + str(args.skip) + '_'+ weight_string + xa_sa_string
        present_image_save_path=os.path.join(output_path, f'{i:04d}_orig_{original_prompt}_edit_{editing_prompt.replace(" ", "_")}_'+os.path.basename(image_path))
        reconstruct_path = os.path.join(output_path, f'{i:04d}_reconstruct_'+os.path.basename(image_path))
        
        if os.path.exists(present_image_save_path):
            print(f"Image {present_image_save_path} exists, skip!")
            #continue
        if not os.path.exists(os.path.dirname(present_image_save_path)):
            os.makedirs(os.path.dirname(present_image_save_path))

        # 5.6. Load Scheduler
        if eta == 0:
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        else:
            scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
        
        ldm_stable_each_query.scheduler = scheduler
        ldm_stable_each_query.scheduler.config.timestep_spacing = "leading"
        ldm_stable_each_query.scheduler.set_timesteps(args.num_diffusion_steps)

        # 5.7. Measure running time if you want!
        # torch.cuda.synchronize()  # Ensure all CUDA operations are done
        # start_time = time.time()

        # 5.8. Load the image
        offsets=(0,0,0,0)
        x0 = load_512(image_path, *offsets, device)

        # 5.9. Encode the original image to latent space using VAE

        with autocast("cuda"), inference_mode():
            w0 = (ldm_stable_each_query.vae.encode(x0).latent_dist.mode() * 0.18215).float()

        # 5.10. find Zs and wts - forward  (inversion) process

        if (eta == 0):
            wt, zs, wts = ddim_inversion(ldm_stable_each_query, w0, original_prompt, cfg_scale_src)
            eta = 1.0 #set eta = 1.0 to account for u_t^orig
        
        elif (eta > 0 and eta <= 1):
            wt, zs, wts, _ = inversion_forward_process_ddpm(ldm_stable_each_query, w0, etas=eta, prompt=original_prompt, cfg_scale_src=cfg_scale_src, num_inference_steps=args.num_diffusion_steps)
        
        else:
            print("Warning: out of range for eta")
            sys.exit(1)
                
        # 5.11. Prepare P2P arguments

        after_skip_steps = args.num_diffusion_steps-args.skip

        # 5.11.1 Check if number of words in encoder and decoder text are equal
        src_tar_len_eq_chosen = (len(original_prompt.split(" ")) == len(editing_prompt.split(" ")))

        # 5.11.2.  blend_word and importance weight eq_params
        prompts = [original_prompt, editing_prompt] 

        if args.mode[-3:] == 'p2p':
            assert len(prompts)>=2, "only for editing with prompts"

            #blend_word is provided in the dataset for P2P or use human knowledge
            #is_global_edit is tricky, require human knowledge
            
            #We provide a preprocessing function here to heuristically choose blend word and word imporantance to focus
            blend_word, eq_params_heuristic = preprocessing(original_prompt, editing_prompt, is_global_edit=True)
            blend_word= (((blended_word[0], ), (blended_word[1], ))) if len(blended_word) else None

            if (args.mode == 'h_edit_R_p2p' or args.mode == 'h_edit_D_p2p') and (args.optimization_steps > 1):
                eq_params={ "words": (blended_word[1], ), "values": (1.25, )} if len(blended_word) else None
            else:
                eq_params={ "words": (blended_word[1], ), "values": (2.0, )} if len(blended_word) else None

            if eq_params_heuristic is not None:
                if eq_params is not None:
                    eq_params_merged = {
                        'words': eq_params['words'] + eq_params_heuristic['words'],
                        'values': eq_params['values'] + eq_params_heuristic['values']
                    }
                else:
                    eq_params_merged = eq_params_heuristic
            else:
                eq_params_merged = eq_params
            
            controller = make_controller(prompts=prompts,  is_replace_controller = src_tar_len_eq_chosen,
                    cross_replace_steps=args.xa, self_replace_steps=args.sa,
                    blend_word=blend_word, equilizer_params=eq_params_merged,
                    num_steps=after_skip_steps, tokenizer=ldm_stable_each_query.tokenizer, device=ldm_stable_each_query.device)
                
        else:
            controller = AttentionStore()

        register_attention_control(ldm_stable_each_query, controller)

        # 5.12 Editing, available methods: h_edit_R, h_edit_D_p2p, h_edit_R_p2p

        cfg_scale_list = [cfg_scale_src, cfg_scale_src_edit, cfg_scale_tar_edit]     

        
        
        edited_w0, _ = h_Edit_p2p_flowedit_w_guide(ldm_stable_each_query, xT=wts[after_skip_steps], eta=eta, prompts=prompts, cfg_scales=cfg_scale_list,
                                                prog_bar = True, zs = zs[:(after_skip_steps)], controller = controller, weight_reconstruction = args.weight_reconstruction,
                                                optimization_steps=args.optimization_steps, after_skip_steps=after_skip_steps, is_ddim_inversion=is_ddim_inversion)
      
        # 5.13. Use VAE to decode image
        
        with autocast("cuda"), inference_mode():
            x0_dec = ldm_stable_each_query.vae.decode(1 / 0.18215 * edited_w0).sample

        if x0_dec.dim()<4:
            x0_dec = x0_dec[None,:,:,:]
        img = image_grid(x0_dec)
    
        img.save(present_image_save_path)

        ldm_stable_each_query.unet.zero_grad()
        del ldm_stable_each_query
        torch.cuda.empty_cache()
        gc.collect()

