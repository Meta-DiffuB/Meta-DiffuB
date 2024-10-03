# from random import random

import torch
from torch import nn

from fairseq import utils
from fairseq.models import register_model, register_model_architecture, transformer
from fairseq.models.nat import NATransformerModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from improved_diffusion.gaussian_diffusion import GaussianDiffusion
from improved_diffusion.respace import SpacedDiffusion, space_timesteps

from .modules import DifformerEncoder, DifformerDecoder, SchedulerEncoder, SchedulerDecoder
from .utils import get_named_beta_schedule

import numpy as np
import math
import copy
import random


@register_model("difformer")
class Difformer(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)

        parser.add_argument(
            "--model-dim",
            type=int, metavar="N",
            help="The dimension of the model"
        )
        parser.add_argument(
            "--latent-dim",
            type=int, metavar="N",
            help="The dimension of $z_t$"
        )

        parser.add_argument(
            "--share-project-in-dim",
            action="store_true",
            help="Share projection layers of the encoder and decoder"
        )

        parser.add_argument(
            "--diffusion-steps",
            type=int, metavar="N", default=2000,
            help="Diffusion steps"
        )

        parser.add_argument(
            "--noise-schedule",
            type=str, metavar="STR", default="sqrt",
            help="The noise schedule during training"
        )
        parser.add_argument(
            "--rescaling-factor",
            type=float, metavar="D", default=1.0,
            help="The rescaling factor during training"
        )
        parser.add_argument(
            "--vp-rf",
            action="store_true",
            help="Use the variance-preserving rescaling factor"
        )

        parser.add_argument(
            "--embed-norm",
            action="store_true",
            help="Add embedding layer normalization"
        )
        parser.add_argument(
            "--embed-norm-affine",
            action="store_true",
            help="Add elementwise affine parameters to the embedding layer normalization"
        )
        parser.add_argument(
            "--embed-norm-before-proj",
            action="store_true",
            help="Put the embedding layer normalization before the projection layers"
        )

        parser.add_argument(
            "--self-cond",
            action="store_true",
            help="Self-conditioning"
        )
        parser.add_argument(
            "--self-cond-before-proj",
            action="store_true",
            help="Concatenate self-conditioning embeddings before the projection layers"
        )

        parser.add_argument(
            "--rounding-loss",
            action="store_true",
            help="Use the rounding loss instead of the anchor loss"
        )

        parser.add_argument(
            "--rescale-timesteps",
            action="store_true",
            help="Pass floating point timesteps into the model"
        )

    def __init__(self, args, encoder, decoder, scheduler_encoder, scheduler_decoder):
        super().__init__(args, encoder, decoder)

        # self.training_diffusion = GaussianDiffusion(
        #     betas=get_named_beta_schedule(
        #         args.noise_schedule,
        #         args.diffusion_steps,
        #         args.rescaling_factor if args.vp_rf else 1.0
        #     ),
        #     model_mean_type=None, model_var_type=None, loss_type=None
        # )

        # # so we have different schedules in training and decoding
        # self.decoding_diffusion = SpacedDiffusion(
        #     space_timesteps(args.diffusion_steps, str(args.decoding_steps)),
        #     betas=get_named_beta_schedule(
        #         args.decoding_noise_schedule if args.decoding_noise_schedule else args.noise_schedule,
        #         args.diffusion_steps,
        #         args.decoding_rescaling_factor if args.decoding_vp_rf else 1.0
        #     ),
        #     model_mean_type=None, model_var_type=None, loss_type=None
        # )

        self.timesteps_scale = (1000.0 / args.diffusion_steps) if args.rescale_timesteps else 1.0

        self.diffusion_steps = args.diffusion_steps
        self.scheduler_encoder = scheduler_encoder
        self.scheduler_decoder = scheduler_decoder
        self.standard_betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps, args.rescaling_factor if args.vp_rf else 1.0)

        self.up_map = {}
        self.down_map = {}
        
        
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, project_in_dim):
        return DifformerEncoder(args, src_dict, embed_tokens, project_in_dim)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, project_in_dim, project_out_dim):
        decoder = DifformerDecoder(args, tgt_dict, embed_tokens, project_in_dim, project_out_dim)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """ Build a new model instance """
        print('difformer build model')
        transformer.base_architecture(args)
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = transformer.DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = transformer.DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        latent_dim = args.latent_dim
        model_dim = args.model_dim

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, latent_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, latent_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, latent_dim, args.encoder_embed_path
            )

        # projection layers
        if latent_dim != model_dim:
            encoder_project_in_dim = nn.Linear(latent_dim, model_dim, bias=False)
            decoder_project_in_dim = (
                encoder_project_in_dim if args.share_project_in_dim
                else nn.Linear(latent_dim, model_dim, bias=False)
            )
            
            decoder_project_out_dim = nn.Linear(model_dim, latent_dim, bias=False)
        
        else:
            encoder_project_in_dim = nn.Identity()
            decoder_project_in_dim = nn.Identity()
            decoder_project_out_dim = nn.Identity()

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, encoder_project_in_dim)
        decoder = cls.build_decoder(
            args, tgt_dict, decoder_embed_tokens,
            decoder_project_in_dim, decoder_project_out_dim
        )

        scheduler_encoder = SchedulerEncoder(len(src_dict), 64, 64)
        scheduler_decoder = SchedulerDecoder(2, 64, 64)

        return cls(args, encoder, decoder, scheduler_encoder, scheduler_decoder)

    def calculate_my_betas(self, beta_ins, standard_betas):
    # beta_ins -> (batch, diffusion_steps), standard_betas -> (diffusion_steps, )
        betas = list()
        
        for i in range(len(beta_ins)):
            point = 0
            now = 0
            standard_betas_copy = standard_betas.copy()
            beta_temp = [standard_betas_copy[point]]
            point += 1
            now = point
            for instruction in beta_ins[i]:
                if instruction: 
                    beta_temp.append(standard_betas_copy[point])
                    now = point
                else: 
                    beta_temp.append(standard_betas_copy[now])
                point += 1
            betas.append(beta_temp)
        betas = torch.tensor(betas)
        return betas

    # BG-DiffuSeq
    def make_t_similar(self, simi_step, x_start, input_ids_mask, t, only_smaller = False):
        t_step = - torch.ones(t.shape, device = x_start.device)
        if not only_smaller: t_step[torch.randn_like(t.float()) > 0] = 1
        t_step *= simi_step
        t_similar = t + t_step
        t_similar = t_similar.clamp(0, self.num_timesteps - 1)
        tt_mask = (t_similar != 0) & (t_similar != self.num_timesteps - 1)
        t_similar = t_similar.long()
        return t_similar, tt_mask

    def make_t_similar_by_noise(self, x_start, input_ids_mask, t, only_smaller = False, use_random = False):
        direction = torch.randn_like(t.float())
        t_similar = copy.deepcopy(t)
        for idx, (i, j) in enumerate(zip(t.to('cpu'), direction.to('cpu'))):
            if j.item() > 0:
                if (not use_random) and i.item() in self.up_map:
                    t_similar[idx] = self.up_map[i.item()]
                elif i.item() in self.up_map and self.up_map[i.item()] != i.item():
                    t_similar[idx] = random.randint(i.item() + 1, self.up_map[i.item()])
            else:
                if (not use_random) and i.item() in self.down_map:
                    t_similar[idx] = self.down_map[i.item()]
                elif i.item() in self.down_map and self.down_map[i.item()] != i.item():
                    t_similar[idx] = random.randint(self.down_map[i.item()], i.item() - 1)

        tt_mask = (t_similar != t)
        return t_similar, tt_mask

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                        produces the cumulative product of (1-beta) up to that
                        part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def get_base_sqrt_one_minus_alphas_cumprod(self):
        betas = self.betas_for_alpha_bar(
            self.diffusion_steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        return sqrt_one_minus_alphas_cumprod.tolist()

    def fill_noise_map(self, simi_noise):
        a = self.get_base_sqrt_one_minus_alphas_cumprod()
        lp = 0
        rp = 0
        while rp < len(a):
            if a[rp] - a[lp] <= simi_noise:
                rp += 1
            else:
                self.up_map[lp] = rp - 1
                # down_map[rp - 1] = lp
                lp += 1

        rp = len(a) - 1
        lp = len(a) - 1
        while lp >= 0:
            if a[rp] - a[lp] <= simi_noise:
                lp -= 1
            else:
                self.down_map[rp] = lp + 1
                rp -= 1
        # print("simi_noise",simi_noise)
        # print("self.down_map", self.down_map)
        # print("self.up_map", self.up_map)
    
    def _x0_helper(self, model_output, x, t):

        if self.predict_xstart:
            pred_xstart = model_output
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else: # predict eps
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        return {'pred_xprev':pred_prev, 'pred_xstart':pred_xstart}

    def forward(self, src_tokens, src_lengths, _, tgt_tokens, **kwargs):
        """ Compute training losses """
        
        print("#####forward#####")

        # modify START
        # add scheduler model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        word2idx = {
            'F': 0,
            'T': 1,
            'BOS': 2,
        }
        weight = 0
        TF_weight = torch.tensor([(1+weight), (1-weight)]).to(device)

        self.scheduler_encoder.eval()
        self.scheduler_decoder.eval()
        with torch.no_grad():
            scheduler_encoder_output, scheduler_encoder_h, scheduler_encoder_c = self.scheduler_encoder(src_tokens)
            scheduler_decoder_input = torch.tensor([[word2idx['BOS']]*(self.diffusion_steps-1) for _ in range(src_tokens.shape[0])])
            scheduler_decoder_input = scheduler_decoder_input.to(device)
            beta_instruction = torch.tensor([[word2idx['F']]*(self.diffusion_steps-1) for _ in range(scheduler_decoder_input.shape[0])])
            for _step in range(self.diffusion_steps-1):
                scheduler_decoder_output = self.scheduler_decoder(scheduler_decoder_input, scheduler_encoder_h, scheduler_encoder_c)
                scheduler_decoder_output = scheduler_decoder_output[:,_step,]
                scheduler_decoder_output = scheduler_decoder_output * TF_weight
                scheduler_decoder_output_TF = torch.multinomial(scheduler_decoder_output, 1)
                scheduler_decoder_output_TF = scheduler_decoder_output_TF[:,0]

                beta_instruction[:, _step] = scheduler_decoder_output_TF
                if _step+1 < self.diffusion_steps-1:
                    scheduler_decoder_input[:,_step+1] = scheduler_decoder_output_TF
                del scheduler_decoder_output, scheduler_decoder_output_TF
                # th.cuda.empty_cache()
            del scheduler_encoder_output, scheduler_encoder_h, scheduler_encoder_c, scheduler_decoder_input
        betas = self.calculate_my_betas(beta_instruction, self.standard_betas)

        # update new noise diffusion
        self.training_diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=None, model_var_type=None, loss_type=None
        )

        # modify END

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
        mask = tgt_tokens.ne(self.pad)

        # diffusion
        z_0 = self.decoder.forward_embedding(tgt_tokens)
        t = torch.randint(0, self.args.diffusion_steps, [len(z_0)], device=z_0.device)
        model_t = t * self.timesteps_scale

        noise = torch.randn_like(z_0) * self.args.rescaling_factor
        z_t = self.training_diffusion.q_sample(z_0, t, noise).type_as(z_0)

        # self-conditioning
        prev_z_0_hat = torch.zeros_like(z_0)
        if self.args.self_cond and random.random() < 0.5:
            with torch.no_grad():
                prev_z_0_hat = self.decoder(z_t, model_t, mask, encoder_out, prev_z_0_hat)[0]
        
        z_0_hat = self.decoder(z_t, model_t, mask, encoder_out, prev_z_0_hat)[0]
        logits = self.decoder.output_layer(z_0 if self.args.rounding_loss else z_0_hat)

        # BG-DiffuSeq
        simi_penalty = "l2_noise_random"
        simi_lambda = -2.0
        simi_step = 10
        simi_noise = 0.05
        
        use_random = False
        if 'random' in simi_penalty: use_random = True

        if simi_penalty.startswith("l2_noise"):
            if len(self.up_map) == 0: self.fill_noise_map(simi_noise)
            t_similar, tt_mask = self.make_t_similar_by_noise(tgt_tokens, mask, t, only_smaller = False, use_random = use_random)
        else:
            t_similar, tt_mask = self.make_t_similar(simi_step, tgt_tokens, mask, t, only_smaller = False)
        # t_similar = t_similar 
        model_t_similar = t_similar * self.timesteps_scale

        noise2 = torch.randn_like(z_0) * self.args.rescaling_factor
        z_t_similar = self.training_diffusion.q_sample(z_0, t_similar, noise2).type_as(z_0)
       
        # self-conditioning
        prev_z_0_hat_similar = torch.zeros_like(z_0)
        if self.args.self_cond and random.random() < 0.5:
            with torch.no_grad():
                prev_z_0_hat_similar = self.decoder(z_t_similar, model_t_similar, mask, encoder_out, prev_z_0_hat_similar)[0]

        if simi_penalty:
            # model_output_from_t_similar = model(z_t_similar, t_similar, **model_kwargs)
            z_0_hat_similar = self.decoder(z_t_similar, model_t_similar, mask, encoder_out, prev_z_0_hat)[0]
            # terms["mse_similar"] = mean_flat((target - model_output_from_t_similar) ** 2)
            mse_similar = (z_0 - z_0_hat_similar)[mask].square().mean()
            
            # model_out_x_start_similar = self._x0_helper(model_output_from_t_similar, x_t_similar, t_similar)['pred_xstart'] 

            # difformer 沒有rounding loss
            # t0_mask_similar = (t_similar == 0)
            # t0_loss_similar = mean_flat((x_start_mean - model_out_x_start_similar) ** 2) # ||emb(w) - f_theta(x_1, 1)||^2

            # terms["mse_similar"] = th.where(t0_mask_similar, t0_loss_similar, terms["mse_similar"])
            # terms["mse"] += terms["mse_similar"]

            # tT_loss *= 2
            # decoder_nll *= 2

        # if simi_penalty == "cosine": # no in this if
        #     if simi_lambda > 0:
        #         terms["cos_penalty"] = mean_flat(1 + th.nn.functional.cosine_similarity(model_output_from_t_similar, model_output, dim = -1)) * simi_lambda
        #     else:
        #         terms["cos_penalty"] = - mean_flat(1 - th.nn.functional.cosine_similarity(model_output_from_t_similar, model_output, dim = -1)) * simi_lambda

        if simi_penalty.startswith('l2'):
            if simi_lambda > 0:
                raise NotImplementedError('simi penalty nedd < 0 while l2 penalty is used')
            else:
                # terms["l2"] = - mean_flat((model_output_from_t_similar -  model_output) ** 2) * simi_lambda
                l2_loss = - (z_0_hat_similar - z_0_hat)[mask].square().mean() * simi_lambda
 
        return {
            "diffusion": {
                "loss": (z_0_hat - z_0)[mask].square().mean() + (l2_loss * tt_mask).mean()
            },

            "word_ins": {
                "out": logits,
                "tgt": tgt_tokens,
                "mask": mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },

            "scheduler": {
                "instruction": beta_instruction,
            }
        }

    def forward_scheduler(self, src_tokens, beta_instruction, **kwargs):
        print("[Scheduler Forward]")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        word2idx = {
            'F': 0,
            'T': 1,
            'BOS': 2,
        }

        self.scheduler_encoder.train()
        self.scheduler_decoder.train()
        scheduler_encoder_output, scheduler_encoder_h, scheduler_encoder_c = self.scheduler_encoder(src_tokens)
        scheduler_decoder_input = torch.cat((torch.tensor([[word2idx['BOS']]]*len(beta_instruction)), beta_instruction), dim=1)
        scheduler_decoder_input = scheduler_decoder_input.to(device)
        scheduler_decoder_output = self.scheduler_decoder(scheduler_decoder_input, scheduler_encoder_h, scheduler_encoder_c)
        scheduler_decoder_output = torch.log(scheduler_decoder_output)

        return scheduler_decoder_output


    def forward_scheduler_decode(self, src_tokens):
        print('[scheduler decode]')
        # add scheduler model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        word2idx = {
            'F': 0,
            'T': 1,
            'BOS': 2,
        }
        weight = 0
        TF_weight = torch.tensor([(1+weight), (1-weight)]).to(device)

        self.scheduler_encoder.eval()
        self.scheduler_decoder.eval()
        with torch.no_grad():
            scheduler_encoder_output, scheduler_encoder_h, scheduler_encoder_c = self.scheduler_encoder(src_tokens)
            scheduler_decoder_input = torch.tensor([[word2idx['BOS']]*(self.diffusion_steps-1) for _ in range(src_tokens.shape[0])])
            scheduler_decoder_input = scheduler_decoder_input.to(device)
            beta_instruction = torch.tensor([[word2idx['F']]*(self.diffusion_steps-1) for _ in range(scheduler_decoder_input.shape[0])])
            for _step in range(self.diffusion_steps-1):
                scheduler_decoder_output = self.scheduler_decoder(scheduler_decoder_input, scheduler_encoder_h, scheduler_encoder_c)
                scheduler_decoder_output = scheduler_decoder_output[:,_step,]
                scheduler_decoder_output = scheduler_decoder_output * TF_weight
                scheduler_decoder_output_TF = torch.multinomial(scheduler_decoder_output, 1)
                scheduler_decoder_output_TF = scheduler_decoder_output_TF[:,0]

                beta_instruction[:, _step] = scheduler_decoder_output_TF
                if _step+1 < self.diffusion_steps-1:
                    scheduler_decoder_input[:,_step+1] = scheduler_decoder_output_TF
                del scheduler_decoder_output, scheduler_decoder_output_TF
                # th.cuda.empty_cache()
            del scheduler_encoder_output, scheduler_encoder_h, scheduler_encoder_c, scheduler_decoder_input
        betas = self.calculate_my_betas(beta_instruction, self.standard_betas)
        return betas

    def forward_decoder(self, z_t, step, mask, encoder_out, betas, prev_z_0_hat=None, **kwargs):
        """ Sample z_{t-1} given z_t """
        print('###forward_decoder###')
        # modify START
        
        # update new noise diffusion
        # so we have different schedules in training and decoding
        self.decoding_diffusion = SpacedDiffusion(
            space_timesteps(self.args.diffusion_steps, str(self.args.decoding_steps)),
            betas=betas,
            model_mean_type=None, model_var_type=None, loss_type=None
        )
        # modify END

        # rescale timesteps
        model_t = (
            self.decoding_diffusion.timestep_map[step]
            if self.args.decoding_fixed_t is None
            else self.args.decoding_fixed_t * self.args.diffusion_steps
        ) * self.timesteps_scale
        model_t = torch.full([len(z_t)], model_t, device=z_t.device)

        # predict z_0            
        z_0_hat = self.decoder(z_t, model_t, mask, encoder_out, prev_z_0_hat)[0]
        # assert True == False, f'{self.decoding_diffusion.posterior_mean_coef1.shape}, {betas.shape}, {model_t.shape}, {z_0_hat.shape}'

        # clamping trick
        if self.args.clamping:
            tokens = self.decoder.output_layer(z_0_hat).argmax(-1)
            z_0_hat = self.decoder.forward_embedding(tokens)

        # sample z_{t-1}
        t = torch.tensor([step]*z_t.shape[0], device=z_t.device)
        mean, _, log_variance = self.decoding_diffusion.q_posterior_mean_variance(z_0_hat, z_t, t)
        noise = torch.randn_like(z_t) * self.args.decoding_rescaling_factor

        z_t = mean + (0.5 * log_variance).exp() * noise
        z_t = z_t.type_as(z_0_hat)

        return z_t, z_0_hat

    def forward_output_layer(self, z_t, mask):
        scores, tokens = self.decoder.output_layer(z_t).log_softmax(-1).max(-1)
        return tokens, scores, mask

    def initialize_z_t(self, encoder_out):
        """ Sample z_T """
        # length prediction
        pred_length = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )

        max_length = pred_length.clamp_(min=2).max()
        z_t = torch.randn(
            (len(pred_length), max_length, self.args.latent_dim),
        ) * self.args.decoding_rescaling_factor

        return z_t, pred_length

    def regenerate_beam(self, pred_length, length_beam_size, noise_beam_size):
        pred_length = (
            pred_length[:, None, None]
            + utils.new_arange(pred_length, 1, noise_beam_size, length_beam_size).transpose(-1, -2)
            - length_beam_size // 2
        ).flatten()  # (bsz * length_beam_size * noise_beam_size)

        max_length = pred_length.clamp_(min=2).max()
        z_t = torch.randn(
            (len(pred_length), max_length, self.args.latent_dim),
        ) * self.args.decoding_rescaling_factor

        return z_t, pred_length


@register_model_architecture("difformer", "difformer")
def base_architecture(args):
    args.model_dim = getattr(args, "model_dim", 512)
    args.latent_dim = getattr(args, "latent_dim", 128)

    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = args.model_dim
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = args.model_dim
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )

    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.latent_dim)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_input_dim)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.share_project_in_dim = getattr(args, "share_project_in_dim", False)

    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)

    args.diffusion_steps = getattr(args, "diffusion_steps", 2000)

    args.noise_schedule = getattr(args, "noise_schedule", "linear")
    args.rescaling_factor = getattr(args, "rescaling_factor", 1.0)
    args.vp_rf = getattr(args, "vp_rf", False)

    args.embed_norm = getattr(args, "embed_norm", False)
    args.embed_norm_affine = getattr(args, "embed_norm_affine", False)
    args.embed_norm_before_proj = getattr(args, "embed_norm_before_proj", False)

    args.self_cond = getattr(args, "self_cond", False)
    args.self_cond_before_proj = getattr(args, "self_cond_before_proj", False)

    args.rounding_loss = getattr(args, "rounding_loss", False)

    args.rescale_timesteps = getattr(args, "rescale_timesteps", False)


@register_model_architecture("difformer", "difformer_base")
def difformer_base(args):
    args.model_dim = getattr(args, "model_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)

    base_architecture(args)


@register_model_architecture("difformer", "difformer_iwslt_de_en")
def difformer_iwslt_de_en(args):
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

    base_architecture(args)


@register_model_architecture("transformer", "transformer_base")
def transformer_base(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    transformer.base_architecture(args)
