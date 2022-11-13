import os
import torch
import pandas as pd
import torch.nn as nn
from PIL import Image
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from models import EncoderBlock, DecoderBlock
from torch.utils.data import random_split, DataLoader
from utils import WordTokenizer, MakeConfig, transform_factory


# :Model options
# :param d_model: embed size
# :param nheads: number of heads of the transformer model
# :param input_size: size of an image patch
# :param vocab_size: number of tokens / words
# :param max_len: maximum number of generated tokens
# :param embd_pdrop: dropout value of embedding layers
# :param mlp_drop: dropout value of transformer mlp layer
# :param num_layers: number of transformer blocks
# :param patch_len: number of patches in a single sequence
# :param learning_rate: learning rate value
# :param weight_decay: weight decay value

# :Training options
# :param model_type: type of the model
# :param num_epochs: number of epochs
# :param grad_clip_val: gradient clipping value
# :param grad_clip_algo: gradient clipping algorithm
# :param model_dir: directory where checkpoints are saved

# ;Data Module options. 
# :param root_dir: image directory
# :param annotations: csv file path
# :param freq_threshold: min frequency of a word in vocab
# :param batch_size: number of samples in a batch
# :param patch_size: dimension of an image patch
# :param num_workers: number of workers


class Collate(object):
    """ Adds padding to a batch of sentences with varying lengths. """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets


class Flicker8kDataset(Dataset):
    """ Pytorch Flicker8k Dataset class. """
    def __init__(self, 
        root_dir: str, 
        annotations: str, 
        patch_size: int, 
        freq_threshold: int):

        self.root_dir = root_dir
        self.df = pd.read_csv(annotations)
        self.imgs = self.df["image_file"]
        self.captions = self.df["captions"]

        # instantiate the tokenizer object.
        self.tokenizer = WordTokenizer()

        # build a vocabulary using the words in the corpus.
        self.tokenizer.build_vocabulary(freq_threshold, self.captions)

        # image transform object applied before training/eval.
        self.transform = transform_factory(mode='feed', patch_size=patch_size)
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]

        # transform image to patches.
        img_path = os.path.join(self.root_dir, img_id)
        with Image.open(img_path).convert('RGB') as image:
            img_patches = self.transform(image)

        # tokenize the caption using the tokenizer.
        encoded_caption = self.tokenizer(caption)
        return img_patches, torch.tensor(encoded_caption)


class FlickerDataModule(pl.LightningDataModule):
    """ Pytorch Lightning Data Module wraps around the Dataset class."""

    @staticmethod
    def get_default_config():
        C = MakeConfig()
        C.batch_size = 32
        C.train_eval_split = 0.8
        C.num_workers =  4
        C.patch_size = 16
        C.freq_threshold = 3
        # these options must be filled in externally
        C.root_dir = None
        C.annotations = None
        # optional entry
        C.data_split = None
        return C

    def __init__(self, confDict):
        super(FlickerDataModule, self).__init__()
        # config is dict of parameters
        # update default options with new values.
        # provide root_dir and annotations.
        config = self.get_default_config()
        config.merge_from_dict(confDict)
        assert config.root_dir is not None
        assert config.annotations is not None

        self.root_dir = config.root_dir
        self.annotations = config.annotations
        self.patch_size = config.patch_size
        self.freq_threshold = config.freq_threshold

        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.data_split = config.data_split
        self.train_eval_split = config.train_eval_split

    def setup(self, stage = None):

        # Pytorch Dataset object.
        self.dataset = Flicker8kDataset(
            root_dir=self.root_dir, 
            annotations=self.annotations, 
            patch_size=self.patch_size, 
            freq_threshold=self.freq_threshold)

        if self.data_split is not None:
            # reduces the overall size of the dataset.
            reduced_data_size = int(len(self.dataset)*self.data_split)
            rem_size = len(self.dataset) - reduced_data_size
            self.dataset, _ = random_split(
                self.dataset, [reduced_data_size, rem_size]
                )

        if stage == "fit" or stage is None:
            train_set_size = int(len(self.dataset)*self.train_eval_split)
            valid_set_size = len(self.dataset) - train_set_size
            self.train, self.validate = random_split(
                self.dataset, [train_set_size, valid_set_size]
            )

    def train_dataloader(self):
        return DataLoader(
            shuffle=True, 
            dataset=self.train, 
            batch_size=self.batch_size, 
            collate_fn=Collate(pad_idx=0),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            shuffle=False, 
            dataset=self.validate, 
            batch_size=self.batch_size,  
            collate_fn=Collate(pad_idx=0),
            num_workers=self.num_workers,
        )


class Img2TextModel(pl.LightningModule):
    """ Pytorch Lightning Module wraps around Pytorch Modules. """

    @staticmethod
    def get_default_config():
        C = MakeConfig()
        # num_layers must be given in the config
        C.input_size = 768
        C.nheads = 8
        C.d_model =  512
        C.patch_len =  256
        C.vocab_size = 4087
        C.max_len = 45
        # these options must be filled in externally
        C.num_layers = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.mlp_drop = 0.1
        # optimizer hyperparameters
        C.learning_rate = 1e-3
        C.weight_decay = 0.001
        return C

    def __init__(self, confDict):
        super(Img2TextModel, self).__init__()
        # config is dict of parameters
        # update default options with new values.
        # provide num_layers
        config = self.get_default_config()
        config.merge_from_dict(confDict)
        assert config.num_layers is not None

        # image encoder part
        self.encoder = nn.ModuleDict(dict(
            patch_proj = nn.Linear(config.input_size, config.d_model),
            patch_pos = nn.Embedding(config.patch_len, config.d_model),
            drop = nn.Dropout(config.embd_pdrop),
            eblocks = nn.ModuleList(
                [
                    EncoderBlock(config) for _ in range(config.num_layers)
                ]
            ),
        ))

        # decoder part
        self.decoder = nn.ModuleDict(dict(
            word_embed = nn.Embedding(config.vocab_size, config.d_model),
            word_pos = nn.Embedding(config.max_len, config.d_model),
            drop = nn.Dropout(config.embd_pdrop),
            dblocks = nn.ModuleList(
                [
                    DecoderBlock(config) for _ in range(config.num_layers)
                ]
            ),
            ln_f = nn.LayerNorm(config.d_model),
        ))

        # generates logits distribution over the entire vocabulary.
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.weight_decay = config.weight_decay
        self.learning_rate = config.learning_rate

        # model weight initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """ randomly initialize parameters """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def encoder_step(self, z: torch.Tensor):
        """ returns encoder output for a batch of data. """
        _, patch_len, _ = z.shape
        pos = torch.arange(
            0, patch_len, dtype=torch.long, device=self.device).unsqueeze(0)

        e = self.encoder
        z = e.drop(e.patch_pos(pos) + e.patch_proj(z))
        for block in e.eblocks:
            z = block(z)
        return z

    def decoder_step(self, enc_out: torch.Tensor, src_x: torch.Tensor):
        """ returns decoder logits for a batch of data. """
        _, src_cap_len = src_x.shape
        pos = torch.arange(
            0, src_cap_len, dtype=torch.long, device=self.device).unsqueeze(0)

        d = self.decoder
        # mask to aid the autoregressive property of the transformer decoder.
        mask = torch.tril(
            torch.ones(
                src_cap_len, src_cap_len, device=self.device
            )
        ).view(1, 1, src_cap_len, src_cap_len).detach()
        
        x = d.drop(d.word_pos(pos) + d.word_embed(src_x))
        for block in d.dblocks:
            x = block(enc_out, x, mask)
        x = d.ln_f(x)
        logits = self.unembed(x)
        return logits

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        """ processes single batch of data. """
        src_x, tgt_x = x[:,:-1], x[:,1:]
        enc_out = self.encoder_step(z)
        logits = self.decoder_step(enc_out, src_x)
        
        # loss calculation 
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt_x.reshape(-1), ignore_index=1)
        return loss, logits

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn 
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params))
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate)
        return optimizer


    @classmethod
    def from_pretrained(cls, model_path):
        """ loads model from existing model checkpoints."""
        # load checkpoint based on model type.
        config = cls.get_default_config()

        # eg. "checkpoints/img2text_tiny_256x256_16p.ckpt" 
        file_name = model_path.split("\\")[-1]
        assert file_name.endswith('ckpt'), 'Invalid file type.'

        # extract model type from file name.
        model_type = file_name.split('.')[0]
        
        # create config to pass during model loading
        if model_type == 'img2text_tiny_256x256_16p':
            config.num_layers = 1
        elif model_type == 'img2text_small_256x256_16p':
            config.num_layers = 3
        elif model_type == 'img2text_big_256x256_16p':
            config.num_layers = 6
        else:
            raise AssertionError('Invalid model type.')

        # model = Img2TextModel(config.to_dict())
        model = Img2TextModel.load_from_checkpoint(
            checkpoint_path=model_path, 
            confDict=config.to_dict())
        return model

    def training_step(self, batch, batch_idx):
        z, x = batch
        train_loss, _ = self(z, x)
        self.log(
            name = 'train_loss',
            value = train_loss,
            on_step = False,
            on_epoch = True,
            prog_bar=True,
            logger = True
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        z, x = batch
        val_loss, _ = self(z, x)
        self.log(
            name='val_loss',
            value=val_loss,
            on_step=False,
            on_epoch=True,
            logger=True
        )
        return  {'val_loss': val_loss,
                    'log': {'val_loss': val_loss}}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([o['val_loss'] for o in outputs], 0).mean()
        out = {'val_loss': loss}
        self.log(
            name='val_loss',
            value=loss,
            on_epoch=True,
            prog_bar=True,
            logger=True)
        return {**out, 'log': out}

    @torch.no_grad()
    def generate(self, z: torch.Tensor, max_tokens: int = 45, temperature: float = 1.0, do_sample: bool = False, top_k :int = None):
        """
        make sure to be in model.eval() mode of operation for this.
        """
        attn_scores = []
        word_count = 0

        def get_scores():
        # the hook signature
            def hook(model, input, output):
                attn_scores.append(output[1].detach())
            return hook

        idx_next = None
        enc_out = self.encoder_step(z)
        # append inital src_x with beginning of sentence tag.
        src_x = torch.tensor([1]).unsqueeze(0)

        # cross attention layer of final decoder block.
        # extracted attention scores are used for interpretation.
        inspect_layer = self.decoder.dblocks[-1].cross_attn
        handle = inspect_layer.register_forward_hook(get_scores())

        # loop until end of sentence tag is generated.
        while(idx_next != 2 and word_count < max_tokens): 
            logits = self.decoder_step(enc_out, src_x) 
            word_count = word_count + 1

            # logits of the current word to be generated.
            # Temperature is used to control the randomness of predictions by scaling 
            # the logits before applying softmax.When the temperature is 1, we compute 
            # the softmax directly on the logits, and using a temperature of 0.6 the 
            # model computes the softmax on logits / 0.6, resulting in a larger value. 
            # Performing softmax on larger values makes the model more confident (less 
            # input is needed to activate the output layer) but also more conservative in 
            # its samples (it is less likely to sample from unlikely candidates). Using a 
            # higher temperature produces a softer probability distribution over the 
            # classes, and makes the model more “easily excited” by samples, resulting 
            # in more iversity and also more mistakes.
            logits = logits[:, -1, :] / temperature

            if top_k is not None: 
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            # append sampled index to the running sequence and continue
            src_x = torch.cat((src_x, idx_next), dim=1)
            
        # remove the handlers
        handle.remove()
        x = src_x.squeeze().tolist()
        return x, attn_scores