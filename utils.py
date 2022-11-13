import torch
import pickle
import numpy as np
import torch.optim
from pathlib import Path
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import torch.nn.functional as F
from spacy.lang.en import English
from torchvision import transforms as T


class ExtractPatches(object):
    """ extracts non overlapping patches of an input image."""
    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        z = z.unsqueeze(0)
        patches = F.unfold(z, kernel_size=self.patch_size, stride=self.patch_size, padding=0)
        patches = torch.permute(patches.view(patches.size(1), patches.size(2)), dims=(1, 0))
        return patches


class MakeConfig:
    """ configuration class from Andrej Karpathy's minGPT """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, MakeConfig):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, MakeConfig) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)


def transform_factory(mode: str, patch_size: int = None):
    """ 
    transforms PIL image based on the activity.
    param mode: str object to be specified
    'feed' mode for training or eval activity. 
    'grid_vis' mode for grid visualization. 
    'adv_vis' mode for attention score interpretation. 
    param patch_size: if mode is 'feed' or 'adv_vis'.
    """
    if mode == 'feed':
    # for training or eval phase.
        return T.Compose([
                    T.Resize(256),
                    T.CenterCrop(256),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                    ExtractPatches(patch_size)
                ])
    elif mode == 'grid_vis':
        # for image visualization.
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
        ])
    elif mode == 'adv_vis':
        # for image visualization.
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
            ExtractPatches(patch_size)
        ])
    else:
        raise AssertionError('Invalid mode choice.')


def grid_visualization(image):
    """ draws 16*16 grids over the input image. """
    img = transform_factory(mode='grid_vis')(image)
    # print(img.size)
    img = np.asarray(np.copy(img))
    # print(img.shape)
    dx, dy = 16,16
    # Custom (rgb) grid color
    grid_color = [0,0,0]
    # Modify the image to include the grid
    img[:,::dy,:] = grid_color
    img[::dx,:,:] = grid_color
    # Show the result
    plt.imshow(img)


# def score_to_contrast(score):
#     """ returns contrast value based on the attention score. """
#     assert score >= 0 and score <= 1, 'invalid score value.'
#     if score < 0.003:
#         contrast_val = math.cos(0.20*math.pi*(score))
#     else:
#         contrast_val = math.cos(0.50*math.pi*(score))
#     return contrast_val

def score_to_contrast(score):
    """ test function """
    assert score >= 0 and score <= 1, 'invalid score value.'
    if score < 0.003:
        contrast_val = 1
    else:
        contrast_val = 0.2
    return contrast_val


def change_contrast(image_patch, patch_score):
    """ changes the contrast value of the input image. """
    enhancer = ImageEnhance.Contrast(image_patch)
    image = enhancer.enhance(score_to_contrast(patch_score))
    return image


def show_attn_seeking_patches(patches: torch.Tensor, attn_scores: list):
    """ 
    based on the attention scores per head, patches opf 16*16 are highlighted 
    to show their importance. For n heads, the function needs to be called n times. 
    Returns a tensor-sequence (256, 768) of flattened image patches.
    param patches: 256 patches of an image input.
    param attn_scores: list of scores assigned to the patches.
    """
    patch_list = []
    for i in range(patches.size(0)):
        # retreive the i'th patch.
        patch = patches[i,:]
        # reshape the patch into C H W form.
        img_patch = torch.reshape(patch, shape=(3,16,16))
        # convert the patch tensor into a PIL image.
        img_patch = T.ToPILImage()(img_patch)
        # retreive the score corresponding to the image patch.
        patch_score = attn_scores[i]
        # apply highlighting function.
        img_patch = change_contrast(img_patch, patch_score)
        img_patch = T.ToTensor()(img_patch)
        patch = torch.reshape(img_patch, shape=(1,768))
        patch_list.append(patch)
    # return the highlighted patches in orginal form.
    patches = torch.stack(patch_list, dim=0)
    return patches


def get_attn_scores(attn_scores: torch.Tensor, word_num: int, head_num: int):
    """ given the word number and the head number, returns list of attention scores. """
    word_index, head_index = word_num - 1, head_num - 1
    scores = attn_scores[word_index][:,head_index,word_index,:].squeeze().tolist()
    return scores


def attn_map_for_word(image, scores: list):
    """ visualizes highlighted patches of an image based on its attn score. """
    patches = transform_factory(mode='adv_vis', patch_size=16)(image)
    patches = patches.squeeze(0)
    highlighted_patches = show_attn_seeking_patches(patches, scores)
    highlighted_patches = highlighted_patches.squeeze(1)
    highlighted_patches = torch.permute(highlighted_patches[None,:,:], (0,2,1))
    highlighted_image = F.fold(highlighted_patches, output_size=256, kernel_size=16, dilation=1, padding=0, stride=16)
    highlighted_image = T.ToPILImage()(highlighted_image.squeeze(0))
    grid_visualization(highlighted_image)


class WordTokenizer: 
    """ English Language Tokenizer """
    def __init__(self):     
        self.itos = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.lang = English()

    def __len__(self):
        return len(self.itos)

    def pre_tokenizer(self, text: str) -> list:
        return [tok.text.lower() for tok in self.lang.tokenizer(text)]

    @staticmethod
    def from_saved():
        """ need not build vocab again during inference. """
        stoi = Path("./data/stoi.pickle")
        itos = Path("./data/itos.pickle")
        if stoi.is_file() and itos.is_file():
            tokenizer = WordTokenizer()
            with open('./data/stoi.pickle', 'rb') as e:
                tokenizer.stoi = pickle.load(e)
            with open('./data/itos.pickle' , 'rb') as f: 
                tokenizer.itos = pickle.load(f)
            return tokenizer
        else:
            raise AssertionError('vocab files do not exist.')

    def build_vocabulary(self, freq_threshold: int, sentence_list: list[str]):
        """ called once before training. """
        frequencies = {}
        id = 4
        for sentence in sentence_list:
            for word in self.pre_tokenizer(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == freq_threshold:
                    self.stoi[word] = id
                    self.itos[id] = word
                    id = id + 1

        # save the stoi and itos dictionaries for future use.
        with open('./data/stoi.pickle' , 'wb') as e: 
            pickle.dump(self.stoi, e, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./data/itos.pickle' , 'wb') as f: 
            pickle.dump(self.itos, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __call__(self, text: str) -> list:
        if len(self.stoi) == 4:
            raise AssertionError('build vocabulary before tokenizing text.')
        else:
            caption = self.pre_tokenizer(text)
            encoded_caption = [self.stoi["<BOS>"]]
            encoded_caption.extend([
                self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                for token in caption
            ])
            encoded_caption.append(self.stoi["<EOS>"])
            return encoded_caption

    def ids_to_sentence(self, ids: list) -> str:
        """ converts list of ids into a sentence. """
        if len(self.itos) == 4:
            raise AssertionError('build vocabulary before tokenizing text.')
        else:
            sentence = ''
            first = True
            for id in ids:
                # avoid including the BOS and EOS tokens.
                if id == 1 or id == 2:
                    continue
                
                # append the word generated.
                word = self.itos[id]
                if first:
                    sentence = word
                    first = False
                else:
                    sentence = ' '.join([sentence, word])
            # return the complete sentence.      
            return sentence