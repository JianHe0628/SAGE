import torch
from sage.utils import utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
import cv2
import random
import numpy as np
import pickle
from pathlib import Path
from vidaug import augmentors as va
from sage.utils.augmentation import *

# global definition
from sage.utils.definition import *

def decompress_from_lzma(filename):
    with lzma.open(filename, "rb") as f:
        return pickle.load(f)  # Decompress and deserialize

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        if isinstance(Image, PIL.Image.Image):
            Image = np.asarray(Image, dtype=np.uint8)
        new_video_x = (Image - 127.5) / 128
        return new_video_x

class SomeOf(object):
    """
    Selects one augmentation from a list.
    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
    """

    def __init__(self, transforms1, transforms2):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, clip):
        select = random.choice([0, 1, 2])
        if select == 0:
            return clip
        elif select == 1:
            if random.random() > 0.5:
                return self.transforms1(clip)
            else:
                return self.transforms2(clip)
        else:
            clip = self.transforms1(clip)
            clip = self.transforms2(clip)
            return clip

class SignDataset(Dataset.Dataset):
    def __init__(self, path, tokenizer, config, args, phase, dataset_name=None):
        self.config = config
        self.args = args
        self.dataset_name = dataset_name
        self.phase = phase if phase in ['train', 'test'] else 'dev'
        with open(path, 'rb') as f:
            full_raw = pickle.load(f)
        self.full_raw = full_raw
        self.phase_raw_data = full_raw[self.phase]
        self.raw_data_name = list(self.phase_raw_data.keys())

        for key in list(self.phase_raw_data.keys()):
            length_of_segment = len(self.phase_raw_data[key]['Frame_Range'])
            if length_of_segment == 0:
                self.raw_data_name.remove(key)
            if length_of_segment > self.args.max_segment_length:
                self.raw_data_name.remove(key)


        
        self.img_root = Path(config['data']['img_path'])

        with open(config['data']['metadata_path'], "rb") as f:
            metadata = pickle.load(f)
        self.trans_to_pseudo = metadata['dict_sentence']

        with open(config['data']['sentence_embedding_path'], 'rb') as f:
            sent_embed = pickle.load(f)
        self.phase_sent_embed = sent_embed[self.phase]
        
        self.tokenizer = tokenizer

        sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
        self.seq = va.Sequential([
            # va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
            # va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
            sometimes(va.RandomRotate(30)),
            # sometimes(va.RandomResize(0.2)),
            # va.RandomCrop(size=(256, 256)),
            sometimes(va.RandomTranslate(x=10, y=10)),

            # sometimes(Brightness(min=0.1, max=1.5)),
            # sometimes(Contrast(min=0.1, max=2.0)),

        ])
        self.seq_color = va.Sequential([
            sometimes(Brightness(min=0.1, max=1.5)),
            sometimes(Color(min=0.1, max=1.5)),
            # sometimes(Contrast(min=0.1, max=2.0)),
            # sometimes(Sharpness(min=0.1, max=2.))
        ])
        # self.seq = SomeOf(self.seq_geo, self.seq_color)
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.raw_data_name)
    
    def __getitem__(self, index):
        key = self.raw_data_name[index]

        frame_range = self.phase_raw_data[key]['Frame_Range']
        # img_list = self.full_data[f'{self.phase}/{key}']['imgs_path']
        tgt_sample = self.phase_raw_data[key]['Translation']

        sample = []
        for fr in frame_range:
            imgs = []
            for frame_path in fr[::2]:
                img = cv2.imread(str(self.img_root / frame_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(Image.fromarray(img))
            sample.append(imgs)
            
        pseudo_gloss = self.trans_to_pseudo[tgt_sample]

        input_text_embed = self.phase_sent_embed[key]['token_input_embeddings']
        output_text_embed = self.phase_sent_embed[key]['token_output_embeddings']
        input_text_embed = torch.tensor(input_text_embed, dtype=torch.float32)
        output_text_embed = torch.tensor(output_text_embed, dtype=torch.float32)

        img_sample = [self.load_imgs(s) for s in sample]

        return key, img_sample, tgt_sample, input_text_embed, output_text_embed, pseudo_gloss
    
    def load_imgs(self, batch_image):
        imgs = torch.zeros(len(batch_image),3, self.args.input_size,self.args.input_size)
        crop_rect, resize = utils.data_augmentation(resize=(self.args.resize, self.args.resize), crop_size=self.args.input_size, is_train=(self.phase=='train'))
        
        if self.phase == 'train':
            batch_image = self.seq(batch_image)

        for i, img in enumerate(batch_image):
            img = img.resize(resize, resample=Image.BICUBIC)
            img = self.data_transform(img).unsqueeze(0)
            imgs[i,:,:,:] = img[:,:,crop_rect[1]:crop_rect[3],crop_rect[0]:crop_rect[2]]
        return imgs

    def collate_fn(self,batch):
        
        src_length_batch, new_padded_img, new_src_lengths = [],[], []
        name_batch, img_tmp, tgt_batch, input_embed, output_embed, batch_pgloss = map(list, zip(*batch))
        
        seg_ids = []
        for count,image_samp in enumerate(img_tmp):
            padding = 2
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(padding, -1, -1, -1), # [None] is just .unsqueeze(0). left_pad is the temporal dim, rest of -1 is for C, H, W, where -1 specifies that it should be same as the original
                    vid,
                    vid[-1][None].expand(padding, -1, -1, -1),
                )
                , dim=0) # Pad at 0th Dimension (Temporally)
                for vid in image_samp]
            seg_ids.extend([count for _ in range(len(padded_video))])
            src_length_batch.extend([len(vid) for vid in padded_video])
            new_src_lengths.extend([len(vid) - 2*padding for vid in padded_video])
            new_padded_img.append(torch.cat(padded_video,0))
            
        src_length_batch = torch.tensor(src_length_batch)
        new_src_lengths = torch.tensor(new_src_lengths)
        seg_ids = torch.tensor(seg_ids)
        img_batch = torch.cat(new_padded_img,0)

        new_src_lengths = new_src_lengths.long()
        mask_gen = []
        for i in new_src_lengths:
            tmp = torch.ones([i]) + 7
            mask_gen.append(tmp)
        mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX,batch_first=True)
        img_padding_mask = (mask_gen != PAD_IDX).long()     

        src_input = {}

        input_embed_padded = pad_sequence(input_embed, padding_value=PAD_IDX, batch_first=True)
        output_embed_padded = pad_sequence(output_embed, padding_value=PAD_IDX, batch_first=True)
        text_padding_mask = (input_embed_padded[:, :, 0] != PAD_IDX).long()

        src_input['input_ids'] = img_batch
        src_input['attention_mask'] = img_padding_mask
        src_input['name_batch'] = name_batch
        src_input['seg_ids'] = seg_ids
        src_input['input_embed'] = input_embed_padded
        src_input['output_embed'] = output_embed_padded
        src_input['text_padding_mask'] = text_padding_mask

        src_input['src_length_batch'] = src_length_batch
        src_input['new_src_length_batch'] = new_src_lengths
        src_input['batch_pgloss'] = batch_pgloss

        tgt_input = self.tokenizer(text_target=tgt_batch, return_tensors="pt", padding=True, truncation=True)

        return src_input, tgt_input
    

    def __str__(self):
        return f'#total {self.phase} set: {len(self.raw_data_name)} Datapoints.'






