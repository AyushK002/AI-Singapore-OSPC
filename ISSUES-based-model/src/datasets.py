import os
import pandas as pd
import torch
import clip

from PIL import Image
from torch.utils.data import Dataset

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TQDM_DISABLE'] = "1"


class MemesDataset(Dataset):
    def __init__(self, root_folder, dataset, split='train', image_size=224, fast=True):
        super(MemesDataset, self).__init__()
        self.root_folder = root_folder
        self.dataset = dataset
        self.split = split

        self.image_size = image_size
        self.fast = fast

        # chek
        self.info_file = os.path.join(root_folder, dataset, f'labels/{dataset}_info.csv')
        self.df = pd.read_csv(self.info_file)
        self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)
        float_cols = self.df.select_dtypes(float).columns
        self.df[float_cols] = self.df[float_cols].fillna(-1).astype('Int64')

        if self.fast:
            self.embds = torch.load(f'{self.root_folder}/{self.dataset}/clip_embds/{split}_no-proj_output.pt')
            self.embdsDF = pd.DataFrame(self.embds)

            assert len(self.embds) == len(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if row['text'] == 'nothing':
            txt = 'null'
        else:
            txt = row['text']

        if self.fast:
            embd_idx = self.embdsDF.loc[self.embdsDF['idx_meme'] == row['id']].index[0]
            embd_row = self.embds[embd_idx]

            # use CLIP pre-calculated embeddings as image and text inputs
            image = embd_row['image']
            text = embd_row['text']

        else:
            # use raw image and text inputs
            if self.dataset == 'hmc':
                image_fn = row['img'].split('/')[1]
            else:
                image_fn = row['image']
            image = Image.open(f"{self.root_folder}/{self.dataset}/img/{image_fn}").convert('RGB')\
                .resize((self.image_size, self.image_size))
            text = txt

        item = {
            'image': image,
            'text': text,
            'label': row['label'],
            'idx_meme': row['id'],
            'origin_text': txt
        }

        return item


class MemesCollator(object):
    def __init__(self, args):
        self.args = args
        if not args.fast_process:
            _, self.clip_preprocess = clip.load("ViT-L/14", device="cuda", jit=False)

    def __call__(self, batch):
        labels = torch.LongTensor([item['label'] for item in batch])
        idx_memes = torch.LongTensor([item['idx_meme'] for item in batch])

        text_input = []
        for el in batch:
            text_input.append(clip.tokenize(f'{"a photo of $"} , {el["origin_text"]}', context_length=77,
                                            truncate=True))

        enh_texts = torch.cat([item for item in text_input], dim=0)

        simple_prompt = clip.tokenize('a photo of $', context_length=77).repeat(labels.shape[0], 1)

        batch_new = {'labels': labels,
                     'idx_memes': idx_memes,
                     'enhanced_texts': enh_texts,
                     'simple_prompt': simple_prompt
                     }

        if self.args.fast_process:
            images_emb = torch.cat([item['image'] for item in batch], dim=0)
            texts_emb = torch.cat([item['text'] for item in batch], dim=0)

            batch_new['images'] = images_emb
            batch_new['texts'] = texts_emb

        else:
            img = []
            texts = []
            for item in batch:
                pixel_values = self.clip_preprocess(item['image']).unsqueeze(0)
                img.append(pixel_values)

                text = clip.tokenize(item['text'], context_length=77, truncate=True)
                texts.append(text)

            pixel_values = torch.cat([item for item in img], dim=0)
            texts = torch.cat([item for item in texts], dim=0)

            batch_new['pixel_values'] = pixel_values
            batch_new['texts'] = texts

        return batch_new

class MCC_Dataset(Dataset):
    def __init__(self, dset, split, mode, image_size):
        super(MCC_Dataset, self).__init__()
        self.dset = dset
        self.split = split
        self.mode = mode
        self.image_size = image_size

        if dset == 'mcc_run':
            self.df = pd.read_csv('./run.csv')
        else:
            self.df = pd.read_csv(f"./resources/datasets/{self.dset}/annotations_ISSUES.csv")

        self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)
        float_cols = self.df.select_dtypes(float).columns
        self.df[float_cols] = self.df[float_cols].fillna(-1).astype('Int64')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if len(str(row['text'])) == 0:
            txt = 'null'
        else:
            txt = str(row['text'])

        # use raw image and text inputs
        if self.dset == 'mcc_run':
            image_fn = row['image']
            image = Image.open(image_fn).convert('RGB')\
                .resize((self.image_size, self.image_size))
        else:
            image_fn = row['image']
            image = Image.open(f"./resources/datasets/{self.dset}/TD_Memes/{image_fn}").convert('RGB')\
                .resize((self.image_size, self.image_size))
        text = txt

        if self.dset == 'mcc_run':
            item = {
            'image': image,
            'text': text,
            'label': row['label'],
            'idx_meme': row['id'],
            'origin_text': txt
        }
        else:
            item = {
                'image': image,
                'text': text,
                'label': row['label'],
                'idx_meme': int(row['image'][4:-4]) , #row['id'],
                'origin_text': txt
            }

        return item

def load_dataset(args, split):

    if args.dataset != 'hmc' and args.dataset != 'harmeme':
        dataset = MCC_Dataset(dset=args.dataset, split=split, mode=args.reproduce, image_size=args.image_size)
    else:
        dataset = MemesDataset(root_folder=f'./resources/datasets', dataset=args.dataset, split=split,
                            image_size=args.image_size, fast=args.fast_process)

    return dataset
