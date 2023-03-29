## COPY THIS FILE TO stable-diffusion/ldm/data/music.py

import json
from pathlib import Path
import random
import math

import numpy as np
import torchaudio
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from einops import rearrange

class MusicData(Dataset):
    def __init__(self,
        root_dir,
        caption_file,
        ) -> None:
        """Create a dataset from a folder of music.
        """
        self.log_file = open("/devel/missing.log", "a+")
        self.root_dir = Path(root_dir)
        with open(caption_file, "rt") as f:
            lines = f.readlines()
            lines = [json.loads(x) for x in lines]
            captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
            self.captions = captions
        self.file_extensions = ["_drums.mp3", "_bass.mp3", "_other.mp3"]

        self.sample_rate = 44100
        self.n_fft = 17640
        self.hop_length = 441
        self.win_length = 4410
        self.n_mels = 171
        self.min_freq = 20
        self.max_freq = 20000
        self.mel_scale_norm = None
        self.mel_scale_type = "htk"

        self.spec_scaler = 1000
        self.audio_seconds = 5.11
        self.sample_length = int(self.audio_seconds * self.sample_rate)

        self.spectrogram_func = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad=0,
            window_fn=torch.hann_window,
            power=None,
            normalized=False,
            wkwargs=None,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )

        self.mel_scale_func = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_min=self.min_freq,
            f_max=self.max_freq,
            n_stft=self.n_fft // 2 + 1,
            norm=self.mel_scale_norm,
            mel_scale=self.mel_scale_type,
        )

        self.to_tensor_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.captions.keys())

    def __getitem__(self, index):
        data = {}
        chosen = list(self.captions.keys())[index]
        caption = self.captions.get(chosen, None)
        filename_base = self.root_dir/chosen

        song_length = int(float(caption.split(' ')[-2]))
        if song_length <= self.audio_seconds:
            print(filename_base, file=self.log_file, flush=True)
            return self.__getitem__((index + 1) % self.__len__())
        sample_point = random.randint(0, (song_length - math.ceil(self.audio_seconds)))
        sample_start = sample_point * self.sample_rate

        specs = []
        try:
            for file_extension in self.file_extensions:
                waveform, _ = torchaudio.load(f"{filename_base}{file_extension}", normalize=True, frame_offset=sample_start, num_frames=self.sample_length)
                specs.append(self.gen_spectrogram(waveform))
        except:
            print(filename_base, file=self.log_file, flush=True)
            return self.__getitem__((index + 1) % self.__len__())

        im = self.concat_spectrograms(specs)
        im = self.process_im(im)

        data["image"] = im
        data["txt"] = caption + f" between position {sample_point} and {sample_point + self.audio_seconds}"

        return data

    def gen_spectrogram(self, waveform):
        spectrogram_complex = self.spectrogram_func(waveform)
        amplitudes = torch.abs(spectrogram_complex)
        spectrogram = self.mel_scale_func(amplitudes).numpy()

        data = spectrogram / self.spec_scaler
        data = np.power(data, .25)
        data = data * 255
        return data.astype(np.uint8)

    def concat_spectrograms(self, image_arrays):
        images = [Image.fromarray(image_array[0], mode="L") for image_array in image_arrays]
        index = 0
        for image in images:
            index += 1
        widths, heights = zip(*(image.size for image in images))

        total_height = sum(heights)
        max_width = max(widths)

        result_image = Image.new('L', (max_width, total_height))

        y_offset = 0
        for image in images:
            result_image.paste(image, (0, y_offset))
            y_offset += image.size[1]

        return result_image.convert("RGB")

    def process_im(self, im):
        im = im.resize((512, 512))
        im_tensor = self.to_tensor_transform(im)
        return rearrange(im_tensor * 2. - 1., 'c h w -> h w c')

if __name__ == '__main__':
    md = MusicData("/devel/data/sources/", "/devel/data/caption_files/debug.jsonl")
    for s in md:
        print(s["txt"])
        break
