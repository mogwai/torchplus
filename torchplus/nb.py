from gc import collect

import numpy as np
import torch
from torch import Tensor
from torch.cuda import empty_cache
from torchvision.transforms.functional import to_pil_image

__all__ = ["dp", "clear_memory", "T", ]


def dp(im):
    """
    Displays an image if the program is running in a notebook
    otherwise fails without crashing the program
    """
    if isinstance(im, np.ndarray):
        im = Tensor(im)
    if isinstance(im, Tensor):
        if len(im.shape) == 4:
            im = im[0]
            print("Showing first image")
        im = to_pil_image(im)

    try:
        display(im)
    except Exception as e:
        print(e)


def clear_memory():
    "Clears the memory of GPU"
    collect()
    empty_cache()
    try:
        1 / 0
    except:
        pass

def notebook_sound(url= "https://flukeout.github.io/simple-sounds/sounds/coin.mp3", volume=1):
        from IPython.display import HTML
        
        script = """
<script>
var sounds = {
    "coin" : {
        url : '"""+url+"""'
    }
};

var soundContext = new AudioContext();

for(var key in sounds) {
    loadSound(key);
}

function loadSound(name){
    var sound = sounds[name];

    var url = sound.url;
    var buffer = sound.buffer;

    var request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.responseType = 'arraybuffer';
    request.onload = function() {
        soundContext.decodeAudioData(request.response, function(newBuffer) {
            sound.buffer = newBuffer;
        });
    }
    request.send();
}

function playSound(name, custumVolume){
    var sound = sounds[name];
    var soundVolume = sounds[name].volume || 1;

    var buffer = sound.buffer;
    if(buffer) {

        var source = soundContext.createBufferSource();
        source.buffer = buffer;

        var volume = soundContext.createGain();

        if (custumVolume) {
            volume.gain.value = soundVolume * custumVolume
        } else {
            volume.gain.value = soundVolume;
        }

        volume.connect(soundContext.destination);
        source.connect(volume);
        source.start(0);
    }
}

setTimeout(()=>playSound('coin',"""+str(volume)+"""), 200)
</script>
        """
        from IPython.display import HTML
        display(HTML(script))

# Easy access to make a tensor quickly
T = torch.tensor
