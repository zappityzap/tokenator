# tokenator
compare tokens between stablediffusion models

Initial script from u/funkmasterplex: https://www.reddit.com/r/StableDiffusion/comments/154xnmm/comment/jss3mt7/

"Something you can do with checkpoints is compare the vectors for each token against the base 1.5 model and see which vectors vary the most. This will give you an idea of what words that checkpoint was trained the most on (or if it's a merge, which words the original models used for the merge were trained on)." --u/funkmasterflex


## Setup
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
The first rule of diffusion club is: don't download ckpt files.

The second rule of diffusion club is: **don't download ckpt files!**

**Inputs must be safetensor format.**

Convert all your models to safetensors. It's quick and easy in [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Run
```
python3 tokenator.py --file1 <...> --file2 <...>
```
## Example
SD 1.5 vs deliberate_v2:
```
$ python3 tokenator.py --file1 ~/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors \
--file2 ~/stable-diffusion-webui/models/Stable-diffusion/deliberate_v2.safetensors 
100% 49408/49408 [00:02<00:00, 17389.67it/s]
dusk         error: 0.07440
lying        error: 0.06915
lips         error: 0.06494
chromatic    error: 0.06378
purple       error: 0.06342
evening      error: 0.06262
cosplay      error: 0.06186
galaxy       error: 0.06000
stomach      error: 0.05997
fac          error: 0.05957
tokyo        error: 0.05957
chest        error: 0.05939
focused      error: 0.05930
outdoors     error: 0.05917
grass        error: 0.05881
orange       error: 0.05817
mustache     error: 0.05801
sted         error: 0.05795
sun          error: 0.05774
business     error: 0.05734
skinny       error: 0.05710
large        error: 0.05707
ty           error: 0.05701
aber         error: 0.05701
fashion      error: 0.05695
female       error: 0.05676
extreme      error: 0.05667
small        error: 0.05652
happy        error: 0.05634
african      error: 0.05630
travel       error: 0.05621
face         error: 0.05612
chinese      error: 0.05609
ration       error: 0.05603
clothes      error: 0.05576
no           error: 0.05554
asian        error: 0.05548
kimono       error: 0.05548
monochrome   error: 0.05533
foreground   error: 0.05524
blurry       error: 0.05521
natural      error: 0.05518
bush         error: 0.05499
many         error: 0.05484
bun          error: 0.05484
lift         error: 0.05478
sky          error: 0.05463
necklace     error: 0.05463
flat         error: 0.05460
huge         error: 0.05457
background   error: 0.05457
9            error: 0.05447
out          error: 0.05435
oil          error: 0.05435
horizon      error: 0.05435
photograph   error: 0.05420
nude         error: 0.05417
graphy       error: 0.05414
profile      error: 0.05405
metal        error: 0.05402
anime        error: 0.05392
5            error: 0.05389
standing     error: 0.05389
cover        error: 0.05368
riding       error: 0.05362
content      error: 0.05350
day          error: 0.05347
girl         error: 0.05347
doors        error: 0.05347
pale         error: 0.05347
ly           error: 0.05344
vibrant      error: 0.05344
her          error: 0.05341
frame        error: 0.05341
town         error: 0.05334
starry       error: 0.05334
ornament     error: 0.05328
reflection   error: 0.05325
indoors      error: 0.05322
smile        error: 0.05313
blueberry    error: 0.05313
flare        error: 0.05310
muscular     error: 0.05301
urban        error: 0.05295
teen         error: 0.05292
driving      error: 0.05286
very         error: 0.05280
autumn       error: 0.05280
endless      error: 0.05276
looking      error: 0.05267
craft        error: 0.05267
posing       error: 0.05267
poolside     error: 0.05267
simple       error: 0.05258
cleavage     error: 0.05258
k            error: 0.05255
dirty        error: 0.05255
different    error: 0.05246
depth        error: 0.05243
brunette     error: 0.05243
```
