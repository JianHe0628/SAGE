# global definition
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SI_TOKEN = "<si>"
SI_IDX,PAD_IDX,UNK_IDX,BOS_IDX, EOS_IDX = 0,1,2,3,4
SPECIAL_SYMBOLS = ['<si>','<pad>', '<unk>','<bos>', '<eos>']
WORD_MASK = "<mask>"

SEG_W_GLOSS = {
    '03June_2010_Thursday_tagesschau-2629_frame_3-10': 'Auch', 
    '03June_2010_Thursday_tagesschau-2629_frame_80-92': 'Temperatur',
    '05March_2011_Saturday_tagesschau-5447_frame_21-29': 'Temperatur',
    '05March_2011_Saturday_tagesschau-5447_frame_106-127': 'Grad',
    '05November_2010_Friday_tagesschau-1872_frame_2-9': 'Auch',
    '05November_2010_Friday_tagesschau-1872_frame_35-42': 'Morgen',
    '16December_2009_Wednesday_tagesschau-5912_frame_5-16': 'Tag',
    '16December_2009_Wednesday_tagesschau-5912_frame_17-25': 'Auch',
    '19October_2010_Tuesday_tagesschau-8289_frame_1-12': 'Tag',
    '19October_2010_Tuesday_tagesschau-8289_frame_50-59': 'Auch',
    '04January_2010_Monday_tagesschau-8552_frame_13-19': 'Heute',
    '24August_2010_Tuesday_tagesschau-2196_frame_10-21': 'Nacht',
    '24August_2010_Tuesday_tagesschau-2196_frame_80-85': 'Auch',
    '31March_2010_Wednesday_tagesschau-1000_frame_13-21': 'Nacht',
    '06June_2010_Sunday_tagesschau-6363_frame_3-13': 'Morgen',
    '06June_2010_Sunday_tagesschau-6363_frame_95-105': 'Nacht',
    '17February_2011_Thursday_heute-387_frame_1-9': 'Temperatur',
    '17February_2011_Thursday_heute-387_frame_17-24': 'Grad',
    '27June_2010_Sunday_tagesschau-7392_frame_1-12': 'Morgen',
    '27June_2010_Sunday_tagesschau-7392_frame_24-33': 'Temperatur',
    '27June_2010_Sunday_tagesschau-7392_frame_52-59': 'Grad', 
    '05November_2010_Friday_tagesschau-1872_frame_59-69': 'Regen',
    '06June_2010_Sunday_tagesschau-6363_frame_150-180': 'Regen',
    '24August_2010_Tuesday_tagesschau-2196_frame_52-57': 'Regen',
    '19October_2010_Tuesday_tagesschau-8289_frame_60-71': 'Regen',
    '10February_2010_Wednesday_tagesschau-2514_frame_30-33': 'Auch',
}
