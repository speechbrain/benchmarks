import torch

class AttentionMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionMLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, x):
        x = self.layers(x)
        att_w = torch.nn.functional.softmax(x, dim=2)
        return att_w
    
    
class Discrete_EmbeddingLayer(torch.nn.Module):
    def __init__(self, num_codebooks, vocab_size, emb_dim,pad_index=0,init=False, freeze=False):
       super(Discrete_EmbeddingLayer, self).__init__()
       self.vocab_size = vocab_size
       self.num_codebooks=num_codebooks
       self.freeze= freeze
       self.embedding =  torch.nn.Embedding(num_codebooks*vocab_size, emb_dim).requires_grad_(not self.freeze)
       #  TODO: handle padding tokens and initialization with embedding from codec

    def forward(self, in_tokens):
        with torch.set_grad_enabled(not self.freeze):
            #  Add unique token IDs across diffrent codebooks by adding num_codebooks * vocab_size
            in_tokens += torch.arange(
                0,
                self.num_codebooks * self.vocab_size,
                self.vocab_size,
                device=in_tokens.device,
            ) 
            # Forward Pass to embedding and 
            in_embs = self.embedding(in_tokens)
            return in_embs

# from speechbrain.lobes.models.huggingface_transformers.encodec import Encodec
# model_hub = "facebook/encodec_24khz"
# save_path = "savedir"
# model = Encodec(model_hub, save_path)
# audio = torch.randn(4, 1000)
# length = torch.tensor([1.0, .5, .75, 1.0])
# tokens, emb = model.encode(audio, length)
# print("hi")      
        
# import torch
# from speechbrain.lobes.models.huggingface_transformers.discrete_ssl import DiscreteSSL
# from speechbrain.lobes.models.huggingface_transformers.hubert import (HuBERT)
# inputs = torch.rand([3, 2000])
# model_hub = "facebook/hubert-large-ll60k"
# save_path = "savedir"
# ssl_layer_num = [7,23]
# deduplicate =[False, True]
# bpe_tokenizers=[None, None]
# kmeans_repo_id = "poonehmousavi/SSL_Quantization"
# kmeans_dataset = "LJSpeech"
# num_clusters = 1000
# ssl_model = HuBERT(model_hub, save_path,output_all_hiddens=True)
# model = DiscreteSSL(save_path, ssl_model, kmeans_repo_id=kmeans_repo_id, kmeans_dataset=kmeans_dataset,num_clusters=num_clusters)
# tokens, embs ,pr_tokens= model(inputs,SSL_layers=ssl_layer_num, deduplicates=deduplicate, bpe_tokenizers=bpe_tokenizers)

# from speechbrain.lobes.models.discrete.dac import DAC

# dac = DAC(load_pretrained=True, model_type="24khz", model_bitrate="8kbps", tag="latest")
# # continuous_embeddings = torch.randn(1, 1024, 100) # Example shape: [Batch, Channels, Time]
# audio = torch.randn(160, 72000)
# # length = torch.tensor([1.0, .5, .75, 1.0])
# tokens , _= dac(audio.unsqueeze(1),n_quantizers=2)
# emb= Discrete_EmbeddingLayer(2, 1024, 1024)
# in_emb = emb(tokens)
# print(in_emb.shape)
        

from speechbrain.lobes.models.discrete.speechtokenizer_interface import SpeechTokenizer_interface

model_hub = "fnlp/SpeechTokenizer"
save_path = "savedir"
model =SpeechTokenizer_interface(model_hub, save_path)  # doctest: +SKIP
# continuous_embeddings = torch.randn(1, 1024, 100) # Example shape: [Batch, Channels, Time]
audio = torch.randn(160, 2000)
# length = torch.tensor([1.0, .5, .75, 1.0])
tokens = model(audio).permute(1,2,0)[:,:,:2]
emb= Discrete_EmbeddingLayer(2, 1024, 1024)
in_emb = emb(tokens)
print(in_emb.shape)