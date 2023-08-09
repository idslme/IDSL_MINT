import torch
from torch import nn

class MS2FP_Model(nn.Module):
    def __init__(self,
                 NUM_MZ_VOCABS: int = 95003,
                 D_MODEL: int = 512, # general dimension of the model
                 MZ_EMBED_MAX_NORM: int = 2,
                 MZ_DROPOUT: float = 0.1,
                 MAX_NUMBER_BITS: int = 166,
                 NUM_FINGERPRINT_VOCABS: int = 2051,
                 NUM_HEAD: int = 8, # numbber attention heads
                 NUM_ENCODER_LAYER: int = 6,
                 NUM_DECODER_LAYER: int = 6,
                 TRANSFORMER_DROPOUT: float = 0.1,
                 TRANSFORMER_ACTIVATION: str = 'relu',
                 ) -> torch.Tensor:
        super().__init__()
        self.D_model = torch.tensor(D_MODEL)
        self.max_number_bits = MAX_NUMBER_BITS

        self.mzTokenEmbedding = nn.Embedding(num_embeddings = (NUM_MZ_VOCABS + 1), embedding_dim = D_MODEL, padding_idx = None, max_norm = MZ_EMBED_MAX_NORM, sparse = False)
        self.mzDropout = nn.Dropout(p = MZ_DROPOUT)
        
        self.nBitsEmbedding = nn.Embedding(num_embeddings = (NUM_FINGERPRINT_VOCABS + 1), embedding_dim = D_MODEL, padding_idx = 0, max_norm = 2, sparse = False)

        self.mzTransformer = nn.Transformer(d_model = D_MODEL, nhead = NUM_HEAD, num_encoder_layers = NUM_ENCODER_LAYER, num_decoder_layers = NUM_DECODER_LAYER,
                                            dim_feedforward = 4*D_MODEL, dropout = TRANSFORMER_DROPOUT, activation = TRANSFORMER_ACTIVATION, batch_first = False)

        self.Perception = nn.Linear(in_features = D_MODEL, out_features = NUM_FINGERPRINT_VOCABS, bias = True)


    def PeakEmbedding(self, MZ_Tokens, INT):
        
        z = self.mzTokenEmbedding(MZ_Tokens)*torch.sqrt(self.D_model) # each m/z has d (=512) features (embeddings)

        device = z.device

        idx2i = torch.arange(0, self.D_model, 2).to(device)
        idx1i = torch.arange(1, self.D_model, 2).to(device)

        zPosEnc = torch.zeros_like(z).to(device)

        zPosEnc[:, :, idx2i] = torch.sin(idx2i*INT/(10000**(2*idx2i/self.D_model)))
        zPosEnc[:, :, idx1i] = torch.cos(idx1i*INT/(10000**(2*idx1i/self.D_model)))
        
        return self.mzDropout(z + zPosEnc)


    def forward(self, MZ_Tokens, INT, FP_Tokens, tgt_key_padding_mask = None) -> torch.Tensor:

        Source = self.PeakEmbedding(MZ_Tokens, INT).permute(1, 0, 2)
        Target = self.nBitsEmbedding(FP_Tokens).permute(1, 0, 2)

        TransformedMZ = self.mzTransformer(src = Source, tgt = Target, src_mask = None, tgt_mask = None, memory_mask = None,
                                           src_key_padding_mask = None, tgt_key_padding_mask = tgt_key_padding_mask, memory_key_padding_mask = None)

        return self.Perception(TransformedMZ.permute(1, 0, 2))

    
    def beam_search_inference(self, arg):

        MZ_Tokens_vector = arg[0]
        INT_vector = arg[1]
        beam_size = arg[2]
        device = arg[3]
        msp_block_name = arg[4]

        MZ_Tokens_vector = torch.tensor(MZ_Tokens_vector, dtype = torch.int).unsqueeze(dim = 0).to(device)
        INT_vector = torch.tensor(INT_vector, dtype = torch.float32).unsqueeze(dim = 0).unsqueeze(dim = 2).to(device)

        with torch.inference_mode():
            logits = self.forward(MZ_Tokens_vector, INT_vector, torch.tensor([[1]], device = device).type(torch.int))

        probs = torch.log(torch.softmax(logits[0, 0, :], dim = 0))
        scores, indices = probs.topk(beam_size, dim = 0)

        beams = {"FP_tokens": [], "Scores": []}


        for b in range(beam_size):
            beams["FP_tokens"].append(torch.tensor([1, indices[b]]).view(1, -1))
            beams["Scores"].append(scores[b])

        
        for i in range(1, self.max_number_bits):

            Scores, Indices = [], []
            
            for b in range(beam_size):

                with torch.inference_mode():
                    logits = self.forward(MZ_Tokens_vector, INT_vector, beams["FP_tokens"][b].to(device))

                probs = torch.log(torch.softmax(logits[0, i, :], dim = 0))
                scores, indices = probs.topk(beam_size, dim = 0)
                
                
                for s in range(beam_size):
                    new_score = scores[s] + beams["Scores"][b]
                    Scores.append(new_score)
                    Indices.append(indices[s])

            indScores, ind = torch.stack(Scores.to('cpu')).topk(beam_size)
            index_FP_tokens = (ind % beam_size).tolist()


            new_FP_tokens = []
            for b in range(beam_size):
                new_FP_tokens.append(torch.cat((beams["FP_tokens"][index_FP_tokens[b]], Indices[b].view(-1, 1).to('cpu')), dim = 1).type(torch.int))

            beams["FP_tokens"] = new_FP_tokens
            beams["Scores"] = indScores.to('cpu')

        
        FP_tokens = beams["FP_tokens"][0]
        FP_tokens = (FP_tokens[~torch.isin(FP_tokens, torch.tensor([0, 1, 2]))].unique().detach().numpy() - 3).tolist()
        output = [msp_block_name, "-".join([str(q) for q in FP_tokens])]

        return output