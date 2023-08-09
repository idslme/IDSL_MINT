import torch
from torch import nn


class FP2MS_Model(nn.Module):
    def __init__(self,
                 NUM_MZ_VOCABS: int = 95003,
                 D_MODEL: int = 512, # general dimension of the model
                 MZ_EMBED_MAX_NORM: int = 2,
                 NUM_FINGERPRINT_VOCABS: int = 2048,
                 NUM_HEAD: int = 8, # numbber attention heads
                 NUM_ENCODER_LAYER: int = 6,
                 NUM_DECODER_LAYER: int = 6,
                 TRANSFORMER_DROPOUT: float = 0.1,
                 TRANSFORMER_ACTIVATION: str = 'relu',
                 ) -> torch.Tensor:
        super().__init__()
        self.D_model = D_MODEL

        self.nBitsEmbedding = nn.Embedding(num_embeddings = (NUM_FINGERPRINT_VOCABS + 1), embedding_dim = D_MODEL, padding_idx = None, max_norm = 2, sparse = False)
        
        self.mzTokenEmbedding = nn.Embedding(num_embeddings = (NUM_MZ_VOCABS + 1), embedding_dim = D_MODEL, padding_idx = 0, max_norm = MZ_EMBED_MAX_NORM, sparse = False)
        
        self.mzTransformer = nn.Transformer(d_model = D_MODEL, nhead = NUM_HEAD, num_encoder_layers = NUM_ENCODER_LAYER, num_decoder_layers = NUM_DECODER_LAYER,
                                            dim_feedforward = 4*D_MODEL, dropout = TRANSFORMER_DROPOUT, activation = TRANSFORMER_ACTIVATION, batch_first = False)

        self.Perception = nn.Linear(in_features = D_MODEL, out_features = NUM_MZ_VOCABS, bias = True)


    def forward(self, FP_Tokens, MZ_Tokens, src_key_padding_mask = None) -> torch.Tensor:

        Source = self.nBitsEmbedding(FP_Tokens).permute(1, 0, 2)
        Target = self.mzTokenEmbedding(MZ_Tokens).permute(1, 0, 2)

        TransformedMZ = self.mzTransformer(src = Source, tgt = Target, src_mask = None, tgt_mask = None, memory_mask = None,
                                           src_key_padding_mask = src_key_padding_mask, tgt_key_padding_mask = None, memory_key_padding_mask = None)

        return self.Perception(TransformedMZ.permute(1, 0, 2))

    
    def beam_search_inference(self, arg):

        FP_Tokens_vector = arg[0]
        beam_size = arg[1]
        device = arg[2]

        FP_Tokens_vector = torch.tensor(FP_Tokens_vector, dtype = torch.int).unsqueeze(dim = 0).to(device)

        with torch.inference_mode():
            logits = self.forward(FP_Tokens_vector, torch.tensor([[1]], device = device).type(torch.int))

        probs = torch.log(torch.softmax(logits[0, 0, :], dim = 0))
        scores, indices = probs.topk(beam_size, dim = 0)

        beams = {"MZ_tokens": [], "Scores": []}


        for b in range(beam_size):
            beams["MZ_tokens"].append(torch.tensor([1, indices[b]]).view(1, -1))
            beams["Scores"].append(scores[b])

        
        for i in range(1, self.D_model):

            Scores, Indices = [], []
            
            for b in range(beam_size):

                with torch.inference_mode():
                    logits = self.forward(FP_Tokens_vector, beams["MZ_tokens"][b].to(device))

                probs = torch.log(torch.softmax(logits[0, i, :], dim = 0))
                scores, indices = probs.topk(beam_size, dim = 0)
                
                
                for s in range(beam_size):
                    new_score = scores[s] + beams["Scores"][b]
                    Scores.append(new_score.to('cpu'))
                    Indices.append(indices[s])

            indScores, ind = torch.stack(Scores).topk(beam_size)
            index_FP_tokens = (ind % beam_size).tolist()


            new_FP_tokens = []
            for b in range(beam_size):
                new_FP_tokens.append(torch.cat((beams["MZ_tokens"][index_FP_tokens[b]], Indices[b].view(-1, 1).to('cpu')), dim = 1).type(torch.int))

            beams["MZ_tokens"] = new_FP_tokens
            beams["Scores"] = indScores

        
        FP_tokens = beams["MZ_tokens"][0]
        FP_tokens = (FP_tokens[~torch.isin(FP_tokens, torch.tensor([0, 1, 2]))].detach().cpu().unique().numpy() - 3).tolist()

        return FP_tokens