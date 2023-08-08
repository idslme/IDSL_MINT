import torch
from torch import nn

from IDSL_MINT.utils.SMILES_utils import SMILER
from IDSL_MINT.utils.get_future_masks import get_future_masks


class MS2SMILES_Model(nn.Module):
    def __init__(self,
                 NUM_MZ_VOCABS: int = 95003,
                 D_MODEL: int = 512, # general dimension of the model
                 MZ_EMBED_MAX_NORM: int = 2,
                 MZ_DROPOUT: float = 0.1,
                 MAX_SMILES_sequence_length: int = 250,
                 SMILES_EMBED_MAX_NORM: float = 2,
                 SEQ_EMBED_MAX_NORM: float = 2,
                 SEQ_DROPOUT: float = 0.1,
                 NUM_HEAD: int = 8, # numbber attention heads
                 NUM_ENCODER_LAYER: int = 6,
                 NUM_DECODER_LAYER: int = 6,
                 TRANSFORMER_DROPOUT: float = 0.1,
                 TRANSFORMER_ACTIVATION: str = 'relu',
                 ) -> torch.Tensor:
        
        super().__init__()
        self.D_model = torch.tensor(D_MODEL)
        self.max_SMILES_sequence_length = MAX_SMILES_sequence_length
        
        self.mzTokenEmbedding = nn.Embedding(num_embeddings = (NUM_MZ_VOCABS + 1), embedding_dim = D_MODEL, padding_idx = None, max_norm = MZ_EMBED_MAX_NORM, sparse = False)
        self.mzDropout = nn.Dropout(p = MZ_DROPOUT)
        
        self.seqSMILESembedding = nn.Embedding(num_embeddings = 591, embedding_dim = D_MODEL, padding_idx = 0, max_norm = SMILES_EMBED_MAX_NORM, sparse = False)
        self.seqSMILESpositionalEmbedding = nn.Embedding(num_embeddings = (MAX_SMILES_sequence_length + 1), embedding_dim = D_MODEL, padding_idx = 0, max_norm = SEQ_EMBED_MAX_NORM, sparse = False)
        self.seqSMILESdropout = nn.Dropout(p = SEQ_DROPOUT)

        self.mzTransformer = nn.Transformer(d_model = D_MODEL, nhead = NUM_HEAD, num_encoder_layers = NUM_ENCODER_LAYER, num_decoder_layers = NUM_DECODER_LAYER,
                                            dim_feedforward = 4*D_MODEL, dropout = TRANSFORMER_DROPOUT, activation = TRANSFORMER_ACTIVATION, batch_first = False)

        self.Perception = nn.Linear(in_features = D_MODEL, out_features = 590, bias = True)



    def PeakEmbedding(self, MZ_Tokens, INT):
        
        z = self.mzTokenEmbedding(MZ_Tokens)*torch.sqrt(self.D_model) # each m/z has d (=512) features (embeddings)

        device = z.device

        idx2i = torch.arange(0, self.D_model, 2).to(device)
        idx1i = torch.arange(1, self.D_model, 2).to(device)

        zPosEnc = torch.zeros_like(z).to(device)

        zPosEnc[:, :, idx2i] = torch.sin(idx2i*INT/(10000**(2*idx2i/self.D_model)))
        zPosEnc[:, :, idx1i] = torch.cos(idx1i*INT/(10000**(2*idx1i/self.D_model)))
        
        return self.mzDropout(z + zPosEnc)



    def SMILESembedding(self, seqSMILES_Tokens):
        
        z = self.seqSMILESembedding(seqSMILES_Tokens) # each m/z has d (=512) features (embeddings)

        zPosEmb = torch.stack([torch.arange(0, self.max_SMILES_sequence_length, 1) + 1] * seqSMILES_Tokens.shape[0])
        z_0 = torch.where(seqSMILES_Tokens == 0, torch.tensor(True), torch.tensor(False))
        zPosEmb[z_0] = 0

        zPosEmb = self.seqSMILESpositionalEmbedding(zPosEmb)
        
        return self.seqSMILESdropout(z + zPosEmb)



    def forward(self, MZ_Tokens, INT, seqSMILES_Tokens, seqSMILESpaddingMask = None):

        Source = self.PeakEmbedding(MZ_Tokens, INT).permute(1, 0, 2)
        Target = self.SMILESembedding(seqSMILES_Tokens).permute(1, 0, 2)

        seqSMILESfutureMask = get_future_masks(seqSMILES_Tokens.shape[1]).to(seqSMILES_Tokens.device)

        TransformedMZ = self.mzTransformer(src = Source, tgt = Target, src_mask = None, tgt_mask = seqSMILESfutureMask, memory_mask = None,
                                           src_key_padding_mask = None, tgt_key_padding_mask = seqSMILESpaddingMask, memory_key_padding_mask = None)

        return self.Perception(TransformedMZ.permute(1, 0, 2))


    
    def beam_search_inference(self, arg):

        MZ_Tokens_vector = arg[0]
        INT_vector = arg[1]
        beam_size = arg[2]
        device = arg[3]
        msp_block_name = arg[4]

        MZ_Tokens_vector = torch.tensor(MZ_Tokens_vector, dtype = torch.int).unsqueeze(dim = 0).to(device)
        INT_vector = torch.tensor(INT_vector, dtype = torch.float32).unsqueeze(dim = 0).unsqueeze(dim = 2).to(device)

        seqSMILESpadded = torch.zeros(size = (1, (self.max_SMILES_sequence_length - 1)))  # 0 is the token number for padding
        seqSMILES = torch.cat((torch.tensor([[1]]), seqSMILESpadded), dim = 1)
        with torch.inference_mode():
            logits = self.forward(MZ_Tokens_vector, INT_vector, (seqSMILES).type(torch.int).to(device))

        probs = torch.log(torch.softmax(logits[0, 0, :], dim = 0))
        scores, indices = probs.topk(beam_size, dim = 0)

        beams = {"seqSMILES_Tokens": [], "Scores": []}


        for b in range(beam_size):
            beams["seqSMILES_Tokens"].append(torch.tensor([1, indices[b]]).view(1, -1))
            beams["Scores"].append(scores[b])


        for _ in range(1, self.max_SMILES_sequence_length):

            Scores, Indices = [], []
            
            for b in range(beam_size):

                seqSMILES = beams["seqSMILES_Tokens"][b]
                if seqSMILES[0, -1].item() != 2:
                    
                    decoding_depth = seqSMILES.shape[1]

                    seqSMILESpadded = torch.zeros(size = (1, (self.max_SMILES_sequence_length - decoding_depth))) # 0 is the token number for padding
                    seqSMILES = torch.cat((seqSMILES, seqSMILESpadded), dim = 1)
                    with torch.inference_mode():
                        logits = self.forward(MZ_Tokens_vector, INT_vector, seqSMILES.type(torch.int).to(device))

                    probs = torch.log(torch.softmax(logits[0, (decoding_depth - 1), :], dim = 0))
                    scores, indices = probs.topk(beam_size, dim = 0)
                    
                    
                    for s in range(beam_size):
                        new_score = scores[s] + beams["Scores"][b]
                        Scores.append(new_score)
                        Indices.append(indices[s])

                else:
                    for s in range(beam_size):
                        Scores.append(beams["Scores"][b])
                        Indices.append(torch.tensor(2))

            indScores, ind = torch.stack(Scores).topk(beam_size)
            index_SMILES_token = (ind % beam_size).tolist()


            new_SMILES_token = []
            for b in range(beam_size):

                if beams["seqSMILES_Tokens"][b][0, -1].item() != 2:
                    new_SMILES_token.append(torch.cat((beams["seqSMILES_Tokens"][index_SMILES_token[b]], Indices[b].view(-1, 1)), dim = 1).type(torch.int))
                else:
                    new_SMILES_token.append(beams["seqSMILES_Tokens"][index_SMILES_token[b]])

            beams["seqSMILES_Tokens"] = new_SMILES_token
            beams["Scores"] = indScores

        Scores = (torch.softmax(beams["Scores"], dim = -1).numpy()*100).tolist()

        SMILES = []
        for smiles_tokens in beams["seqSMILES_Tokens"]:
            SMILES.append(SMILER(smiles_tokens.squeeze(dim = 0).numpy()))
        
        
        return msp_block_name, Scores, SMILES