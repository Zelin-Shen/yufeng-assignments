import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Enhanced Image Encoder using pretrained ResNet-101
    Extracts visual features with modern improvements
    """

    def __init__(self, encoded_image_size=14):
        """
        Initialize the encoder
        
        Args:
            encoded_image_size: Size of the encoded feature map (default: 14x14)
        """
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # Load pretrained ResNet-101
        resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        
        # Remove final layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # Feature refinement with residual connection
        self.feature_refine = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        self.fine_tune(fine_tune=False)
        self._init_feature_refine()

    def _init_feature_refine(self):
        """Initialize feature refinement layers"""
        for m in self.feature_refine.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images):
        """
        Forward propagation
        
        Args:
            images: Input images, shape (batch_size, 3, image_size, image_size)
            
        Returns:
            Encoded images: shape (batch_size, encoded_image_size, encoded_image_size, 2048)
        """
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        
        # Residual feature refinement
        out = out + self.feature_refine(out)
        
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """Enable or disable fine-tuning"""
        for p in self.resnet.parameters():
            p.requires_grad = False
        
        if fine_tune:
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = True
        
        for p in self.feature_refine.parameters():
            p.requires_grad = True


class AdaptiveAttention(nn.Module):
    """
    Modern Multi-Head Adaptive Attention (Bahdanau-style)
    
    Improvements while keeping the Bahdanau additive attention framework:
    - Multi-head mechanism for richer representations
    - Layer normalization for stability
    - Enhanced non-linearity (GELU instead of Tanh)
    - Better initialization
    - Residual connections
    
    Maintains the original "additive attention" semantic but with modern enhancements.
    """
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim, num_heads=4, dropout=0.1):
        """
        Initialize modern adaptive attention
        
        Args:
            encoder_dim: Encoder output dimension (2048)
            decoder_dim: Decoder hidden state dimension
            attention_dim: Attention intermediate dimension (must be divisible by num_heads)
            num_heads: Number of attention heads (default: 4, smaller than Transformer for efficiency)
            dropout: Dropout probability
        """
        super(AdaptiveAttention, self).__init__()
        
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        # Multi-head encoder transformation (replaces single encoder_att)
        self.encoder_att_heads = nn.ModuleList([
            nn.Linear(encoder_dim, self.head_dim) for _ in range(num_heads)
        ])
        
        # Multi-head decoder transformation (replaces single decoder_att)
        self.decoder_att_heads = nn.ModuleList([
            nn.Linear(decoder_dim, self.head_dim) for _ in range(num_heads)
        ])
        
        # Attention score computation for each head (replaces single full_att)
        self.score_heads = nn.ModuleList([
            nn.Linear(self.head_dim, 1) for _ in range(num_heads)
        ])
        
        # Modern activation: GELU performs better than Tanh in many cases
        self.activation = nn.GELU()
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(encoder_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output projection to combine multi-head results
        self.output_proj = nn.Linear(encoder_dim * num_heads, encoder_dim)
        
        # Head importance weights (learnable)
        self.head_weights = nn.Parameter(torch.ones(num_heads))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with modern strategies"""
        # Xavier initialization for all linear layers
        for head_modules in [self.encoder_att_heads, self.decoder_att_heads, self.score_heads]:
            for module in head_modules:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Output projection
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)
        
        # Head importance weights
        nn.init.constant_(self.head_weights, 1.0 / self.num_heads)
    
    def forward(self, encoder_out, decoder_hidden):
        """
        Compute multi-head adaptive attention (Bahdanau-style)
        
        Args:
            encoder_out: Encoded features, shape (batch_size, num_pixels, encoder_dim)
            decoder_hidden: Decoder hidden state, shape (batch_size, decoder_dim)
            
        Returns:
            context: Attention-weighted context, shape (batch_size, encoder_dim)
            alpha: Combined attention weights, shape (batch_size, num_pixels)
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        
        # Store for residual connection
        residual = encoder_out.mean(dim=1)
        
        # Compute attention for each head
        head_contexts = []
        head_alphas = []
        
        for i in range(self.num_heads):
            # Transform encoder features for this head
            # Shape: (batch_size, num_pixels, head_dim)
            att1 = self.encoder_att_heads[i](encoder_out)
            
            # Transform decoder hidden state for this head
            # Shape: (batch_size, head_dim)
            att2 = self.decoder_att_heads[i](decoder_hidden)
            
            # Bahdanau additive attention mechanism
            # Add att2 (broadcast) to att1, apply activation, then score
            # Shape: (batch_size, num_pixels)
            att_combined = self.activation(att1 + att2.unsqueeze(1))
            att_scores = self.score_heads[i](att_combined).squeeze(2)
            
            # Normalize to get attention weights for this head
            # Shape: (batch_size, num_pixels)
            alpha_head = F.softmax(att_scores, dim=1)
            alpha_head = self.dropout(alpha_head)
            
            # Compute context for this head
            # Shape: (batch_size, encoder_dim)
            context_head = (encoder_out * alpha_head.unsqueeze(2)).sum(dim=1)
            
            head_contexts.append(context_head)
            head_alphas.append(alpha_head)
        
        # Combine contexts from all heads with learned importance weights
        # Stack: (batch_size, num_heads, encoder_dim)
        stacked_contexts = torch.stack(head_contexts, dim=1)
        
        # Normalize head weights
        normalized_head_weights = F.softmax(self.head_weights, dim=0)
        
        # Weighted combination of head contexts
        # Shape: (batch_size, encoder_dim)
        combined_context = (stacked_contexts * normalized_head_weights.view(1, -1, 1)).sum(dim=1)
        
        # Apply residual connection and layer normalization
        context = self.layer_norm(combined_context + residual)
        
        # Combine attention weights (weighted average across heads)
        # Stack: (batch_size, num_heads, num_pixels)
        stacked_alphas = torch.stack(head_alphas, dim=1)
        
        # Weighted average: (batch_size, num_pixels)
        alpha = (stacked_alphas * normalized_head_weights.view(1, -1, 1)).sum(dim=1)
        
        return context, alpha


class DecoderWithAdaptiveAttention(nn.Module):
    """
    Enhanced LSTM Decoder with Modern Adaptive Attention
    
    Maintains the "Knowing When to Look" framework while incorporating:
    - Modern multi-head adaptive attention
    - Enhanced visual sentinel mechanism
    - Layer normalization throughout
    - Improved gating and initialization
    - Residual connections where appropriate
    """
    
    def __init__(self, cfg, encoder_dim=2048):
        """
        Initialize decoder
        
        Args:
            cfg: Configuration dictionary containing:
                - decoder_dim: LSTM hidden dimension
                - attention_dim: Attention dimension
                - embed_dim: Embedding dimension
                - vocab_size: Vocabulary size
                - dropout: Dropout probability
                - device: Computation device
            encoder_dim: Encoder output dimension (default: 2048)
        """
        super(DecoderWithAdaptiveAttention, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = cfg['decoder_dim']
        self.attention_dim = cfg['attention_dim']
        self.embed_dim = cfg['embed_dim']
        self.vocab_size = cfg['vocab_size']
        self.dropout = cfg['dropout']
        self.device = cfg['device']
        
        # Modern multi-head adaptive attention (Bahdanau-style)
        self.attention = AdaptiveAttention(
            encoder_dim, 
            self.decoder_dim, 
            self.attention_dim,
            num_heads=4,  # 4 heads for good balance of performance and efficiency
            dropout=self.dropout
        )
        
        # Word embedding with layer norm
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        # LSTM cell
        self.decode_step = nn.LSTMCell(
            self.embed_dim + encoder_dim, 
            self.decoder_dim, 
            bias=True
        )
        
        # LSTM output layer norm
        self.lstm_layer_norm = nn.LayerNorm(self.decoder_dim)
        
        # Initialize LSTM states from encoder (with non-linearity)
        self.init_h = nn.Sequential(
            nn.Linear(encoder_dim, self.decoder_dim),
            nn.LayerNorm(self.decoder_dim),
            nn.Tanh()
        )
        self.init_c = nn.Sequential(
            nn.Linear(encoder_dim, self.decoder_dim),
            nn.LayerNorm(self.decoder_dim),
            nn.Tanh()
        )
        
        # Enhanced context gating mechanism
        self.f_beta = nn.Sequential(
            nn.Linear(self.decoder_dim, encoder_dim),
            nn.LayerNorm(encoder_dim)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Enhanced visual sentinel mechanism
        sentinel_input_dim = self.decoder_dim * 2 + self.embed_dim + encoder_dim
        self.sentinel_gate = nn.Sequential(
            nn.Linear(sentinel_input_dim, self.decoder_dim * 2),
            nn.LayerNorm(self.decoder_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.decoder_dim * 2, self.decoder_dim),
            nn.Tanh()
        )
        
        # Visual sentinel attention (enhanced with MLP)
        self.visual_sentinel_att = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim // 2),
            nn.LayerNorm(self.decoder_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.decoder_dim // 2, 1)
        )
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim),
            nn.LayerNorm(self.decoder_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.decoder_dim, self.vocab_size)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Initialize all weights with modern strategies"""
        # Embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # LSTM: orthogonal initialization for recurrent weights
        for name, param in self.decode_step.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Forget gate bias = 1 (helps with long-term dependencies)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        # All other linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in [self.decode_step]:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def load_pretrained_embeddings(self, embeddings):
        """Load pretrained embeddings"""
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """Enable/disable embedding fine-tuning"""
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def _compute_sentinel(self, h_prev, c_prev, embeddings_t, context):
        """
        Compute enhanced visual sentinel
        
        The sentinel represents the decoder's internal language model state,
        enhanced with modern architecture (MLP with GELU, LayerNorm, Dropout).
        
        Args:
            h_prev: Previous hidden state, shape (batch_size, decoder_dim)
            c_prev: Previous cell state, shape (batch_size, decoder_dim)
            embeddings_t: Current embedding, shape (batch_size, embed_dim)
            context: Visual context, shape (batch_size, encoder_dim)
            
        Returns:
            sentinel: Shape (batch_size, decoder_dim)
        """
        sentinel_input = torch.cat([h_prev, c_prev, embeddings_t, context], dim=1)
        sentinel = self.sentinel_gate(sentinel_input)
        return sentinel
    
    def _adaptive_attention_step(self, context, alpha_spatial, sentinel, h):
        """
        Apply adaptive attention mechanism
        
        This is the core "Knowing When to Look" mechanism:
        - Balances visual attention (context) vs language model (sentinel)
        - Beta indicates reliance: high beta = more language, low beta = more visual
        
        Args:
            context: Spatial attention context, shape (batch_size, encoder_dim)
            alpha_spatial: Spatial weights, shape (batch_size, num_pixels)
            sentinel: Sentinel vector, shape (batch_size, decoder_dim)
            h: Hidden state, shape (batch_size, decoder_dim)
            
        Returns:
            final_context: Adaptively weighted context, shape (batch_size, encoder_dim)
            alpha_final: Final spatial weights, shape (batch_size, num_pixels)
            beta: Sentinel weight, shape (batch_size, 1)
        """
        # Compute sentinel attention score
        sentinel_score = self.visual_sentinel_att(sentinel)
        
        # Extend attention distribution to include sentinel
        extended_scores = torch.cat([alpha_spatial, sentinel_score], dim=1)
        extended_alpha = F.softmax(extended_scores, dim=1)
        
        # Split into spatial attention and sentinel weight
        alpha_final = extended_alpha[:, :-1]
        beta = extended_alpha[:, -1:]
        
        # Smooth beta to prevent extreme values (helps stability)
        beta = torch.clamp(beta, min=0.01, max=0.99)
        
        # Project sentinel to encoder dimension
        sentinel_proj = self.f_beta(sentinel)
        
        # Compute final context as weighted combination
        final_context = (1 - beta) * context + beta * sentinel_proj
        
        return final_context, alpha_final, beta
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation with teacher forcing
        
        Args:
            encoder_out: Encoded images, shape (batch_size, enc_image_size, enc_image_size, encoder_dim)
            encoded_captions: Encoded captions, shape (batch_size, max_caption_length)
            caption_lengths: Caption lengths, shape (batch_size, 1)
            
        Returns:
            predictions: Prediction scores, shape (batch_size, max_decode_length, vocab_size)
            sorted_captions: Sorted captions
            decode_lengths: Decode lengths
            alphas: Spatial attention weights, shape (batch_size, max_decode_length, num_pixels)
            betas: Sentinel gates, shape (batch_size, max_decode_length, 1)
            sort_ind: Sorting indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        # Flatten encoder output
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Sort by caption length (required for efficient packing)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # Get embeddings with layer normalization
        embeddings = self.embedding(encoded_captions)
        embeddings = self.embed_layer_norm(embeddings)
        
        # Decode lengths (exclude <end> token)
        decode_lengths = (caption_lengths - 1).tolist()
        
        # Initialize output tensors
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)
        betas = torch.zeros(batch_size, max(decode_lengths), 1).to(self.device)
        
        # Initialize LSTM states from encoder output mean
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        
        # Decode step by step
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            
            # Current embeddings
            embeddings_t = embeddings[:batch_size_t, t, :]
            
            # Compute spatial attention
            context, alpha_spatial = self.attention(
                encoder_out[:batch_size_t], 
                h[:batch_size_t]
            )
            
            # Gate the context
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            context_gated = gate * context
            
            # Compute sentinel
            sentinel = self._compute_sentinel(
                h[:batch_size_t],
                c[:batch_size_t],
                embeddings_t,
                context_gated
            )
            
            # LSTM input
            lstm_input = torch.cat([embeddings_t, context_gated], dim=1)
            lstm_input = self.dropout_layer(lstm_input)
            
            # LSTM step
            h_new, c_new = self.decode_step(
                lstm_input, 
                (h[:batch_size_t], c[:batch_size_t])
            )
            
            # Apply layer normalization to hidden state
            h_new = self.lstm_layer_norm(h_new)
            
            # Adaptive attention
            final_context, alpha_final, beta = self._adaptive_attention_step(
                context_gated,
                alpha_spatial,
                sentinel,
                h_new
            )
            
            # Predict
            preds = self.fc(h_new)
            
            # Update states
            if batch_size_t < batch_size:
                h = torch.cat([h_new, h[batch_size_t:]], dim=0)
                c = torch.cat([c_new, c[batch_size_t:]], dim=0)
            else:
                h = h_new
                c = c_new
            
            # Store outputs
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha_final
            betas[:batch_size_t, t, :] = beta
            
        return predictions, encoded_captions, decode_lengths, alphas, betas, sort_ind
    
    def one_step(self, embeddings, encoder_out, h, c):
        """
        Single decode step for beam search
        
        Args:
            embeddings: Word embeddings, shape (batch_size, embed_dim)
            encoder_out: Encoded images, shape (batch_size, num_pixels, encoder_dim)
            h: Hidden state, shape (batch_size, decoder_dim)
            c: Cell state, shape (batch_size, decoder_dim)
            
        Returns:
            preds: Predictions, shape (batch_size, vocab_size)
            alpha: Attention weights, shape (batch_size, num_pixels)
            beta: Sentinel gate, shape (batch_size, 1)
            h_new: New hidden state, shape (batch_size, decoder_dim)
            c_new: New cell state, shape (batch_size, decoder_dim)
        """
        # Normalize embeddings
        embeddings = self.embed_layer_norm(embeddings)
        
        # Compute attention
        context, alpha_spatial = self.attention(encoder_out, h)
        
        # Gate context
        gate = self.sigmoid(self.f_beta(h))
        context_gated = gate * context
        
        # Compute sentinel
        sentinel = self._compute_sentinel(h, c, embeddings, context_gated)
        
        # LSTM input (no dropout during inference)
        lstm_input = torch.cat([embeddings, context_gated], dim=1)
        
        # LSTM step
        h_new, c_new = self.decode_step(lstm_input, (h, c))
        h_new = self.lstm_layer_norm(h_new)
        
        # Adaptive attention
        final_context, alpha_final, beta = self._adaptive_attention_step(
            context_gated,
            alpha_spatial,
            sentinel,
            h_new
        )
        
        # Predict (no dropout)
        preds = self.fc(h_new)
        
        return preds, alpha_final, beta, h_new, c_new


# Backward compatibility aliases
DecoderWithAttention = DecoderWithAdaptiveAttention
Attention = AdaptiveAttention


class DecoderWithRNN(nn.Module):
    """
    Legacy RNN Decoder - Backward compatibility wrapper
    Redirects to DecoderWithAdaptiveAttention
    """
    
    def __init__(self, cfg, encoder_dim=14*14*2048):
        super(DecoderWithRNN, self).__init__()
        print("Warning: DecoderWithRNN is deprecated. Using DecoderWithAdaptiveAttention instead.")
        self.decoder = DecoderWithAdaptiveAttention(cfg, encoder_dim=2048)
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """Forward pass for backward compatibility"""
        predictions, encoded_captions, decode_lengths, alphas, betas, sort_ind = \
            self.decoder(encoder_out, encoded_captions, caption_lengths)
        # Return without betas for compatibility
        return predictions, encoded_captions, decode_lengths, sort_ind
    
    def one_step(self, embeddings, h, c):
        """Not implemented - use DecoderWithAdaptiveAttention"""
        raise NotImplementedError("Use DecoderWithAdaptiveAttention for beam search")