import torch
from torch import nn
from diffusor.utils import ops
from torch.nn import init
import torch.nn.functional as F


class GPT2SelfAttention(nn.Module):
    def __init__(self, n_positions, n_embd, n_head, attn_pdrop, resid_pdrop, init_index=0):
        super().__init__()

        self.max_pos = n_positions
        self.embd_dim = n_embd
        self.num_heads = n_head
        self.head_dim = self.embd_dim // self.num_heads
        self.attn_dropout = attn_pdrop
        self.resid_dropout = resid_pdrop
        self.scale_attn_weights = True

        self.ln = nn.Linear(self.embd_dim, 3 * self.embd_dim)
        self.drop1 = nn.Dropout(self.attn_dropout)
        self.drop2 = nn.Dropout(self.resid_dropout)
        self.ln2 = nn.Linear(self.embd_dim, self.embd_dim)
        self.init_index = init_index

        self.initialize()

    def initialize(self):
        if len(self.ln.weight.shape) < 2:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln.weight.unsqueeze(0))
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln.weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_uniform_(self.ln.weight.unsqueeze(0))
        else:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln.weight)
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln.weight)
            else:
                torch.nn.init.kaiming_uniform_(self.ln.weight)
        # init.xavier_uniform_(self.ln.weight)
        init.zeros_(self.ln.bias)
        if len(self.ln2.weight.shape) < 2:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln2.weight.unsqueeze(0))
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln2.weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_uniform_(self.ln2.weight.unsqueeze(0))
        else:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln2.weight)
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln2.weight)
            else:
                torch.nn.init.kaiming_uniform_(self.ln2.weight)
        init.zeros_(self.ln2.bias)

    def forward(self, x, layer_past=None, attn_mask=None, head_mask=None, use_cache=False):
        """
        Run attention.

        Args:
            x (tensor): Input tensor.
            layer_past (Tuple): Tuple of past keys and values.
            attn_mask (tensor): Mask to avoid performing attention on padding token indices.
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules.
            use_cache (bool): If True, keys and values are returned (past_key_values).

        Returns:
            (tensor, Tuple): Output tensor, tuple of keys and values.
        """
        # ln = nn.Linear(x.shape[-1], 3 * self.embd_dim).to(x.device)
        x = self.ln(x)

        # query, key, value = torch.split(x, 3, dim=2)
        x_split_tuple = torch.split(x, 256, dim=2)
        query = x_split_tuple[0]
        key = x_split_tuple[1]
        value = x_split_tuple[2]
        # Input tensor, shape [B, seq_len, embd_dim] or [B, blocks, block_len, embd_dim].
        query = ops.split_heads(query, self.num_heads, self.head_dim)
        value = ops.split_heads(value, self.num_heads, self.head_dim)
        key = ops.split_heads(key, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = torch.tril(torch.ones((1, 1, self.max_pos, self.max_pos)))[:, :, key_len - query_len:key_len,
                      :key_len]
        # casual_mask = jnp.ones((1, 1, self.max_pos, self.max_pos))[:, :, key_len - query_len :key_len, :key_len]
        casual_mask = casual_mask.to(dtype=torch.bool, device=x.device)

        out, _attn_weights = ops.attention(query, key, value, casual_mask, -1e4, self.drop1, self.scale_attn_weights,
                                           attn_mask, head_mask)
        out = ops.merge_heads(out, self.num_heads, self.head_dim)

        # ln2 = nn.Linear(out.shape[-1], self.embd_dim).to(x.device)
        out = self.ln2(out)

        out = self.drop2(out)
        return out, present, _attn_weights


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_dim, n_embd, resid_pdrop, activation_function, init_index=0):
        super().__init__()

        self.intermediate_dim = intermediate_dim
        self.embd_dim = n_embd
        self.resid_dropout = resid_pdrop
        self.activation = activation_function

        self.fc1 = nn.Linear(self.embd_dim, self.intermediate_dim)
        self.fc2 = nn.Linear(self.intermediate_dim, self.embd_dim)
        self.dropout = nn.Dropout(self.resid_dropout)
        self.init_index = init_index

        self.initialize()

    def initialize(self):
        if len(self.fc1.weight.shape) < 2:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.fc1.weight.unsqueeze(0))
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.fc1.weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_uniform_(self.fc1.weight.unsqueeze(0))
        else:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.fc1.weight)
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.fc1.weight)
            else:
                torch.nn.init.kaiming_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        if len(self.fc2.weight.shape) < 2:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.fc2.weight.unsqueeze(0))
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.fc2.weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_uniform_(self.fc2.weight.unsqueeze(0))
        else:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.fc2.weight)
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.fc2.weight)
            else:
                torch.nn.init.kaiming_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        Run the MLP.

        Args:
            x (tensor): Input tensor.
        """
        x = self.fc1(x)
        x = ops.apply_activation(x, activation=self.activation)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, n_embd, layer_norm_epsilon, n_inner, n_positions, n_head, attn_pdrop, resid_pdrop,
                 activation_function, init_index=0):
        super().__init__()

        self.embd_dim = n_embd
        self.eps = layer_norm_epsilon
        self.inner_dim = n_inner if n_inner is not None else 4 * self.embd_dim
        self.n_positions = n_positions
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.activation = activation_function
        self.init_index = init_index

        self.ln_1 = nn.LayerNorm(n_embd, eps=self.eps)
        self.attn = GPT2SelfAttention(self.n_positions, self.embd_dim, self.n_head, self.attn_pdrop, self.resid_pdrop, init_index=self.init_index)
        self.ln_2 = nn.LayerNorm(n_embd, eps=self.eps)
        self.mlp = GPT2MLP(self.inner_dim, self.embd_dim, self.resid_pdrop, self.activation, init_index=self.init_index)

        self.initialize()

    def initialize(self):
        if len(self.ln_1.weight.shape) < 2:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln_1.weight.unsqueeze(0))
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln_1.weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_uniform_(self.ln_1.weight.unsqueeze(0))
        else:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln_1.weight)
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln_1.weight)
            else:
                torch.nn.init.kaiming_uniform_(self.ln_1.weight)
        # init.xavier_uniform_(self.ln.weight)
        init.zeros_(self.ln_1.bias)
        if len(self.ln_2.weight.shape) < 2:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln_2.weight.unsqueeze(0))
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln_2.weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_uniform_(self.ln_2.weight.unsqueeze(0))
        else:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln_2.weight)
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln_2.weight)
            else:
                torch.nn.init.kaiming_uniform_(self.ln_2.weight)
        init.zeros_(self.ln_2.bias)


    def forward(self, x, layer_past=None, attn_mask=None, head_mask=None, use_cache=False):
        """
        Run the block.

        Args:
            x (tensor): Input tensor.
            layer_past (Tuple): Tuple of past keys and values.
            attn_mask (tensor): Mask to avoid performing attention on padding token indices.
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules.
            use_cache (bool): If True, keys and values are returned (past_key_values).

        Returns:
            (tensor, Tuple): Output tensor, tuple of keys and values.
        """
        residual = x
        x = self.ln_1(x)
        kwargs = {'layer_past': layer_past, 'attn_mask': attn_mask, 'head_mask': head_mask,
                  'use_cache': use_cache}
        x, present, _attn_weights = self.attn(x, **kwargs)
        x += residual
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x += residual
        return x, present, _attn_weights


class GPT2Model(nn.Module):
    def __init__(self, vocab_size, n_positions, n_embd, embd_pdrop, n_layer, layer_norm_epsilon,
                 n_inner, n_head, attn_pdrop, resid_pdrop, activation_function, init_index=0):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_pos = n_positions
        self.embd_dim = n_embd
        self.embd_dropout = embd_pdrop
        self.num_layers = n_layer
        self.eps = layer_norm_epsilon
        self.n_inner = n_inner
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.activation_function = activation_function
        self.init_index = init_index

        self.drop = nn.Dropout(self.embd_dropout)
        self.ln_f = nn.LayerNorm(self.embd_dim, eps=self.eps)
        self.block = GPT2Block(self.embd_dim, self.eps, self.n_inner, self.max_pos, self.n_head,
                               self.attn_pdrop, self.resid_pdrop, self.activation_function, init_index=self.init_index)
        self.h = nn.ModuleList([self.block for _ in range(self.num_layers)])

        self.initialize()

    def initialize(self):
        if len(self.ln_f.weight.shape) < 2:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln_f.weight.unsqueeze(0))
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln_f.weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_uniform_(self.ln_f.weight.unsqueeze(0))
        else:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln_f.weight)
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln_f.weight)
            else:
                torch.nn.init.kaiming_uniform_(self.ln_f.weight)
        init.zeros_(self.ln_f.bias)

    def forward(self,
                input_ids=None,
                past_key_values=None,
                input_embds=None,
                position_ids=None,
                attn_mask=None,
                head_mask=None,
                use_cache=False
                ):
        """
        Run the model.

        Args:
            input_ids (tensor): Input token ids, shape [B, seq_len].
            past_key_values (Tuple): Precomputed hidden keys and values, tuple of tuples.
                                     If past_key_values is used, only input_ids that do not have their
                                     past calculated should be passed as input_ids.
            input_embds (tensor): Input embeddings, shape [B, seq_len, embd_dim].
            position_ids (tensor): Indices of positions of each input sequence tokens in the position embeddings, shape [B, seq_len].
            attn_mask (tensor): Mask to avoid performing attention on padding token indices, shape [B, seq_len].
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules, shape [num_heads] or [num_layers, num_heads].
            use_cache (bool): If True, keys and values are returned (past_key_values).

        Returns:
            (dict): Dictionary containing 'last_hidden_state', 'past_key_values'.
        """
        if input_ids is not None and input_embds is not None:
            raise ValueError('You cannot specify both input_ids and input_embd at the same time.')
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif input_embds is not None:
            input_shape = input_embds.shape[:-1]
            batch_size = input_embds.shape[0]
        else:
            raise ValueError('You have to specify either input_ids or input_embd.')

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.num_layers)
        else:
            past_length = past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0)

        if input_embds is None:
            input_embds = nn.Embedding(self.vocab_size, self.embd_dim)(input_ids)

        if attn_mask is not None:
            attn_mask = ops.get_attention_mask(attn_mask, batch_size)

        if head_mask is not None:
            head_mask = ops.get_head_mask(head_mask, self.num_layers)
        else:
            head_mask = [None] * self.num_layers

        # position_embds = nn.Embed(num_embeddings=self.max_pos, features=self.embd_dim)(position_ids)

        # x = input_embds + position_embds
        x = input_embds

        x = self.drop(x)
        output_shape = input_shape + (x.shape[-1],)

        presents = () if use_cache else None
        attn_weights_list = []
        for i in range(self.num_layers):
            kwargs = {'layer_past': past_key_values[i], 'attn_mask': attn_mask, 'head_mask': head_mask[i],
                      'use_cache': use_cache}
            x, present, attn_weights = self.h[i](x, **kwargs)

            if use_cache:
                presents = presents + (present,)
            attn_weights_list.append(attn_weights)

        x = self.ln_f(x)
        return {'last_hidden_state': x, 'past_key_values': presents, 'attn_weights_list': attn_weights_list}


class TransRewardModel(nn.Module):
    def __init__(self, observation_dim, action_dim, activation='gelu_new', activation_final='none', max_episode_steps=1001,
                 vocab_size=1, n_positions=1024, n_embd=256, embd_pdrop=0.1, n_layer=1, layer_norm_epsilon=1e-05,
                 n_inner=4, n_head=4, attn_pdrop=0.1, resid_pdrop=0.1, pref_attn_embd_dim=256, init_index=0):
        super().__init__()

        self.obs_dim = observation_dim
        self.act_dim = action_dim
        self.activation_function = activation
        self.activation_final = activation_final
        self.vocab_size = vocab_size
        self.max_pos = n_positions
        self.embd_dim = n_embd
        self.pref_attn_embd_dim = pref_attn_embd_dim
        self.embd_dropout = embd_pdrop
        self.attn_dropout = attn_pdrop
        self.resid_dropout = resid_pdrop
        self.num_layers = n_layer
        self.inner_dim = n_embd // 2
        self.eps = layer_norm_epsilon
        self.n_head = n_head
        self.mlp_inner = n_inner
        self.init_index = init_index

        self.obs_fc = nn.Linear(self.obs_dim, self.embd_dim)
        self.act_fc = nn.Linear(self.act_dim, self.embd_dim)
        self.emb_t = nn.Embedding(num_embeddings=max_episode_steps + 1, embedding_dim=self.embd_dim)
        self.ln_f = nn.LayerNorm(self.embd_dim, eps=self.eps)
        self.gpt2 = GPT2Model(self.vocab_size, self.max_pos, self.embd_dim, self.embd_dropout, self.num_layers,
                              self.eps, self.mlp_inner, self.n_head, self.attn_dropout, self.resid_dropout,
                              self.activation_function, init_index=self.init_index)

        self.ln = nn.Linear(self.embd_dim, self.inner_dim)
        self.ln2 = nn.Linear(self.inner_dim, 1)
        self.ln_w = nn.Linear(self.embd_dim, 2 * self.pref_attn_embd_dim + 1)

        self.initialize()

    def initialize(self):
        if len(self.ln.weight.shape) < 2:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln.weight.unsqueeze(0))
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln.weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_uniform_(self.ln.weight.unsqueeze(0))
        else:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln.weight)
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln.weight)
            else:
                torch.nn.init.kaiming_uniform_(self.ln.weight)

        init.zeros_(self.ln.bias)

        if len(self.ln2.weight.shape) < 2:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln2.weight.unsqueeze(0))
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln2.weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_uniform_(self.ln2.weight.unsqueeze(0))
        else:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln2.weight)
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln2.weight)
            else:
                torch.nn.init.kaiming_uniform_(self.ln2.weight)
        init.zeros_(self.ln2.bias)

        if len(self.ln_w.weight.shape) < 2:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln_w.weight.unsqueeze(0))
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln_w.weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_uniform_(self.ln_w.weight.unsqueeze(0))
        else:
            if self.init_index == 0:
                torch.nn.init.kaiming_normal_(self.ln_w.weight)
            elif self.init_index == 1:
                torch.nn.init.xavier_normal_(self.ln_w.weight)
            else:
                torch.nn.init.kaiming_uniform_(self.ln_w.weight)
        init.zeros_(self.ln_w.bias)

    def forward(
            self,
            states,
            actions,
            timesteps,
            attn_mask=None,
            reverse=False,
            use_weighted_sum=False,
            target_idx=1
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attn_mask is None:
            attn_mask = torch.ones((batch_size, seq_length), dtype=torch.float32, device=states.device)

        embd_state = self.obs_fc(states)
        embd_action = self.act_fc(actions)
        embd_timestep = self.emb_t(timesteps.int())

        embd_state = embd_state + embd_timestep
        embd_action = embd_action + embd_timestep

        if reverse:
            stacked_inputs = torch.stack(
                [embd_state, embd_action],
                dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 2 * seq_length, self.embd_dim)
        else:
            stacked_inputs = torch.stack(
                [embd_action, embd_state],
                dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 2 * seq_length, self.embd_dim)


        stacked_inputs = self.ln_f(stacked_inputs) # (128, 200, 256)

        stacked_attn_mask = torch.stack(
            [attn_mask, attn_mask],
            dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2 * seq_length)

        transformer_outputs = self.gpt2(
            input_embds=stacked_inputs,
            attn_mask=stacked_attn_mask
        )

        x = transformer_outputs["last_hidden_state"] # sa: (128,200,256)
        attn_weights_list = transformer_outputs["attn_weights_list"]    #sa: (128,4,200,200)
        x = x.reshape(batch_size, seq_length, 2, self.embd_dim).permute(0, 2, 1, 3)
        hidden_output = x[:, target_idx]  #sa: x(128,2,100,256)
        # hidden: sa (128,100,256)
        # hidden_output = x
        if use_weighted_sum:
            '''
            add additional Attention Layer for Weighted Sum.
            x (= output, tensor): Predicted Reward, shape [B, seq_len, embd_dim]    (b, h, t)
            '''
            # ln = nn.Linear(hidden_output.shape[-1], 2 * self.pref_attn_embd_dim + 1).to(hidden_output.device)
            x = self.ln_w(hidden_output)
            # only one head, because value has 1 dim for predicting rewards directly.
            num_heads = 1

            # query: [B, seq_len, embd_dim]
            # key: [B, seq_len, embd_dim]
            # value: [B, seq_len, 1]
            # query, key, value = torch.split(x, [self.pref_attn_embd_dim, self.pref_attn_embd_dim * 2], dim=2)

            x_split_tuple = torch.split(x, [self.pref_attn_embd_dim, self.pref_attn_embd_dim, 1], dim=2)
            query = x_split_tuple[0]
            key = x_split_tuple[1]
            value = x_split_tuple[2]

            query = ops.split_heads(query, num_heads, self.pref_attn_embd_dim)
            key = ops.split_heads(key, num_heads, self.pref_attn_embd_dim)
            value = ops.split_heads(value, num_heads, 1)

            # query: [B, 1, seq_len, embd_dim]
            # key: [B, 1, seq_len, embd_dim]
            # value: [B, 1, seq_len, 1]

            query_len, key_len = query.shape[-2], key.shape[-2]
            # casual_mask = jnp.tril(jnp.ones((1, 1, self.config_.n_positions, self.config_.n_positions)))[:, :, key_len - query_len :key_len, :key_len]
            # casual_mask = casual_mask.astype(bool)
            casual_mask = torch.ones((1, 1, seq_length, seq_length))[:, :, key_len - query_len:key_len, :key_len]
            casual_mask = casual_mask.to(dtype=torch.bool)

            # attn_dropout = nn.Dropout(rate=self.attn_dropout) # split dropout rate
            attn_dropout = nn.Dropout(0.0).to(hidden_output.device)  # boilerplate code.
            new_attn_mask = ops.get_attention_mask(attn_mask, batch_size)

            out, last_attn_weights = ops.attention(query, key, value, casual_mask, -1e-4, attn_dropout,
                                                   scale_attn_weights=True, attn_mask=new_attn_mask,
                                                   head_mask=None)
            attn_weights_list.append(last_attn_weights)
            # out: [B, 1, seq_len, 1]
            output = ops.merge_heads(out, num_heads, 1)
            # output: [B, seq_len, 1]

            # output = nn.Dropout(rate=self.resid_dropout)(out)
            return {"weighted_sum": output, "value": value}, attn_weights_list

        else:
            # ln = nn.Linear(hidden_output.shape[-1], self.inner_dim).to(hidden_output.device)
            x = self.ln(hidden_output)                                                      # batch, seq_len, embd_dim / 2
            x = ops.apply_activation(x, activation=self.activation_function)                
            # ln2 = nn.Linear(x.shape[-1], 1).to(hidden_output.device)
            output = self.ln2(x)                                                            # batch, seq_len, 1
            if self.activation_final != 'none':
                output = ops.apply_activation(output, activation=self.activation_final)

            return {"value": output}, attn_weights_list

def soft_cross_entropy(preds, soft_targets):
    """
    计算Soft Cross Entropy
    :param preds: 预测值，大小为 [batch_size, num_classes]
    :param soft_targets: 软目标（概率分布），大小为 [batch_size, num_classes]
    :return: 损失值
    """
    return torch.mean(torch.sum(- soft_targets * F.log_softmax(preds, dim=1), 1))


class HLGaussLoss(nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = torch.linspace(
            min_value, max_value, num_bins + 1, dtype=torch.float32
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=logits.device)
        for i in range(logits.shape[1]):
            soft_target = self.transform_to_probs(target[:, i])
            # loss += F.cross_entropy(logits[:, i].squeeze(), mid[:, i].squeeze())
            loss += soft_cross_entropy(logits[:, i], soft_target)
        return loss

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        cdf_evals = torch.special.erf(
            (self.support.to(target.device) - target.unsqueeze(-1))
            / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers.to(probs.device), dim=-1)