import os
import math
import torch 
import torch.nn as nn

import torch.nn.functional as F

class make_node_pooling(nn.Module):
    def __init__(self, input_feature, output_node):
        super(make_node_pooling, self).__init__()
        self.output_node = output_node
        self.input_feature = input_feature
        self.embedding_node = nn.Linear(input_feature, output_node)

    def forward(self, V, A):
        # output (1,2708,output_node)
        output = self.embedding_node(V)
        # output (1,2708,no_vertices) softmax thorugh 2708
        output = F.softmax(output, dim=1)
        # (1,no_vertices,2708) * (1,2708,input_feature) = (1,no_vertices,input_feature)
        output_V = torch.bmm(output.permute(0, 2, 1), V)
        if self.output_node == 1:
            ## (1,no_vertices,input_feature) - (1*no_vertices,input_feature) where no_vertices == 1
            output_V, output_A = output_V.reshape(-1, self.input_feature), A
            return output_V, output_A
        # (1, 2708, 4, 2708) - (1,2708*4,2708)
        output_A = A.reshape(A.shape[0], -1, A.shape[1])
        # (1,2708*2,2708)* (1,2708,no_vertices) = (1,2708*2,no_vertices)
        output_A = torch.bmm(output_A, output)
        # (1,2708*2,no_vertices) - (1,2708,no_vertices*2)
        output_A = output_A.reshape(A.shape[0], A.shape[1], -1)
        # (1,no_vertices,2708)*(1,2708,no_vertices*2) =(1,no_vertices,no_vertices*2)
        output_A = torch.bmm(output.permute(0, 2, 1), output_A)
        # (1,no_vertices,no_vertices*2) - (1,no_vertices,2,no_vertices)
        output_A = output_A.reshape(A.shape[0], self.output_node, A.shape[2],
                                    self.output_node)

        return output_V, output_A

class make_node_attention_output(nn.Module):
    def __init__(self, input_feature, output_node):
        super(make_node_attention_output, self).__init__()
        self.output_node = output_node
        self.input_feature = input_feature
        self.embedding_node = nn.Linear(input_feature, output_node)

    def forward(self, V):
        # output (1,2708,output_node)
        output = self.embedding_node(V)
        # output (1,2708,no_vertices) softmax thorugh 2708
        output = F.softmax(output, dim=1)
        # (1,no_vertices,2708) * (1,2708,input_feature) = (1,no_vertices,input_feature)
        output_V = torch.bmm(output.permute(0, 2, 1), V)
        ## (1,no_vertices,input_feature) - (1*no_vertices,input_feature) where no_vertices == 1
        output_V = output_V.reshape(-1, self.input_feature)
        return output_V


class make_Embedding(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.3,
                 is_output=False):
        super(make_Embedding, self).__init__()

        self.is_output = is_output
        if not is_output:
            self.embedding = torch.nn.Sequential(
                nn.Linear(in_features, out_features, bias),
                nn.Dropout(dropout),
            )
            self.bn_ac = torch.nn.Sequential(nn.BatchNorm1d(out_features),
                                             nn.LeakyReLU(inplace=False))
        else:
            self.embedding = nn.Linear(in_features, out_features, bias)

    def forward(self, V):
        output = self.embedding(V)
        output = self.bn_ac(output)
        return output


class make_node_Embedding(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.3,
                 is_output=False):
        super(make_node_Embedding, self).__init__()

        self.is_output = is_output
        if not is_output:
            self.embedding = torch.nn.Sequential(
                nn.Linear(in_features, out_features, bias),
                nn.Dropout(dropout),
            )
            self.bn_ac = torch.nn.Sequential(nn.BatchNorm1d(out_features),
                                             nn.LeakyReLU(inplace=False))
        else:
            self.embedding = nn.Linear(in_features, out_features, bias)

    def forward(self, V, A):
        if self.is_output:
            output = self.embedding(V)

        else:
            output = self.bn_ac(self.embedding(V).squeeze(0)).unsqueeze(0)
        return output, A



class make_dense_gcn_layer(nn.Module):
    def __init__(self, input_feature, no_A, repeat_time):
        super(make_dense_gcn_layer, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(repeat_time):
            self.layers.append(
                make_GraphAttentionLayer(no_A, input_feature * (repeat + 1),
                                         input_feature, 0.1))
        self.squeeze_block = make_GraphAttentionLayer(
            no_A, input_feature * (repeat_time + 1), input_feature, 0.1)

    def forward(self, V, A):
        input_V = V
        for layer in self.layers:
            update_V = layer(input_V, A)[0]
            input_V = torch.cat((input_V, update_V), dim=-1)
        output = self.squeeze_block(input_V, A)[0]
        return output, A

class make_Parameter_W(nn.Module):
    def __init__(self, in_features, out_features):
        super(make_Parameter_W, self).__init__()
        self.parameter = nn.Parameter(
            torch.zeros(size=(in_features, out_features)))
        torch.nn.init.xavier_uniform_(self.parameter)

    def forward(self):
        return self.parameter

class make_Parameter_a(nn.Module):
    def __init__(self, out_features):
        super(make_Parameter_a, self).__init__()
        self.parameter = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        torch.nn.init.xavier_uniform_(self.parameter)

    def forward(self):
        return self.parameter


class make_GraphAttentionLayer(nn.Module):
    def __init__(self,
                 no_A,
                 in_features,
                 out_features,
                 dropout,
                 xavier_uniform=True,
                 debug=False):
        super(make_GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.xavier_uniform = xavier_uniform
        self.no_A = no_A
        self.debug = debug

        self.W = nn.ModuleList()
        for _ in range(no_A + 1):
            self.W.append(make_Parameter_W(self.in_features,
                                           self.out_features))

        self.a = nn.ModuleList()
        for _ in range(no_A + 1):
            self.a.append(make_Parameter_a((self.out_features // 8)))

        self.embedding = nn.ModuleList()
        for _ in range(no_A + 1):
            self.embedding.append(
                make_Parameter_W(self.out_features, self.out_features // 8))

        self.squeeze = nn.Linear(self.out_features * (no_A + 1), out_features,
                                 True)

        self.leakyrelu = nn.LeakyReLU(inplace=False)

    def forward(self, V, adj):

        N = V.size()[1]
        b = V.size()[0]
        output = torch.zeros(size=(b, N, self.out_features))
        for num_A in range(self.no_A + 1):

            h = torch.matmul(V, self.W[num_A]())
            #output = (1, N, N, out_features*2)
            #out_features*2 = self_out_features+ other_out_features

            # 1,100,100,out_features*2 * out_features*2,1
            h_embedding = torch.matmul(h, self.embedding[num_A]())

            a_input = torch.cat([
                h_embedding.repeat(1, 1, N).view(1, N * N, -1),
                h_embedding.repeat(1, N, 1)
            ],
                                dim=1).view(1, N, N, -1)

            # (1,100,100,out_features*2) -> (1,100,100,1)  -> (1,100,100)
            e = self.leakyrelu(
                torch.matmul(a_input, self.a[num_A]()).squeeze(-1))
            if num_A < self.no_A:
                num_adj = adj[:, :, num_A, :]
            else:
                num_adj = torch.eye(N).unsqueeze(0).to(e.device)

            zero_vec = -9e15 * torch.ones_like(e)

            attention = torch.where(num_adj > 0, e, zero_vec)

            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention,
                                  self.dropout,
                                  training=self.training)
            h_prime = torch.matmul(attention, h)

            if num_A == 0:
                output = h_prime
            else:
                output = torch.cat((output, h_prime), -1)

        output = self.squeeze(output)
        output = self.leakyrelu(output)

        if self.in_features == self.out_features:
            output += V

        if self.debug:
            print('batch_size: {}'.format(b))
            print('vertice numbers: {}'.format(N))
            print('heatmap_size: {}'.format(h.size()))
            print('heatmap_concat_size: {}'.format(a_input.size()))
            print('heatmap_transform_size: {}'.format(e.size()))
            print('single_adj_size: {}'.format(num_adj.size()))
            print('attention_map size: {}'.format(attention.size()))
            print('output_size: {}'.format(output.size()))
            print(
                'number_adjacency_matrix_with_identity: {}'.format(self.no_A +
                                                                   1))

        return output, adj

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.in_features) + ' -> ' + str(self.out_features) + ')'

class Temperature(nn.Module):
    def __init__(self, T):
        super(Temperature, self).__init__()
        self.T = T

    def forward(self, x):
        return x / self.T
    
class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Attention_GCN_backbone(nn.Module):
    def __init__(self, input_feature, no_A, class_formal_key=7, key_index=None, num_classification=3):
        super(Attention_GCN_backbone, self).__init__()
        self.class_formal_key = class_formal_key
        self.fullflow = mySequential(
            make_node_Embedding(input_feature, 128),
            make_dense_gcn_layer(128, no_A, 3),
            make_GraphAttentionLayer(no_A, 128, 128, 0.3),
            make_node_Embedding(128, 64),
            make_node_Embedding(64, 32),
        )

        self.formal_key = make_node_Embedding(32,
                                              class_formal_key,
                                              is_output=True)
        self.final_output = make_node_Embedding(32,
                                                class_formal_key * 2 - 1,
                                                is_output=True)

        self.classification_output = make_node_Embedding(32,
                                                num_classification,
                                                is_output=True)

        # self.class_weight = nn.Linear(class_formal_key * 2 - 1, num_classification)
        self.class_weight = nn.Parameter(torch.FloatTensor(class_formal_key * 2 - 1))

        # nn.init.zeros_(self.class_weight)
        # bound = 1 / math.sqrt(self.class_weight.shape[0])
        # nn.init.uniform_(self.class_weight, -1e-2, 1e-2)
        # nn.init.zeros_(self.class_weight)
        self.class_weight.data.fill_((1 - 0.7) / (len(self.class_weight) - 2))

        assert "patient_status" in key_index and "date" in key_index

        # print(key_index["patient_status"], key_index["date"])

        self.class_weight.data[key_index["patient_status"] * 2] = 0.5
        self.class_weight.data[key_index["date"] * 2] = 0.2

        self.class_weight.requires_grad = False

        self.classification_output = nn.Linear(input_feature, num_classification)

    def forward(self, input_image, V, A, labels=None):

        output, A = self.fullflow(V, A)
        output_formal = self.formal_key(output, A)[0]
        output_final = self.final_output(output, A)[0]
        
        if labels is not None:
            final_class = labels
        else:
            final_class = torch.max(output_final, -1)[1]

        # weights = F.softmax(self.class_weight, -1)
        weights = self.class_weight
        weights = weights[final_class]

        output = V.permute([0, 2, 1])

        weights = (weights / weights.sum(-1)).unsqueeze(-1)

        output = output.bmm(weights).squeeze(-1)

        output_classification = self.classification_output(output)

        return output_formal, output_final, output_classification

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(
                torch.load(base_file,
                           map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


class Jeff_Attention_GCN_backbone(nn.Module):
    def __init__(self, input_feature, no_A, class_formal_key=7, key_index=None, num_classification=3):
        super(Jeff_Attention_GCN_backbone, self).__init__()
        self.class_formal_key = class_formal_key
        self.fullflow = mySequential(
            make_node_Embedding(input_feature, 128),
            make_dense_gcn_layer(128, no_A, 3),
            make_GraphAttentionLayer(no_A, 128, 128, 0.3),
            make_node_Embedding(128, 64),
            make_node_Embedding(64, 32),
        )

        self.formal_key = make_node_Embedding(32,
                                              class_formal_key,
                                              is_output=True)
        self.final_output = make_node_Embedding(32,
                                                class_formal_key * 2 - 1,
                                                is_output=True)

        self.classification_gnn = make_node_attention_output(32, 1)

        self.classification_output = nn.Linear(32, num_classification)

    def forward(self, input_image, V, A, labels=None):

        output, A = self.fullflow(V, A)
        output_formal = self.formal_key(output, A)[0]
        output_final = self.final_output(output, A)[0]

        output_classification = self.classification_gnn(output)
        output_classification = self.classification_output(output_classification)

        return output_formal, output_final, output_classification

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(
                torch.load(base_file,
                           map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

class SelfAttention_GCN_backbone(nn.Module):
    def __init__(self, input_feature, no_A, class_formal_key=7, key_index=None, num_classification=3):
        super(SelfAttention_GCN_backbone, self).__init__()
        self.class_formal_key = class_formal_key
        self.fullflow = mySequential(
            make_node_Embedding(input_feature, 128),
            make_dense_gcn_layer(128, no_A, 3),
            make_GraphAttentionLayer(no_A, 128, 128, 0.3),
            make_node_Embedding(128, 64),
            make_node_Embedding(64, 32),
        )

        self.formal_key = make_node_Embedding(32,
                                              class_formal_key,
                                              is_output=True)
        self.final_output = make_node_Embedding(32,
                                                class_formal_key * 2 - 1,
                                                is_output=True)

        self.classification_output = make_node_Embedding(32,
                                                num_classification,
                                                is_output=True)

        # self.class_weight = nn.Linear(class_formal_key * 2 - 1, num_classification)
        self.classification_weight = nn.Sequential(
                                                nn.Linear(32, class_formal_key * 2 - 1),
                                                Temperature(0.1),
                                                nn.Softmax(dim=-1)
                                                )

        self.classification_output = nn.Linear(input_feature, num_classification)

    def forward(self, input_image, V, A, labels=None):

        output, A = self.fullflow(V, A)
        output_formal = self.formal_key(output, A)[0]
        output_final = self.final_output(output, A)[0]
        
        if labels is not None:
            final_class = labels.unsqueeze(-1)
        else:
            final_class = torch.max(output_final, -1)[1].unsqueeze(-1)

        # weights = F.softmax(self.class_weight, -1)
        weights = self.classification_weight(output)

        weights = weights.gather(-1, final_class).squeeze(-1)

        output = V.permute([0, 2, 1])

        weights = (weights / weights.sum(-1)).unsqueeze(-1)

        output = output.bmm(weights).squeeze(-1)

        output_classification = self.classification_output(output)

        return output_formal, output_final, output_classification

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(
                torch.load(base_file,
                           map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


class SelfAttention_GCN_classifier(nn.Module):
    def __init__(self, input_feature, no_A, class_formal_key=7, key_index=None, num_classification=3):
        super(SelfAttention_GCN_classifier, self).__init__()
        self.class_formal_key = class_formal_key
        self.fullflow = mySequential(
            make_node_Embedding(input_feature, 128),
            make_dense_gcn_layer(128, no_A, 3),
            make_GraphAttentionLayer(no_A, 128, 128, 0.3),
            make_node_Embedding(128, 64),
            make_node_Embedding(64, 32),
        )

        self.formal_key = make_node_Embedding(32,
                                              class_formal_key,
                                              is_output=True)
        self.final_output = make_node_Embedding(32,
                                                class_formal_key * 2 - 1,
                                                is_output=True)

        self.classification_output = make_node_Embedding(32,
                                                num_classification,
                                                is_output=True)

        # self.class_weight = nn.Linear(class_formal_key * 2 - 1, num_classification)
        self.classification_weight = nn.Sequential(
                                                nn.Linear(32, 1)
                                                )

        self.classification_output = nn.Linear(input_feature, num_classification)

    def forward(self, input_image, V, A, labels=None):

        output, A = self.fullflow(V, A)
        output_formal = self.formal_key(output, A)[0]
        output_final = self.final_output(output, A)[0]

        # weights = F.softmax(self.class_weight, -1)
        weights = self.classification_weight(output).squeeze(-1)

        weights = F.softmax(weights, -1)

        output = V.permute([0, 2, 1])

        weights = (weights / weights.sum(-1)).unsqueeze(-1)

        output = output.bmm(weights).squeeze(-1)

        output_classification = self.classification_output(output)

        return output_formal, output_final, output_classification

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(
                torch.load(base_file,
                           map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

class MultiAttention_GCN_backbone(nn.Module):
    def __init__(self, input_feature, no_A, class_formal_key=7, key_index=None, num_classification=3):
        super(MultiAttention_GCN_backbone, self).__init__()
        self.class_formal_key = class_formal_key
        self.fullflow = mySequential(
            make_node_Embedding(input_feature, 128),
            make_dense_gcn_layer(128, no_A, 3),
            make_GraphAttentionLayer(no_A, 128, 128, 0.3),
            make_node_Embedding(128, 64),
            make_node_Embedding(64, 32),
        )

        self.formal_key = make_node_Embedding(32,
                                              class_formal_key,
                                              is_output=True)
        self.final_output = make_node_Embedding(32,
                                                class_formal_key * 2 - 1,
                                                is_output=True)

        self.classification_output = make_node_Embedding(32,
                                                num_classification,
                                                is_output=True)

        # self.class_weight = nn.Linear(class_formal_key * 2 - 1, num_classification)
        self.class_weight = nn.Parameter(torch.FloatTensor(class_formal_key * 2 - 1))

        # nn.init.zeros_(self.class_weight)
        bound = 1 / math.sqrt(self.class_weight.shape[0])
        # nn.init.uniform_(self.class_weight, -1e-2, 1e-2)
        nn.init.zeros_(self.class_weight)
        self.class_weight.data.fill_((1 - 0.3) / (len(self.class_weight) - 2))

        assert "patient_status" in key_index and "date" in key_index

        # print(key_index["patient_status"], key_index["date"])

        self.class_weight.data[key_index["patient_status"] * 2] = 0.2
        self.class_weight.data[key_index["date"] * 2] = 0.1

        self.class_weight.requires_grad = False

        self.classification_output = nn.Linear(input_feature, num_classification)

        self.classification_weight = nn.Sequential(
                                                nn.Linear(32, 1)
                                                )

    def forward(self, input_image, V, A, labels=None):

        output, A = self.fullflow(V, A)
        output_formal = self.formal_key(output, A)[0]
        output_final = self.final_output(output, A)[0]
        
        if labels is not None:
            final_class = labels
        else:
            final_class = torch.max(output_final, -1)[1]

        # class weight
        weights = self.class_weight
        weights = weights[final_class]

        V = V.permute([0, 2, 1])

        weights = (weights / weights.sum(-1)).unsqueeze(-1)
        class_weighted_output = V.bmm(weights).squeeze(-1)

        # self attention
        weights = self.classification_weight(output).squeeze(-1)
        weights = F.softmax(weights, -1).unsqueeze(-1)

        self_weighted_output = V.bmm(weights).squeeze(-1)

        output_classification = self.classification_output(class_weighted_output + self_weighted_output)

        return output_formal, output_final, output_classification

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(
                torch.load(base_file,
                           map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')