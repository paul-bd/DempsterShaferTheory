import torch
import numpy as np

class Distance_layer(torch.nn.Module):
    '''
    verified
    '''
    def __init__(self, n_prototypes, n_feature_maps):
        super(Distance_layer, self).__init__()
        self.w = torch.nn.Linear(in_features=n_feature_maps, out_features=n_prototypes, bias=False).weight
        self.n_prototypes = n_prototypes

    def forward(self, inputs):
        for i in range(self.n_prototypes):
            if i == 0:
                un_mass_i = (self.w[i, :] - inputs) ** 2
                un_mass_i = torch.sum(un_mass_i, dim=-1, keepdim=True)
                un_mass = un_mass_i

            if i >= 1:
                un_mass_i = (self.w[i, :] - inputs) ** 2
                un_mass_i = torch.sum(un_mass_i, dim=-1, keepdim=True)
                un_mass = torch.cat([un_mass, un_mass_i], -1)
        return un_mass


class DistanceActivation_layer(torch.nn.Module):
    '''
    verified
    '''
    def __init__(self, n_prototypes,init_alpha=0,init_gamma=0.1):
        super(DistanceActivation_layer, self).__init__()
        self.eta = torch.nn.Linear(in_features=n_prototypes, out_features=1, bias=False)#.weight.data.fill_(torch.from_numpy(np.array(init_gamma)).to(device))
        self.xi = torch.nn.Linear(in_features=n_prototypes, out_features=1, bias=False)#.weight.data.fill_(torch.from_numpy(np.array(init_alpha)).to(device))
        #torch.nn.init.kaiming_uniform_(self.eta.weight)
        #torch.nn.init.kaiming_uniform_(self.xi.weight)
        torch.nn.init.constant_(self.eta.weight,init_gamma)
        torch.nn.init.constant_(self.xi.weight,init_alpha)
        #self.alpha_test = 1/(torch.exp(-self.xi.weight)+1)
        self.n_prototypes = n_prototypes
        self.alpha = None

    def forward(self, inputs):
        gamma=torch.square(self.eta.weight)
        alpha=torch.neg(self.xi.weight)
        alpha=torch.exp(alpha)+1
        alpha=torch.div(1, alpha)
        self.alpha=alpha
        si=torch.mul(gamma, inputs)
        si=torch.neg(si)
        si=torch.exp(si)
        si = torch.mul(si, alpha)
        max_val, max_idx = torch.max(si, dim=-1, keepdim=True)
        si /= (max_val + 0.0001)

        return si


'''class Belief_layer(torch.nn.Module):
    def __init__(self, prototypes, num_class):
        super(DS2, self).__init__()
        self.beta = torch.nn.Linear(in_features=prototypes, out_features=num_class, bias=False).weight
        self.prototypes = prototypes
        self.num_class = num_class

    def forward(self, inputs):
        beta = torch.square(self.beta)
        beta_sum = torch.sum(beta, dim=0, keepdim=True)
        self.u = torch.div(beta, beta_sum)
        inputs_new = torch.unsqueeze(inputs, dim=-2)
        for i in range(self.prototypes):
            if i == 0:
                mass_prototype_i = torch.mul(self.u[:, i], inputs_new[..., i])  #batch_size * n_class
                mass_prototype = torch.unsqueeze(mass_prototype_i, -2)
            if i > 0:
                mass_prototype_i = torch.unsqueeze(torch.mul(self.u[:, i], inputs_new[..., i]), -2)
                mass_prototype = torch.cat([mass_prototype, mass_prototype_i], -2)
        return mass_prototype'''

class Belief_layer(torch.nn.Module):
    '''
    verified
    '''
    def __init__(self, n_prototypes, num_class):
        super(Belief_layer, self).__init__()
        self.beta = torch.nn.Linear(in_features=n_prototypes, out_features=num_class, bias=False).weight
        self.num_class = num_class
    def forward(self, inputs):
        beta = torch.square(self.beta)
        beta_sum = torch.sum(beta, dim=0, keepdim=True)
        u = torch.div(beta, beta_sum)
        mass_prototype = torch.einsum('cp,b...p->b...pc',u, inputs)
        return mass_prototype

class Omega_layer(torch.nn.Module):
    '''
    verified, give same results

    '''
    def __init__(self, n_prototypes, num_class):
        super(Omega_layer, self).__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs):
        mass_omega_sum = 1 - torch.sum(inputs, -1, keepdim=True)
        #mass_omega_sum = 1. - mass_omega_sum[..., 0]
        #mass_omega_sum = torch.unsqueeze(mass_omega_sum, -1)
        mass_with_omega = torch.cat([inputs, mass_omega_sum], -1)
        return mass_with_omega

class Dempster_layer(torch.nn.Module):
    '''
    verified give same results

    '''
    def __init__(self, n_prototypes, num_class):
        super(Dempster_layer, self).__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs):
        m1 = inputs[..., 0, :]
        omega1 = torch.unsqueeze(inputs[..., 0, -1], -1)
        for i in range(self.n_prototypes - 1):
            m2 = inputs[..., (i + 1), :]
            omega2 = torch.unsqueeze(inputs[..., (i + 1), -1], -1)
            combine1 = torch.mul(m1, m2)
            combine2 = torch.mul(m1, omega2)
            combine3 = torch.mul(omega1, m2)
            combine1_2 = combine1 + combine2
            combine2_3 = combine1_2 + combine3
            combine2_3 = combine2_3 / torch.sum(combine2_3, dim=-1, keepdim=True)
            m1 = combine2_3
            omega1 = torch.unsqueeze(combine2_3[..., -1], -1)
        return m1


class DempsterNormalize_layer(torch.nn.Module):
    '''
    verified

    '''
    def __init__(self):
        super(DempsterNormalize_layer, self).__init__()
    def forward(self, inputs):
        mass_combine_normalize = inputs / torch.sum(inputs, dim=-1, keepdim=True)
        return mass_combine_normalize


class Dempster_Shafer_module(torch.nn.Module):
    def __init__(self, n_feature_maps, n_classes, n_prototypes):
        super(Dempster_Shafer_module, self).__init__()
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.n_feature_maps = n_feature_maps
        self.ds1 = Distance_layer(n_prototypes=self.n_prototypes, n_feature_maps=self.n_feature_maps)
        self.ds1_activate = DistanceActivation_layer(n_prototypes = self.n_prototypes)
        self.ds2 = Belief_layer(n_prototypes= self.n_prototypes, num_class=self.n_classes)
        self.ds2_omega = Omega_layer(n_prototypes= self.n_prototypes,num_class= self.n_classes)
        self.ds3_dempster = Dempster_layer(n_prototypes= self.n_prototypes,num_class= self.n_classes)
        self.ds3_normalize = DempsterNormalize_layer()

    def forward(self, inputs):
        '''
        '''
        ED = self.ds1(inputs)
        ED_ac = self.ds1_activate(ED)
        mass_prototypes = self.ds2(ED_ac)
        mass_prototypes_omega = self.ds2_omega(mass_prototypes)
        mass_Dempster = self.ds3_dempster(mass_prototypes_omega)
        mass_Dempster_normalize = self.ds3_normalize(mass_Dempster)
        return mass_Dempster_normalize



def tile(a, dim, n_tile, device):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        device)
    return torch.index_select(a, dim, order_index)


class DM(torch.nn.Module):
    def __init__(self, num_class, nu=0.9, device=torch.device('cpu')):
        super(DM, self).__init__()
        self.nu = nu
        self.num_class = num_class
        self.device = device

    def forward(self, inputs):
        upper = torch.unsqueeze((1 - self.nu) * inputs[..., -1], -1)  # here 0.1 = 1 - \nu
        upper = tile(upper, dim=-1, n_tile=self.num_class + 1, device=self.device)
        outputs = (inputs + upper)[..., :-1]
        return outputs
