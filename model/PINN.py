import torch
import torch.nn as nn


class PINNNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        n_input = 4
        n_output = 3
        n_nodes = 30

        self.hidden_layer1 = nn.Linear(n_input,n_nodes)
        nn.init.xavier_uniform_(self.hidden_layer1.weight)
        nn.init.normal_(self.hidden_layer1.bias)

        self.hidden_layer2 = nn.Linear(n_nodes,n_nodes)
        nn.init.xavier_uniform_(self.hidden_layer2.weight)
        nn.init.normal_(self.hidden_layer2.bias)

        self.hidden_layer3 = nn.Linear(n_nodes,n_nodes)
        nn.init.xavier_uniform_(self.hidden_layer3.weight)
        nn.init.normal_(self.hidden_layer3.bias)

        self.hidden_layer4 = nn.Linear(n_nodes,n_nodes)
        nn.init.xavier_uniform_(self.hidden_layer4.weight)
        nn.init.normal_(self.hidden_layer4.bias)

        self.hidden_layer5 = nn.Linear(n_nodes,n_nodes)
        nn.init.xavier_uniform_(self.hidden_layer5.weight)
        nn.init.normal_(self.hidden_layer5.bias)

        self.hidden_layer6 = nn.Linear(n_nodes,n_nodes)
        nn.init.xavier_uniform_(self.hidden_layer6.weight)
        nn.init.normal_(self.hidden_layer6.bias)

        self.output_layer = nn.Linear(n_nodes, n_output)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.normal_(self.output_layer.bias)

    def forward(self, x,y,z,t):

        inputs = torch.cat([x,y,z,t],axis=1)

        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        layer5_out = torch.tanh(self.hidden_layer5(layer4_out))
        layer6_out = torch.tanh(self.hidden_layer6(layer5_out))

        output = self.output_layer(layer6_out) ## For regression, no activation is used in output layer

        return output

    def pinnLoss(self, x,y,z,t, mse):

        logits = self.forward(x,y,z,t)
        u = logits[:, 0].reshape(-1,1)
        v = logits[:, 1].reshape(-1,1)
        w = logits[:, 2].reshape(-1,1)
        
        
        # E = 1
        # nu = 0.3
        
        # # 拉梅常数
        # lmbd = E * nu/((1+nu)*(1-2*nu))
        # mu = E/(2*(1+nu))


        # u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

        # u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        # u_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]

        # v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        # v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        # v_z = torch.autograd.grad(v.sum(), z, create_graph=True)[0]

        # w_x = torch.autograd.grad(w.sum(), x, create_graph=True)[0]
        # w_y = torch.autograd.grad(w.sum(), y, create_graph=True)[0]
        # w_z = torch.autograd.grad(w.sum(), z, create_graph=True)[0]

        # exx = u_x
        # eyy = v_y
        # ezz = w_z

        # exy = 1/2*(u_y + v_x)
        # eyz = 1/2*(v_z + w_y)
        # ezx = 1/2*(w_x + u_z)

        # sxx = (lmbd+2*mu)*exx + lmbd*eyy + lmbd*ezz
        # syy = (lmbd+2*mu)*eyy + lmbd*exx + lmbd*ezz
        # szz = (lmbd+2*mu)*ezz + lmbd*exx + lmbd*eyy
        # sxy = 2*mu*exy
        # syz = 2*mu*eyz
        # sxz = 2*mu*ezx

        # sxx_x = torch.autograd.grad(sxx.sum(), x, create_graph=True)[0]
        # sxy_y = torch.autograd.grad(sxy.sum(), y, create_graph=True)[0]
        # syz_z = torch.autograd.grad(syz.sum(), z, create_graph=True)[0]

        # syy_y = torch.autograd.grad(syy.sum(), y, create_graph=True)[0]
        # syz_z = torch.autograd.grad(syz.sum(), z, create_graph=True)[0]
        # sxy_x = torch.autograd.grad(sxy.sum(), x, create_graph=True)[0]

        # szz_z = torch.autograd.grad(szz.sum(), z, create_graph=True)[0]
        # sxz_x = torch.autograd.grad(sxz.sum(), x, create_graph=True)[0]
        # syz_y = torch.autograd.grad(syz.sum(), y, create_graph=True)[0]

        # Fx = (sxx_x + sxy_y + syz_z)
        # Fy = (sxy_x + syz_z + syy_y)
        # Fz = (szz_z + sxz_x + syz_y)

        # mse_losspde= mse(Fx, torch.zeros_like(x)) + mse(Fy, torch.zeros_like(x)) + mse(Fz, torch.zeros_like(x))


        # return mse_losspde, u, v, w, sxx, syy, szz
        return 0, u, v, w, 0, 0, 0