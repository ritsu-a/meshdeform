import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import open3d as o3d
import tetgen
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import meshio
import ipdb
from mesh_process import tetrahedralize, load_mesh, compute_tetrahedron_centroids, save_pointcloud, save_mesh, plt_mesh
from tqdm import tqdm

torch.manual_seed(2222)
device = torch.device("cuda:0")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        n_input = 3
        n_output = 1
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

    def forward(self, x,y,z):

        inputs = torch.cat([x,y,z],axis=1)

        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        layer5_out = torch.tanh(self.hidden_layer5(layer4_out))
        layer6_out = torch.tanh(self.hidden_layer6(layer5_out))

        output = self.output_layer(layer6_out) ## For regression, no activation is used in output layer

        return output

def pinnLoss(x,y,z, mse, net_u, net_v, net_w):

    u = net_u(x,y,z)
    v = net_v(x,y,z)
    w = net_w(x,y,z)

    
    E = 1
    nu = 0.3
    
    # 拉梅常数
    lmbd = E * nu/((1+nu)*(1-2*nu))
    mu = E/(2*(1+nu))


    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]

    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    v_z = torch.autograd.grad(v.sum(), z, create_graph=True)[0]

    w_x = torch.autograd.grad(w.sum(), x, create_graph=True)[0]
    w_y = torch.autograd.grad(w.sum(), y, create_graph=True)[0]
    w_z = torch.autograd.grad(w.sum(), z, create_graph=True)[0]

    exx = u_x
    eyy = v_y
    ezz = w_z

    exy = 1/2*(u_y + v_x)
    eyz = 1/2*(v_z + w_y)
    ezx = 1/2*(w_x + u_z)

    sxx = (lmbd+2*mu)*exx + lmbd*eyy + lmbd*ezz
    syy = (lmbd+2*mu)*eyy + lmbd*exx + lmbd*ezz
    szz = (lmbd+2*mu)*ezz + lmbd*exx + lmbd*eyy
    sxy = 2*mu*exy
    syz = 2*mu*eyz
    sxz = 2*mu*ezx

    sxx_x = torch.autograd.grad(sxx.sum(), x, create_graph=True)[0]
    sxy_y = torch.autograd.grad(sxy.sum(), y, create_graph=True)[0]
    syz_z = torch.autograd.grad(syz.sum(), z, create_graph=True)[0]

    syy_y = torch.autograd.grad(syy.sum(), y, create_graph=True)[0]
    syz_z = torch.autograd.grad(syz.sum(), z, create_graph=True)[0]
    sxy_x = torch.autograd.grad(sxy.sum(), x, create_graph=True)[0]

    szz_z = torch.autograd.grad(szz.sum(), z, create_graph=True)[0]
    sxz_x = torch.autograd.grad(sxz.sum(), x, create_graph=True)[0]
    syz_y = torch.autograd.grad(syz.sum(), y, create_graph=True)[0]

    Fx = (sxx_x + sxy_y + syz_z)
    Fy = (sxy_x + syz_z + syy_y)
    Fz = (szz_z + sxz_x + syz_y)

    mse_losspde= mse(Fx, torch.zeros_like(x)) + mse(Fy, torch.zeros_like(x)) + mse(Fz, torch.zeros_like(x))


    return mse_losspde, u, v, w, sxx, syy, szz


def dataGenerate(data_root):

    vertices, faces = load_mesh(data_root)
    mesh = tetrahedralize(vertices,faces)
    centroids = compute_tetrahedron_centroids(mesh.points, mesh.cells)

    verts_num = len(vertices)

    dataPoints = np.vstack((vertices, centroids))
    # dataBC =  dataPoints[:verts_num]
    # dataI = dataPoints[verts_num:]

    return dataPoints, faces, verts_num


def boundaryLoss(f, dataBC, mse, net_u, net_v, net_w):

    static_verts = [1536, 1537, 1193, 1196, 1197, 1199, 1200, 1533, 1198, 1417, 1706, 1419, 1420, 1710, 1426, 1418]
    _, u_sta, v_sta, w_sta, _, _, _ = pinnLoss(dataBC[static_verts,0].reshape(-1,1), dataBC[static_verts,1].reshape(-1,1), dataBC[static_verts,2].reshape(-1,1), mse, net_u, net_v, net_w)

    handle_verts = [71, 73, 81, 82, 86, 88, 80, 272, 307, 308, 279, 282, 284, 281, 452, 485, 487, 488, 461, 472, 486, 1385, 1386, 1675, 1388, 1677, 1390, 1387]
    _, _, _, _, sxx_han, syy_han, szz_han = pinnLoss(dataBC[handle_verts,0].reshape(-1,1), dataBC[handle_verts,1].reshape(-1,1), dataBC[handle_verts,2].reshape(-1,1), mse, net_u, net_v, net_w)


    all_verts = set(range(len(dataBC)))
    static_verts = set(static_verts)
    handle_verts = set(handle_verts)

    remaining_verts = all_verts - static_verts - handle_verts
    remaining_verts = list(remaining_verts)

    _, _, _, _, sxx_rem, syy_rem, szz_rem = pinnLoss(dataBC[remaining_verts,0].reshape(-1,1),dataBC[remaining_verts,1].reshape(-1,1), dataBC[remaining_verts,2].reshape(-1,1), mse, net_u, net_v, net_w)


    net_bc_static_u_free = u_sta.view(-1,1)
    net_bc_static_v_free = v_sta.view(-1,1)
    net_bc_static_w_free = w_sta.view(-1,1)

    net_bc_remaining_sxx_free = sxx_rem.view(-1,1)
    net_bc_remaining_syy_free = syy_rem.view(-1,1)
    net_bc_remaining_szz_free = szz_rem.view(-1,1)

    net_bc_handle_sxx_load = sxx_han.view(-1,1)
    net_bc_handle_syy_load = syy_han.view(-1,1)
    net_bc_handle_szz_load = szz_han.view(-1,1)

    mse_bc_static_u = mse(net_bc_static_u_free, torch.zeros_like(net_bc_static_u_free))
    mse_bc_static_v = mse(net_bc_static_v_free, torch.zeros_like(net_bc_static_v_free))
    mse_bc_static_w = mse(net_bc_static_w_free, torch.zeros_like(net_bc_static_w_free))

    mse_bc_remaining_sxx = mse(net_bc_remaining_sxx_free, torch.zeros_like(net_bc_remaining_sxx_free))
    mse_bc_remaining_syy = mse(net_bc_remaining_syy_free, torch.zeros_like(net_bc_remaining_syy_free))
    mse_bc_remaining_szz = mse(net_bc_remaining_szz_free, torch.zeros_like(net_bc_remaining_szz_free))

    mse_bc_handle_sxx_load = mse(net_bc_handle_sxx_load, f*torch.ones_like(net_bc_handle_sxx_load))
    # mse_bc_handle_syy_load = mse(net_bc_handle_syy_load, f*torch.ones_like(net_bc_handle_syy_load))
    # mse_bc_handle_szz_load = mse(net_bc_handle_szz_load, f*torch.ones_like(net_bc_handle_szz_load))
    mse_bc_handle_syy_load = mse(net_bc_handle_syy_load, torch.zeros_like(net_bc_handle_syy_load))
    mse_bc_handle_szz_load = mse(net_bc_handle_szz_load, torch.zeros_like(net_bc_handle_szz_load))

    mse_lossbc = (mse_bc_static_u + mse_bc_static_v + mse_bc_static_w
            + mse_bc_remaining_sxx + mse_bc_remaining_syy + mse_bc_remaining_szz
            + mse_bc_handle_sxx_load + mse_bc_handle_syy_load + mse_bc_handle_szz_load)

    return mse_lossbc ,handle_verts, static_verts

def transform_pointcloud(pointcloud, num_points, translation_prob=0.3, rotation_prob=0.3):

    num_points_original = pointcloud.shape[0]
    num_dims = pointcloud.shape[1]
    
    transformed_pointcloud = np.zeros((num_points, num_points_original, num_dims))
    
    for i in range(num_points):

        transformation_type = np.random.choice(['translation', 'rotation', 'both'], p=[translation_prob, rotation_prob, 1-(translation_prob+rotation_prob)])
        # TODO: 修改随机的范围
        if transformation_type == 'translation':
            translation_vector = np.random.uniform(-10, 10, size=(num_dims,))
            transformed_points = pointcloud + translation_vector

        elif transformation_type == 'rotation':
        # TODO:简单起见，目前只绕z轴旋转
            angle = np.random.uniform(0, 2 * np.math.pi)
            rotation_matrices = np.array([[np.math.cos(angle), -np.math.sin(angle), 0],
                                            [np.math.sin(angle), np.math.cos(angle), 0],
                                            [0, 0, 1]])

            transformed_points = np.dot(pointcloud, rotation_matrices.T)
        else:

            translation_vector = np.random.uniform(-10, 10, size=(num_dims,))
            translated_points = pointcloud + translation_vector
            
            angle = np.random.uniform(0, 2 * np.math.pi)
            rotation_matrices = np.array([[np.math.cos(angle), -np.math.sin(angle), 0],
                                            [np.math.sin(angle), np.math.cos(angle), 0],
                                            [0, 0, 1]])

            transformed_points = np.dot(translated_points, rotation_matrices.T)


        transformed_pointcloud[i] = transformed_points
    
    return transformed_pointcloud

if __name__ == "__main__":

    f = -0.5
    save_dir = "./data_3d_1922_1"
    os.makedirs(save_dir, exist_ok =True)
    dataPoints, faces, num = dataGenerate('/share/lxx/deformable_result/deformable_result/model/1/0516_manifold_3000.obj')
    TotaldataPoints = transform_pointcloud(dataPoints, 4)

    TotaldataPoints = torch.tensor(TotaldataPoints, dtype = torch.float32, requires_grad = True).to(device)

    net_u = Net().to(device)
    net_v = Net().to(device)
    net_w = Net().to(device)

    num_epochs = 1000
    lr = 0.001
    mse = torch.nn.MSELoss() # Mean squared error
    params = list(net_u.parameters()) + list(net_v.parameters()) + list(net_w.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    loss_pde= []
    loss_bc = []
    losses = []


    for epoch in tqdm(range(num_epochs)):

        optimizer.zero_grad()

        temp_lossbc = torch.tensor([]).to(device)
        temp_losspde = torch.tensor([]).to(device)
        temp_loss = torch.tensor([]).to(device)

        for dataPoint in TotaldataPoints:

            mse_losspde, _, _, _, _, _, _ = pinnLoss(dataPoint[:,0].reshape(-1,1), dataPoint[:,1].reshape(-1,1), dataPoint[:,2].reshape(-1,1), mse, net_u, net_v, net_w)
            dataBC =  dataPoint[:num]
            dataI = dataPoint[num:]

            mse_lossbc, handle_verts, static_verts = boundaryLoss(f, dataBC, mse, net_u, net_v, net_w)


            loss = mse_losspde + mse_lossbc


            temp_losspde = torch.cat((temp_losspde, mse_losspde.unsqueeze(0)))
            temp_lossbc = torch.cat((temp_lossbc, mse_lossbc.unsqueeze(0)))
            temp_loss = torch.cat((temp_loss, loss.unsqueeze(0)))

        losspde = torch.mean(temp_losspde)
        lossbc = torch.mean(temp_lossbc)
        loss = torch.mean(temp_loss)

        loss.backward(retain_graph = True)
        optimizer.step()

        loss_pde.append(losspde.item())
        loss_bc.append(lossbc.item())
        losses.append(loss.item())

        if (epoch+1) % 100 == 0:
            print(epoch+1,"Traning Loss:",loss.item())
            print(f'PDE Loss: {mse_losspde:.4e}, BC Loss: {mse_lossbc:.4e}')
        if (epoch+1) % 100 == 0:

            dataTest = torch.tensor(dataPoints, dtype = torch.float32, requires_grad = True).to(device)
            _, uPred, vPred, wPred, sxxPred, syyPred, szzPred = pinnLoss(dataTest[:,0].reshape(-1,1), dataTest[:,1].reshape(-1,1), dataTest[:,2].reshape(-1,1), mse, net_u, net_v, net_w)
            
            strain = torch.cat([uPred,vPred,wPred],axis=1)
            strain = strain.data.cpu().numpy()
            dataTest = dataTest.data.cpu().numpy()

            dataShift = dataTest + strain

            ep = epoch + 1

            save_pointcloud(dataShift, os.path.join(save_dir, f"epoch:{ep}_all_pointcloud.ply" ))
            save_mesh(dataShift[:num], faces, list(handle_verts) ,list(static_verts), os.path.join(save_dir, f"epoch:{ep}_deformed_mesh.ply"))
            plt_mesh(dataShift[:num], faces, list(handle_verts) ,list(static_verts), os.path.join(save_dir, f"epoch:{ep}_deformed_mesh.png"))


    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Total_Loss')
    plt.title('Loss Decrease Over Time')
    plt.savefig(os.path.join(save_dir, 'total_loss.png'), dpi=300)
    plt.close()

    plt.plot(loss_pde)
    plt.xlabel('Epoch')
    plt.ylabel('PDE_Loss')
    plt.title('PDE Loss Decrease Over Time')
    plt.savefig(os.path.join(save_dir, 'loss_pde.png'), dpi=300)
    plt.close()

    plt.plot(loss_bc)
    plt.xlabel('Epoch')
    plt.ylabel('BC_Loss')
    plt.title('BC Loss Decrease Over Time')
    plt.savefig(os.path.join(save_dir, 'loss_bc.png'), dpi=300)
    plt.close()