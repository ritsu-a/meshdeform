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
from mesh_process import tetrahedralize, load_mesh, compute_tetrahedron_centroids, save_pointcloud, save_mesh, plt_mesh, plt_meshes, plt_mesh2, plt_mesh3
from tqdm import tqdm
from torch_util import angle_axis_to_rotation_matrix
import cv2 

from pytorch3d.loss import chamfer_distance

torch.manual_seed(2222)
device = torch.device("cuda:2")

class Net(nn.Module):
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

def pinnLoss(x,y,z,t, mse, net):

    logits = net(x,y,z,t)
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

def dataGenerate(data_root):

    vertices, faces = load_mesh(data_root)
    mesh = tetrahedralize(vertices,faces)
    centroids = compute_tetrahedron_centroids(mesh.points, mesh.cells)

    verts_num = len(vertices)

    dataPoints = np.vstack((vertices, centroids))
    # dataBC =  dataPoints[:verts_num]
    # dataI = dataPoints[verts_num:]

    return dataPoints, faces, verts_num

def samplePcd(pcd, num):
    dist = np.inf * np.ones((pcd.shape[0]))
    result_pcd = np.zeros((num, 3))
    for idx in range(num):
        pt = pcd[np.argmax(dist)]
        result_pcd[idx, :] = pt 
        # print(pt)
        dist = np.minimum(dist, np.linalg.norm(pcd - pt, axis=1))
    return result_pcd


def chamferLoss(pcd, pcd_gt):
    loss = 0
    for idx in range(pcd.shape[0]):
        pt = pcd[idx, :]
        loss += torch.min(torch.linalg.norm(pcd_gt - pt, dim=1))
    return loss


if __name__ == "__main__":

    f = -0.5
    save_dir = "./data_noisy_1123_2"
    os.makedirs(save_dir, exist_ok =True)
    os.makedirs(save_dir + "/video", exist_ok = True)
    os.makedirs(save_dir + "/imgs", exist_ok = True)
    os.makedirs(save_dir + "/pointcloud", exist_ok = True)
    os.makedirs(save_dir + "/ply", exist_ok = True)
    os.makedirs(save_dir + "/png", exist_ok = True)


    meta = np.load(f"./constForce/data_2.npz")

    for idx in range(100):
        verts = meta["deformed_verts"][idx]
        faces = meta["faces"][idx]

    canonical_verts = meta["deformed_verts"][0]
    canonical_faces = meta["faces"][0]

    dataPoints = torch.tensor(canonical_verts, dtype = torch.float32, requires_grad = True).to(device)

    net = Net().to(device)

    num_epochs = 100000
    
    lr = 0.001
    mse = torch.nn.MSELoss() # Mean squared error
    params = net.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)

    loss_pde= []
    loss_bc = []
    losses = []
    loss_u = []
    loss_v = []
    loss_w = []


    X = dataPoints[:, 0].reshape(-1, 1)
    Y = dataPoints[:, 1].reshape(-1, 1)
    Z = dataPoints[:, 2].reshape(-1, 1)

    point_gt_sampled = np.zeros((100, 100, 3))
    for idx in range(100):
        point_gt_sampled[idx, :, :] = samplePcd(meta["deformed_verts"][idx], 100)


    point_gt_noisy = point_gt_sampled + np.random.rand(100, 100, 3) * 0.04 - 0.02
    point_gt_sampled_gpu = torch.tensor(point_gt_sampled, dtype = torch.float32, requires_grad = False).to(device)
    point_gt_noisy_gpu = torch.tensor(point_gt_noisy, dtype = torch.float32, requires_grad = False).to(device)
    

    for epoch in tqdm(range(num_epochs)):

        t = epoch % 100

        optimizer.zero_grad()

        temp_lossbc = torch.tensor([]).to(device)
        temp_losspde = torch.tensor([]).to(device)
        temp_loss = torch.tensor([]).to(device)

        dataT = torch.tensor(meta["deformed_verts"][t], dtype = torch.float32, requires_grad = True).to(device)
        XT = dataT[:, 0].reshape(-1, 1)
        YT = dataT[:, 1].reshape(-1, 1)
        ZT = dataT[:, 2].reshape(-1, 1)


        T = t * torch.ones_like(dataPoints[:,0].reshape(-1,1), dtype = torch.float32, requires_grad = True).to(device)
       

        mse_losspde, u_pred, v_pred, w_pred, _, _, _ = pinnLoss(X, Y, Z, T, mse, net)


        pointPredict = torch.cat([u_pred,v_pred,w_pred],axis=1) + dataPoints

        chamfer_loss, _ = chamfer_distance(pointPredict[None, :, :], point_gt_noisy_gpu[t][None, :, :])

        u_loss = mse(u_pred, XT-X)
        v_loss = mse(v_pred, YT-Y)
        w_loss = mse(w_pred, ZT-Z)
        # loss = mse_losspde + u_loss + v_loss + w_loss
        # loss = u_loss + v_loss + w_loss
        loss = chamfer_loss
        # print(u_loss.item(), v_loss.item(), w_loss.item())

        loss.backward(retain_graph = True)
        optimizer.step()

        losses.append(loss.item())
        loss_u.append(u_loss.item())
        loss_v.append(v_loss.item())
        loss_w.append(w_loss.item())

        # if (epoch+1) % 100 == 0:
        #     print(epoch+1,"Traning Loss:",loss.item())
        # if (epoch+1) % 100 == 0:

        #     dataTest = dataPoints.clone()
        #     _, uPred, vPred, wPred, sxxPred, syyPred, szzPred = pinnLoss(X, Y, Z, T, mse, net)
            
        #     strain = torch.cat([uPred,vPred,wPred],axis=1)
        #     strain = strain.data.cpu().numpy()
        #     dataTest = dataTest.data.cpu().numpy()

        #     dataShift = dataTest + strain

        #     ep = epoch + 1

        #     save_pointcloud(dataShift, os.path.join(save_dir + "/pointcloud", f"epoch:{ep}_all_pointcloud_{t}.ply" ))
        #     save_mesh(dataShift, faces, os.path.join(save_dir + "/ply", f"epoch:{ep}_deformed_mesh_{t}.ply"))
        #     plt_mesh(dataShift, faces, os.path.join(save_dir + "/png", f"epoch:{ep}_deformed_mesh_{t}.png"))

    meshes = []
    for idx in range(100):
        t = idx

        T = idx * torch.ones_like(dataPoints[:,0].reshape(-1,1), dtype = torch.float32, requires_grad = True).to(device)
        
       

        _, uPred, vPred, wPred, sxxPred, syyPred, szzPred = pinnLoss(X, Y, Z, T, mse, net)
            
        strain = torch.cat([uPred,vPred,wPred],axis=1)
        strain = strain.data.cpu().numpy()
        predict = meta["deformed_verts"][0] + strain 
        plt_mesh3(point_gt_sampled[idx], point_gt_noisy[idx], predict, faces, save_dir + f"/imgs/data_{idx}.png")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, channels = cv2.imread(save_dir + "/imgs/data_0.png").shape
    frame_size = (width, height)
    out = cv2.VideoWriter(save_dir + "/output.mp4", fourcc, 5, frame_size)


    file_list = os.listdir(save_dir + "/imgs/")
    for file in file_list:
        cv_dst = cv2.imread(save_dir + f"/imgs/{file}")
        out.write(cv_dst)


    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Total_Loss')
    plt.title('Loss Decrease Over Time')
    plt.savefig(os.path.join(save_dir, 'total_loss.png'), dpi=300)
    plt.close()

    plt.plot(loss_u)
    plt.xlabel('Epoch')
    plt.ylabel('U_Loss')
    plt.title('Loss Decrease Over Time')
    plt.savefig(os.path.join(save_dir, 'u_loss.png'), dpi=300)
    plt.close()
    
    plt.plot(loss_v)
    plt.xlabel('Epoch')
    plt.ylabel('V_Loss')
    plt.title('Loss Decrease Over Time')
    plt.savefig(os.path.join(save_dir, 'v_loss.png'), dpi=300)
    plt.close()
    
    plt.plot(loss_w)
    plt.xlabel('Epoch')
    plt.ylabel('W_Loss')
    plt.title('Loss Decrease Over Time')
    plt.savefig(os.path.join(save_dir, 'w_loss.png'), dpi=300)
    plt.close()

