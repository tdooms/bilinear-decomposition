import torch
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from mnist.utils import *

def create_Q_from_upper_tri_idxs(Q_vec, idxs):
    triu_indices = torch.triu_indices(len(idxs),len(idxs))
    tril_indices = torch.tril_indices(len(idxs),len(idxs))
    Q = torch.zeros((len(idxs),len(idxs))).to(Q_vec.device)
    Q[triu_indices[0],triu_indices[1]] = Q_vec
    Q[tril_indices[0],tril_indices[1]] = Q.T[tril_indices[0],tril_indices[1]]
    return Q

class EigenvectorPlotter():
    # Eigenvector Plotter for image-based datasets
    def __init__(self, B, logits, dataset = None, img_size=(28,28), Embed = None):
        self.B = B      #[component, in1, in2]
        self.logits = logits    #[component, out]
        self.dataset = dataset
        self.img_size = img_size
        self.Embed = Embed #[hidden, input]

    def plot_component(self, component, suptitle=None, topk_eigs = 3, sort='eigs', 
        vmax=None, classes = None, filename=None, input_img_labels = False, **kwargs):
        device = self.B.device
        Q = self.B[component]
        
        eigvals, eigvecs = torch.linalg.eigh(Q)
        if self.Embed is not None:
            eigvecs = self.Embed.T @ eigvecs
        eigvals_orig = eigvals.clone()

        if self.dataset is not None:
            mean_acts, avg_sims = self.get_mean_eigen_acts(eigvecs, eigvals, self.dataset)
        else:
            mean_acts, avg_sims = torch.ones(self.img_size[0]*self.img_size[1]), None  

        title_fn = self.get_title_fn(sort)

        eigvecs, eigvals, mean_acts = self.select_eigvecs(topk_eigs, sort, eigvecs, eigvals, mean_acts, avg_sims)

        #create image matrix
        images = eigvecs.T
        if vmax is None:
            vmax = 0.9 * images[torch.logical_not(images.isnan())].abs().max()

        #get top input activations & define mosaic
        if self.dataset is not None:
            top_imgs, top_acts, top_sims = self.get_top_act_images(eigvecs, eigvals, self.dataset, k=3)
            plot_signs = self.fix_eig_signs(top_sims, eigvecs)

            mosaics = []

            mosaic = []
            widths = [1.05]
            for j in range(3):
                line = ["logits"]
                for i in range(topk_eigs):
                    line += [f"pos_eig_{i}", f"pos_act_{i}_{j}"]
                    if j == 0:
                        widths += [1, 0.33]
                mosaic.append(line)
            mosaics.append(mosaic)

            mosaic = []
            for j in range(3):
                line = ["eig_dist"]
                for i in range(topk_eigs):
                    line += [f"neg_eig_{i}", f"neg_act_{i}_{j}"]
                mosaic.append(line)
            mosaics.append(mosaic)
        else:
            mosaics = []
            mosaic_line = ["logits"] + [f"pos_eig_{i}" for i in range(topk_eigs)]
            mosaics.append(mosaic_line)
            
            mosaic_line = ["eig_dist"] + [f"neg_eig_{i}" for i in range(topk_eigs)]
            mosaics.append(mosaic_line)

            widths = [1.05] + topk_eigs*[1]
            plot_signs = [1] * len(eigvecs.shape[1])

        #subplots
        h = 4
        w = 4
        figsize = (w * (topk_eigs*(1.33)+1), 2 * h + 0)

        fig = plt.figure(figsize=figsize, layout='constrained')
        subfigs = fig.subfigures(2, 1)

        #first row
        subfigs[0].suptitle('Positive Eigenvectors', fontsize=21)
        width_ratios = [1] + topk_eigs * [1.3]
        row_subfigs = subfigs[0].subfigures(1,1+topk_eigs, 
                                            width_ratios=width_ratios,
                                            # wspace = 0.07
                                            )
        
        ax = row_subfigs[0].add_subplot(111)
        self.plot_eigvals(ax, eigvals_orig)
        
        
        for i, subfig in enumerate(row_subfigs[1:]):
            colorbar = True if i == topk_eigs-1 else False
            title = title_fn(eigvals[i], mean_acts[i])
            self.plot_eigenvector(subfig, images[i] * plot_signs[i], top_imgs[:,i], 
            top_acts[:,i], colorbar, vmax, title=title, 
            input_img_labels = input_img_labels, **kwargs)

        #second row
        subfigs[1].suptitle('Negative Eigenvectors', fontsize=21)
        width_ratios = [1] + topk_eigs * [1.3]
        row_subfigs = subfigs[1].subfigures(1,1+topk_eigs, 
                                            width_ratios=width_ratios,
                                            # wspace = 0.07
                                            )
        
        
        ax = row_subfigs[0].add_subplot(111)
        self.plot_logits(ax, component, classes)
        
        for i, subfig in enumerate(row_subfigs[1:]):
            colorbar = True if i == topk_eigs-1 else False
            j = topk_eigs + i
            title = title_fn(eigvals[j], mean_acts[j])
            self.plot_eigenvector(subfig, images[j] * plot_signs[j], top_imgs[:,j], 
            top_acts[:,j], colorbar, vmax, title=title, 
            input_img_labels = input_img_labels, **kwargs)

        subfigs[0].text(0.05,0.99,f"{suptitle}", va="center", ha="left", size=27)
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight')
        plt.show()

    @staticmethod
    def get_mean_eigen_acts(eigvecs, eigvals, dataset, img_size=(28,28)):
        device = eigvecs.device
        images = dataset.data.to(device).reshape(-1, img_size[0]*img_size[1])/255 #convert uint8 to float
        sims = (images @ eigvecs)
        acts = eigvals * (sims)**2
        return acts.mean(dim=0), sims.mean(dim=0)

    @staticmethod
    def get_top_act_images(eigvecs, eigvals, dataset, k = 3, img_size=(28,28)):
        device = eigvecs.device
        images = dataset.data.to(device).reshape(-1, img_size[0]*img_size[1])/255 #convert uint8 to float
        sims = (images @ eigvecs)
        acts = eigvals * (sims)**2
        topk_idxs = acts.abs().topk(k, dim=0).indices
        eig_idxs = torch.arange(eigvecs.shape[1]).repeat(k,1)
        top_acts = acts[topk_idxs, eig_idxs]
        top_imgs = images[topk_idxs]
        top_sims = sims[topk_idxs, eig_idxs]
        return top_imgs, top_acts, top_sims

    def fix_eig_signs(self, top_sims, eigvecs):
        signs = top_sims.mean(dim=0).sign()
        return signs

    def get_title_fn(self, sort):
        if self.dataset is not None:
            if sort == 'activations':
                return lambda x,y: f"Mean Act={y:.2f}, Eig={x:.2f}"
            else:
                return lambda x,y: f"Eig={x:.2f}, Mean Act={y:.2f}"
        else:
            title_fn = lambda x,y: f"Eig={x:.2f}"
    
    def select_eigvecs(self, topk_eigs, sort, eigvecs, eigvals, mean_acts, avg_sims):
        #flip sign of eigvecs
        if avg_sims is not None:
            signs = avg_sims.sign()
        else:
            signs = eigvecs.sum(dim=0).sign()
        eigvecs = eigvecs * signs.unsqueeze(0)

        #sort
        if (self.dataset is not None) and (sort=='activations'):
            sort_idxs = mean_acts.argsort()
        else:
            sort_idxs = eigvals.argsort()
        eigvecs = eigvecs[:,sort_idxs]
        eigvals = eigvals[sort_idxs]
        mean_acts = mean_acts[sort_idxs]

        #subset to topk positive and negative eigs
        eig_indices = torch.arange(topk_eigs).to(eigvecs.device)
        eig_indices = torch.cat([-eig_indices-1, eig_indices])
        eigvals = eigvals[eig_indices]
        eigvecs = eigvecs[:,eig_indices]
        mean_acts = mean_acts[eig_indices]

        return eigvecs, eigvals, mean_acts

    

    def plot_logits(self, ax, component, classes):
        logits = self.logits[component]
        if classes is None:
            classes = torch.arange(len(logits))
            rotation = None
            direction = None
            pad = 0
            va = None
        else:
            rotation = 'vertical'
            direction = 'in'
            pad = -10
            va = 'bottom'
        ax.bar(range(len(classes)), logits.cpu().detach())
        ax.set_title('Logit Outputs', fontsize=18)
        ax.set_xticks(range(len(classes)), classes, rotation=rotation, va=va)
        ax.tick_params(labelsize=15)
        ax.tick_params(axis='x', direction=direction, pad=pad)
        # ax.set_xlabel('Classes', fontsize=18)
        ax.set_ylabel('Logits', fontsize=18)

    def plot_eigvals(self, ax, eigvals):
        ax.plot(eigvals.cpu().detach(), '.-', markersize=10)
        ax.set_title('Spectrum', fontsize=18)
        ax.set_ylabel('Eigenvalues', fontsize=18)
        # ax.set_xlabel('Index', fontsize=18)

    def plot_eigenvector(self, fig, image, top_imgs, top_acts, colorbar, vmax, title=None, input_img_labels=False, **kwargs):  
        subfigs = fig.subfigures(1,2, width_ratios=[1, 0.18], wspace=0)
        if title:
            fig.suptitle(title, fontsize=18)

        ax = subfigs[0].add_subplot(111)
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'RdBu'
        if 'norm' in kwargs:
            im = ax.imshow(image.reshape(*self.img_size).cpu().detach(), **kwargs)
        else:
            im = ax.imshow(image.reshape(*self.img_size).cpu().detach(), 
            vmin=-vmax, vmax=vmax, **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        if colorbar:
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_ticks(ticks=[vmax, 0, -vmax])
            cbar.ax.tick_params(labelsize=10)

        axs = subfigs[1].subplots(3,1)
        for i, ax in enumerate(axs):
            self.plot_input_image(ax, top_imgs[i], top_acts[i], 
            input_img_labels = input_img_labels)

    def plot_input_image(self, ax, img, act, input_img_labels = False):
        ax.imshow(img.reshape(*self.img_size), cmap='Greys', vmin=0, vmax=1)
        if input_img_labels:
            ax.set_title(f'act={act:.1f}')
        ax.set_xticks([])
        ax.set_yticks([])


def plot_B_tensor_image_eigenvectors(B,  idx, **kwargs):
    #legacy
    class FakeSVD():
        def __init__(self):
            device = B.device
            d = B.shape[0]
            self.U = torch.eye(d,d).to(device)
            self.S = torch.ones(d).to(device)
            self.V = B.T
    
    fake_svd = FakeSVD()
    plot_full_svd_component_for_image(fake_svd, torch.eye(B.shape[0], B.shape[0]), idx, 
        **kwargs)


def plot_full_svd_component_for_image(svd, W_out, svd_comp, idxs=None,
    topk_eigs = 4, img_size = (28,28), upper_triangular = True, classes = np.arange(10),
    title = 'SVD Component', vmax=None, dataset = None, sort='eigs'):

    device = svd.V.device
    if idxs is None:
        idxs = torch.arange(img_size[0] * img_size[1])
        upper_triangular = False
    idx_pairs = torch.tensor(list(itertools.combinations_with_replacement(idxs,2))).to(device)

    # logit outputs
    U_S = svd.U[:,svd_comp] * svd.S[svd_comp]
    logits = (W_out @ U_S).unsqueeze(0)

    if upper_triangular:
        Q_vec = svd.V[:,svd_comp]
        Q = create_Q_from_upper_tri_idxs(Q_vec, idxs)
    else:
        Q = svd.V[:,svd_comp].reshape(len(idxs), len(idxs))
    B = Q.unsqueeze(0)

    Plotter = EigenvectorPlotter(B, logits, dataset = dataset, img_size=img_size)
    Plotter.plot_component(0, suptitle=title, topk_eigs=topk_eigs, sort=sort, vmax=vmax, classes=classes)



def plot_topk_model_bottleneck(model, svds, sing_val_type, svd_components, topK_list, test_loader, 
    input_idxs = None, print_bool = False, device=None, rms_norm = None):

    accuracy_dict = defaultdict(list)
    for layer in tqdm(range(len(model.layers))):
        for topK in tqdm(topK_list, leave=False):
            topKs = [svd_components] * len(model.layers)
            topKs[layer] = topK
            topk_model = BilinearModelTopK(model, svds, sing_val_type, input_idxs=input_idxs)
            if rms_norm is not None:
                topk_model.cfg.rms_norm = rms_norm
            topk_model.set_parameters(topKs)
            if device is not None:
                topk_model = topk_model.to(device)
            accuracy = topk_model.validation_accuracy(test_loader, print_acc=False)
            accuracy_dict[layer].append(accuracy)
            if print_bool:
                print(f'Layer = {layer}, Components = {topK}, Accuracy = {accuracy:.2f}%')

    topk_baseline_model = get_topK_baseline_model(model, input_idxs)
    baseline_accuracy = topk_baseline_model.validation_accuracy(test_loader, print_acc=False)


    plt.figure(figsize=(5,4))
    for layer in range(len(model.layers)):
        acc = np.array(accuracy_dict[layer])
        topKs = np.array(topK_list)
        acc_drop = 100 * (baseline_accuracy - acc)/baseline_accuracy
        plt.plot(topKs, acc_drop, '-', label=f'Layer {layer}')

    plt.xlabel('SVD Components')
    plt.ylabel('Accuracy Drop (%)\nCompared to base model')
    plt.title('Single Layer Bottlenecks')
    plt.yscale('log')
    plt.xscale('log')

    ax = plt.gca()
    ax.set_yticks([0.1, 1, 10, 100], ['0.1%', '1%', '10%', '100%'])
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 300], [1, 2, 5, 10, 20, 50, 100, 300])
    plt.legend()


def plot_max_activations(Q, idxs = None, img_size = (28,28)):
    device = Q.device
    if idxs is None:
        idxs = torch.arange(img_size[0] * img_size[1])
    x_pos, x_neg, act_pos, act_neg = get_max_pos_neg_activations(Q)
    
    plt.subplot(1, 2, 1)
    max_img = torch.zeros(img_size[0]*img_size[1]).to(device)
    max_img[:] = float('nan')
    max_img[idxs] = x_pos
    max_img = max_img.reshape((img_size[0], img_size[1]))
    plt.imshow(max_img.cpu().detach().numpy(), cmap='RdBu', vmin=-1, vmax=1)
    plt.title(f"Max Pos. Activation, a={act_pos:.2f}")

    plt.subplot(1, 2, 2)
    max_img = torch.zeros(img_size[0]*img_size[1]).to(device)
    max_img[:] = float('nan')
    max_img[idxs] = x_neg
    max_img = max_img.reshape((img_size[0], img_size[1]))
    plt.imshow(max_img.cpu().detach().numpy(), cmap='RdBu', vmin=-1, vmax=1)
    plt.title(f"Max Neg. Activation, a={act_neg:.2f}")
    plt.show()
