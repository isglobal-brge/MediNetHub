import torch
from torch import optim
import numpy as np
import rich
from sklearn.metrics.pairwise import pairwise_kernels

console = rich.get_console()


def h(alphas, svs):
    """
    Construct the hyperplane:
                w = sum_i alpha_i * sv_i
    where alpha_i is the dual variable and sv_i is a support vector.
    """

    res = 0
    for i in range(len(alphas)):
        res += alphas[i] * svs[i]
    return res


class FedSVM_RandomDeltas:

    """
    Class that define random deltas for each support vector.
    """

    def fit(self, device, svs_to_opt) -> torch.Tensor:
        console.log(
            f"[green bold][Client {self.client_no}][/] Computing for delta displacements, using random deltas..."
        )
        delta = np.random.randn(len(svs_to_opt), self.svs.shape[1]) 
        delta = delta / np.linalg.norm(delta, axis=1).reshape(-1,1) * 0.4
        delta = torch.tensor(
            delta, requires_grad=False, dtype=torch.float32, device=device
        )
        return delta


class FedSVM_NoOptim_MultipleDeltas:

    """
    Class that defines the fitting behavior of a client in the FedSVM algorithm
    when using multiple deltas, one for each support vector, and no optimization
    to find the deltas. Each delta is found by randomly sampling from a normal distribution
    and their last element is set so that the delta vector is orthogonal to the decision boundary.
    """

    def fit(self, device, svs_to_opt) -> torch.Tensor:
        console.log(
            f"[green bold][Client {self.client_no}][/] Computing for delta displacements, using multiple deltas..."
        )
        w = h(self.alphas, self.svs).cpu()
        deltas = []
        for _ in range(len(self.svs)):
            delta0 = np.random.uniform(0.1, 0.3)
            delta = np.random.randn(self.svs.shape[1])
            delta[-1] = -delta[:-1].dot(w[:-1]) / w[-1]

            delta = delta / np.linalg.norm(delta)
            delta *= delta0
            deltas.append(delta.reshape(1,-1))

        deltas = np.concatenate(deltas, axis=0)
        deltas = torch.tensor(deltas, requires_grad=False)
        return deltas[svs_to_opt]


class FedSVM_NoOptim_SingleDelta:

    """
    Class that defines the fitting behavior of a client in the FedSVM algorithm
    when using a single delta vector for every support vectors, and no optimization
    to find it. Delta is found by randomly sampling from a normal distribution and its
    last element is set so that the delta vector is orthogonal to the decision boundary.
    """

    def fit(self, device, svs_to_opt) -> torch.Tensor:
        console.log(
            f"[green bold][Client {self.client_no}][/] Computing for delta displacements, using single deltas..."
        )
        self.delta0 = 0.2
        w = h(self.alphas, self.svs).cpu()
        delta = np.random.randn(self.svs.shape[1])
        delta[-1] = -delta[:-1].dot(w[:-1]) / w[-1]

        delta = delta / np.linalg.norm(delta)
        delta *= self.delta0

        return torch.tensor(delta)


class FedSVM_Optim:

    """
    Class that defines the fitting behavior of a client in the FedSVM algorithm
    when the delta(s) vector(s) are found using an optimization algorithm, so that
    the support vectors are shifed parallel to the decision boundary.
    """

    def fit(self, device, svs_to_opt) -> torch.Tensor:
        console.log(
            f"[green bold][Client {self.client_no}][/] Optimizing delta displacements, using {len(self.svs)} svs..."
        )

        patience = 20
        not_improving = 0

        delta = self.create_delta(device, svs_to_opt)
        best_delta = delta.clone()
        best_loss = np.inf

        optimizer = optim.Adam([delta], lr=0.001)
        # delta = delta.to(device)
        num_iterations = 20000

        K = self.kernel_fun(self.svs, self.svs, **self.kernel)

        for i in range(num_iterations):
            optimizer.zero_grad()
            loss_tuple = self.objective_function(delta, K, svs_to_opt)
            loss = loss_tuple[0] + loss_tuple[1]

            loss.backward()
            optimizer.step()

            if loss >= best_loss:
                not_improving += 1
                if not_improving > patience:
                    console.log(
                        f"[green bold][Client {self.client_no}]\t NOT IMPROVING: stopped at iter {i+1} with loss {loss:.2f} [{loss_tuple[0].item():.4f} + {loss_tuple[1].item():.4f}] and delta.grad norm {torch.norm(delta.grad):.2f}"
                    )
                    break
            else:
                not_improving = 0
                best_loss = loss
                best_delta = delta.clone()

            if (
                loss < self.client_eps
                or (torch.norm(delta.grad) / delta.shape[0]) < 1e-6
            ):
                console.log(
                    f"[green bold][Client {self.client_no}][/] stopped at iter {i+1} with loss {loss:.2f} [{loss_tuple[0].item():.4f} + {loss_tuple[1].item():.4f}] and delta.grad norm {torch.norm(delta.grad):.2f}"
                )
                break

            if (i + 1) % 100 == 0:
                console.log(
                    f"[green bold][Client {self.client_no}][/] loss iter {i+1}: {loss:.4f} [{loss_tuple[0].item():.4f} + {loss_tuple[1].item():.4f}]"
                )

        return best_delta


class FedSVM_SingleDeltas:

    """
    Class that the define the way delta is created, the objective function to
    optimize it and the way it is collected for a client.
    """

    def create_delta(self, device, svs_to_opt):
        """
        Create a delta vector by randomly sampling from a normal distribution
        rescaled by a factor of 0.001.
        Delta0 is a random number between 0.1 and 0.3 and defines the
        desired norm of the delta vector.
        """

        delta = np.random.randn(self.svs.shape[1]) * 0.001
        delta = torch.tensor(
            delta, requires_grad=True, dtype=torch.float32, device=device
        )
        self.delta0 = (0.1 + torch.rand(1) * 0.2) * torch.ones(1)

        return delta

    # Lorenzo's objective function
    # def objective_function(self, delta, K):
    #     delta_matrix = torch.zeros(self.svs.shape) + delta.reshape(1, -1)
    #     K_d = delta_matrix @ self.svs.T
    #     aK = self.alphas @ (K_d.T)
    #     res = aK.dot(aK)
    #     return res, ((torch.norm(delta) - self.delta0[0]) ** 2) * self.svs.shape[0]

    def objective_function(self, delta, K, svs_to_opt):
        """
        The objective function to optimize the delta vector is the sum of two terms:
            1. The first one is the dual constraint that guarantees that the support vectors
            are shifted parallel to the decision boundary;
            2. The second one is the constraint that guarantees that the norm of the delta
            vector is equal to delta0.
        """

        K_d = self.kernel_fun(
            self.svs + (delta.reshape(1, -1)), self.svs, **self.kernel
        )
        aK = self.alphas @ (K_d.T - K)
        res = aK.dot(aK)
        return res, ((torch.norm(delta) - self.delta0[0]) ** 2) * self.svs.shape[0]

   
    def collect_delta_and_shas(self):
        """
        Return the perturbed support vectors each one with its corresponding delta vector
        and their sha as identifier.
        """

        return [(self.svs_shas[index], sv) for index, sv in enumerate(self.svs)]



# class FedSVM_SingleDeltas_Kernel(FedSVM_SingleDeltas):
#     def objective_function(self, delta, K):
#         """
#         The objective function to optimize the delta vector is the sum of two terms:
#             1. The first one is the dual constraint that guarantees that the support vectors
#             are shifted parallel to the decision boundary;
#             2. The second one is the constraint that guarantees that the norm of the delta
#             vector is equal to delta0.
#         """
#         K_d = self.kernel_fun(self.svs + delta, self.svs, **self.kernel)
#         aK = self.alphas @ (K_d.T - K)
#         res = aK.dot(aK)
#         return res, ((torch.norm(delta) - self.delta0[0]) ** 2) * self.svs.shape[0]


class FedSVM_MultipleDeltas:

    """
    Class that the define the way the deltas are created the objective function to
    optimize them and the way they are collected for a client.
    """

    def create_delta(self, device, svs_to_opt):
        """
        Create a matrix of delta vectors by randomly sampling from a normal distribution,
        one for each support vector, rescaled by a factor of 0.001.
        The delta0 vector is a vector containing the desired norm of each delta vector.
        """

        delta = np.random.randn(len(svs_to_opt), self.svs.shape[1]) * 0.001
        delta = torch.tensor(
            delta, requires_grad=True, dtype=torch.float32, device=device
        )
        self.delta0 = (0.1 + torch.rand([len(svs_to_opt)]) * 0.2) * torch.ones(
            len(svs_to_opt)
        )

        return delta

    def objective_function(self, delta, K, svs_to_opt):  # CALCOLO SEMPLIFICATO
        K_d = self.kernel_fun(
            self.svs[svs_to_opt] + delta, self.svs, **self.kernel
        )  # K_d = kernel_fun(svs + delta, svs)
        K = K[svs_to_opt]
        # aK = self.alphas[svs_to_opt] @ (K_d - K)
        aK = self.alphas @ (K_d - K).T
        res = aK.dot(aK)

        return (
            res,
            ((torch.norm(delta, dim=1) - self.delta0.to(self.device)) ** 2).sum()
        )

    def collect_delta_and_shas(self):
        """
        Return the perturbed support vectors each one with its corresponding delta vector
        and their sha as identifier.
        """

        return [(self.svs_shas[index], sv) for index, sv in enumerate(self.svs)]
