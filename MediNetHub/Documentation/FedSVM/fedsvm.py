import sys
import io
import os

# Forzar UTF-8 en Windows - DEBE estar ANTES de cualquier import de rich
if sys.platform == 'win32':
    # Configurar variable de entorno
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Reconfigurar stdout y stderr
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

if "sklearnex" in sys.modules:
    from sklearnex import patch_sklearn

    patch_sklearn()

import json
import numpy as np
from math import floor as floor
from joblib import dump, load
from copy import deepcopy
from hashlib import sha256
from collections import OrderedDict
from rich.table import Table


import torch

# from rff import RandomFourier
from sklearn.kernel_approximation import RBFSampler as RandomFourier
import noniid
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import typer
from typing_extensions import Annotated
from rich.progress import Progress, track
import rich

from datasets import load_dataset
from exphandler import ExpHandler

console = None

from fc_configs import (
    FedSVM_NoOptim_SingleDelta,
    FedSVM_NoOptim_MultipleDeltas,
    FedSVM_SingleDeltas,
    FedSVM_Optim,
    FedSVM_MultipleDeltas,
    FedSVM_RandomDeltas,
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FedSVMClient:

    """
    Class to define the common behavior of a client in the FedSVM algorithm,
    regardless of the type of delta used and the optimization used to find it.
    """

    def __init__(
        self, client_no, X, y, rff, client_eps, C, kernel, device, only_load=False
    ) -> None:
        if only_load:
            return

        self.console = rich.get_console()
        self.client_no = client_no
        self.device = device
        self.X = X
        self.y = y
        self.C = C
        self.svs = None
        self.svs_shas = None
        self.svs_labels = None
        self.alphas = None
        self.delta0 = None
        # self.prev_w = None
        # self.w = None
        self.kernel = kernel
        if kernel["kernel"] == "rff":
            self.kernel["kernel"] = "linear"
        self.Z_ = (
            torch.tensor(rff.transform(self.X))
            if rff is not None
            else torch.tensor(self.X)
        )
        self.known_shas = self.init_known_shas(self.Z_, self.y)
        self.sent = set()
        self.svc = self.local_svm(self.Z_, self.y)
        self.client_eps = client_eps
        self.latest_delta = None
        self.kernel_fun = self.linear_kernel
        self.n_round = 0

    def linear_kernel(self, X, Y, **_):
        return torch.mm(X, Y.T)

    def save_data(self):
        """
        A client can be saved to disk using joblib, only its svc is needed.
        """

        return self.svc

    def load_data(self, data):
        """
        If a client has been saved to disk, it can be loaded using joblib.
        Its behavior is defined by the svc loaded from disk.
        """

        self.svc = data

    def predict(self, X, rff):
        """
        Predict the labels of the examples X, using the random fourier features transformer rff,
        and the svc learned.
        """
        if rff is not None:
            Z = torch.tensor(rff.transform(X))
        else:
            Z = torch.tensor(X)
        return self.svc.predict(Z)

    def local_svm(self, X, y):
        """
        Define the local SVM of the client, using the examples X and the labels y.
        When the rff kernel is used, the X are the random fourier features of the original examples.
        """

        self.console.log(
            f"[green bold][Client {self.client_no}][/] Fitting SVM on {X.shape[0]} samples, {X.shape[1]} features"
        )
        svc = SVC(C=self.C, **self.kernel)
        svc = svc.fit(X, y)
        self.svs = torch.tensor(
            svc.support_vectors_, dtype=torch.float32, device=self.device
        )

        self.console.log(
            f"[green bold][Client {self.client_no}][/] # of known_shas: {len(self.known_shas)}"
        )

        shas = list(self.known_shas.keys())
        self.svs_shas = [shas[index] for index in svc.support_]

        self.alphas = torch.tensor(
            svc.dual_coef_[0], dtype=torch.float32, device=self.device
        )
        self.svs_labels = y[svc.support_]

        # w = torch.tensor(h(self.alphas.numpy(), self.svs.numpy()))
        # if self.prev_w is not None:
        #     w_norm = w / torch.norm(w)
        #     w_d_norm = self.prev_w / torch.norm(self.prev_w)
        #     console.log(
        #         f"\tdot product of w and prev w: {torch.dot(w_norm, w_d_norm):.2f}"
        #     )
        # self.prev_w = deepcopy(w)
        # self.w = w
        return svc

    def Z(self):
        """
        Return the random fourier features of the examples known by the client.
        """

        return torch.cat(
            [sv.cpu().reshape(1, -1) for sv, _ in self.known_shas.values()], dim=0
        )

    def labels(self):
        """
        Return the labels of the examples known by the client.
        """

        return torch.tensor([label for _, label in self.known_shas.values()])

    def init_known_shas(self, X, y):
        """
        Initialize the dictionary of examples known by the client.
        The key is the sha256 hash of the example, the value is a tuple containing the example itself
        and its label. We use the sha256 hash to identify the examples so that its dataset does not explode in size.
        """

        known_shas = OrderedDict()
        for i in range(len(X)):
            known_shas[sha256(X[i].numpy()).hexdigest()] = (X[i], y[i])
        return known_shas

    def update_known_shas(self, svs_clients, y_clients):
        """
        Only the new examples received from the server are added to the known examples.
        New examples are identified by their unique sha256 hash.
        """

        self.console.log(
            f"[green bold][Client {self.client_no}][/] Filtering received examples. received and known shas: "
            + f"{len([ex for client in svs_clients for ex in client])}/{len(self.known_shas)}"
        )
        for ind_client, client in enumerate(svs_clients):
            for ind_sv, (sv_key, sv) in enumerate(client):
                if sv_key not in self.known_shas:
                    self.known_shas[sv_key] = (sv, y_clients[ind_client][ind_sv])
                    self.sent.add(sv_key)
        self.console.log(
            f"[green bold][Client {self.client_no}][/] Number of examples after filtering: {len(self.known_shas)}"
        )

    def receive_svs(self, svs_clients, labels_clients):
        """
        The client receives the support vectors of the other clients from the server
        and updates its known examples. After that, it trains its local SVM using the new examples.
        """

        self.update_known_shas(svs_clients, labels_clients)
        self.svc = self.local_svm(self.Z(), self.labels())

    def send_svs(self):
        """
        The client sends its support vectors to the server but before that it computes the delta
        perturbation to apply to the support vectors.
        """
        self.n_round += 1

        svs_and_shas = self.collect_delta_and_shas()

        # collect into to_send only the support vectors that have not been sent yet
        to_send = []
        to_send_idx = []
        for index, (sv_key, sv) in enumerate(svs_and_shas):
            if sv_key not in self.sent:
                to_send.append((sv_key, sv))
                to_send_idx.append(index)

        # sample only a fraction of the support vectors to send
        perc = sigmoid(self.n_round - 3)  # sigmoid(10 * X/T - 3)
        sampled_idx = np.random.choice(
            range(len(to_send_idx)),
            size=int(np.ceil(len(to_send_idx) * perc)),
            replace=False,
        )

        # filter the support vectors to send based on the sampled indices
        to_send_idx = [to_send_idx[idx] for idx in sampled_idx]
        to_send = [to_send[idx] for idx in sampled_idx]
        self.console.log(
            f"[green bold][Client {self.client_no}][/] sampled ",
            (floor(100 * perc)),
            "% of support vectors: only ",
            len(to_send_idx),
            " new deltas needed",
        )

        # compute the delta perturbation to apply to the support vectors
        self.latest_delta = self.fit(self.device, to_send_idx)

        delta_norm = torch.mean(
            torch.norm(self.latest_delta, dim=len(self.latest_delta.shape) - 1)
        )
        self.console.log(
            f"[green bold][Client {self.client_no}][/] avg delta norm: {delta_norm:.2f}"
        )

        # apply the delta perturbation to the support vectors, store them among the sent ones
        # and send them to the server
        self.sent = self.sent.union(set([sv_key for sv_key, _ in to_send]))
        to_send_delta = []
        for index, (sv_key, sv) in enumerate(to_send):
            indexed_delta = (
                self.latest_delta[index].detach()
                if len(self.latest_delta.shape) > 1
                else self.latest_delta.detach()
            )
            to_send_delta.append((sv_key, sv.cpu() + indexed_delta.cpu()))

        return to_send_delta, self.svs_labels[to_send_idx]


class FedSVMClientNoOptSD(
    FedSVMClient, FedSVM_NoOptim_SingleDelta, FedSVM_SingleDeltas
):
    """
    Class of a client in the FedSVM algorithm that uses a single delta and no optimisation.
    """

    def __init__(self, client_no, X, y, rff, client_eps, C, kernel, device):
        super().__init__(client_no, X, y, rff, client_eps, C, kernel, device)
        self.console.log(
            "[green bold][Client {self.client_no}][/] Using [red bold]single deltas[/] [blue]no optim[/]"
        )


class FedSVMClientNoOptMD(
    FedSVMClient, FedSVM_NoOptim_MultipleDeltas, FedSVM_MultipleDeltas
):
    """
    Class of a client in the FedSVM algorithm that uses multiple deltas and no optimisation.
    """

    def __init__(self, client_no, X, y, rff, client_eps, C, kernel, device):
        super().__init__(client_no, X, y, rff, client_eps, C, kernel, device)
        self.console.log(
            f"[green bold][Client {self.client_no}][/] Using [red bold]multiple deltas[/] [blue]no optim[/]"
        )


class FedSVMClientRandomMD(FedSVMClient, FedSVM_RandomDeltas, FedSVM_MultipleDeltas):
    """
    Class of a client in the FedSVM algorithm that uses multiple deltas and no optimisation.
    """

    def __init__(self, client_no, X, y, rff, client_eps, C, kernel, device):
        super().__init__(client_no, X, y, rff, client_eps, C, kernel, device)
        self.console.log(
            f"[green bold][Client {self.client_no}][/] Using [red bold]random multiple deltas[/] [blue]no optim[/]"
        )


# class FedSVMClientOptSD(FedSVMClient, FedSVM_Optim, FedSVM_SingleDeltas):

#     """
#     Class of a client in the FedSVM algorithm that uses a single delta and optimisation.
#     """

#     def __init__(self, client_no, X, y, rff, client_eps, C, kernel, device):
#         super().__init__(client_no, X, y, rff, client_eps, C, kernel, device)
#         console.log(
#             "[green bold][Client {self.client_no}][/] Using [red]single deltas[/] [bold blue]optim[/]"
#         )


class FedSVMClientKernel:
    def rbf_kernel(self, X, Y, **params):
        return torch.exp(-params["gamma"] * torch.cdist(X, Y) ** 2)

    def poly_kernel(self, X, Y, **params):
        return (X @ Y.T + 1.0).pow_(params["degree"])
        
    def linear_kernel(self, X, Y, **_):
        return X @ Y.T

    def select_kernel_fun(self):
        self.console.log(
            f"[green bold][Client {self.client_no}][/] Using kernel: {self.kernel}"
        )
        if self.kernel["kernel"] == "rbf":
            return self.rbf_kernel
        elif self.kernel["kernel"] == "poly":
            return self.poly_kernel
        elif self.kernel["kernel"] == "linear" or self.kernel["kernel"] == "rff":
            return self.linear_kernel
        else:
            raise ValueError("Kernel not supported")


class FedSVMClientOptSD(
    FedSVMClient, FedSVM_Optim, FedSVM_SingleDeltas, FedSVMClientKernel
):

    """
    Class of a client in the FedSVM algorithm that uses a single delta and optimisation.
    """

    def __init__(self, client_no, X, y, rff, client_eps, C, kernel, device):
        super().__init__(client_no, X, y, rff, client_eps, C, kernel, device)
        self.console.log(
            "[green bold][Client {self.client_no}][/] Using [red]single deltas[/] [bold blue]optim[/]"
        )

        self.kernel_fun = self.select_kernel_fun()


class FedSVMClientOptMD(
    FedSVMClient, FedSVM_Optim, FedSVM_MultipleDeltas, FedSVMClientKernel
):
    def __init__(self, client_no, X, y, rff, client_eps, C, kernel, device):
        super().__init__(client_no, X, y, rff, client_eps, C, kernel, device)
        self.console.log(
            f"[green bold][Client {self.client_no}][/] Using [red]multiple deltas[/] [bold blue]optim[/]"
        )

        self.kernel_fun = self.select_kernel_fun()


def evaluate(y_true, y_pred):
    """
    Evaluate the predictions y_pred against the true labels y_true.
    """

    return {
        "accuracy": [accuracy_score(y_true, y_pred)],
        "precision": [precision_score(y_true, y_pred)],
        "recall": [recall_score(y_true, y_pred)],
        "f1": [f1_score(y_true, y_pred)],
        "auc": [roc_auc_score(y_true, y_pred)],
    }


def evaluate_all(y, preds):
    """
    Return the accuracy, precision, recall and f1 score and auc of the client-th
    client on the examples X in a dictionary. If client is None, return the metrics for all clients.
    """

    results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": [],
    }

    for pred in preds:
        evals = evaluate(y, pred)
        for metric, value in evals.items():
            results[metric] += value

    return results


class FedSVMServer:

    """
    The server class of the FedSVM algorithm. It contains all the clients and orchestrates
    the communication between them: it receives the support vectors from the clients and sends
    them back so that each one receives the support vectors from all the others.
    When the average dot product between the clients' w vectors is below a certain threshold,
    the algorithm is considered converged and the server stops.
    """

    def __init__(
        self,
        X,
        y,
        n_clients,
        n_random_features,
        server_eps=1e-6,
        client_eps=1e-4,
        C=1e2,
        client_class=FedSVMClientNoOptSD,
        kernel={"kernel": "linear"},
        seed=42,
        device="cpu",
    ) -> None:
        self.console = rich.get_console()
        self.X = X
        self.y = y
        self.seed = seed
        self.server_eps = server_eps
        self.clients_eps = client_eps
        self.C = C

        self.n_features = X.shape[1]
        self.n_random_features = n_random_features
        self.n_clients = n_clients
        self.rff = self.init_rff(kernel)
        self.clients = []
        self.client_class = client_class
        self.kernel = kernel
        self.device = torch.device(device)
        self.converged = False

    def save(self, path):
        """
        Save the server to disk using joblib.
        The server is saved as a list containing the rff, the original svcs of the clients
        and the svcs learned by the clients.
        """

        data_server = [self.rff, self.original_clients_svms]
        data_clients = [client.save_data() for client in self.clients]
        dump([data_server, data_clients], path)

    def load(self, path):
        """
        Load the server from disk using joblib. For each client, its svc is loaded from disk.
        """

        data_server, data_clients = load(path)
        self.rff = data_server[0]
        self.original_clients_svms = data_server[1]

        for client_data in data_clients:
            self.clients.append(
                FedSVMClient(None, None, None, None, None, None, only_load=True)
            )
            self.clients[-1].load_data(client_data)

    def init_rff(self, kernel):
        """
        Initialize the random fourier features transformer that will be used to transform the data
        such that the SVM can be trained in a linear fashion in the random fourier space.
        """
        if kernel["kernel"] != "rff":
            return None

        rff = RandomFourier(
            n_components=self.n_random_features,
            gamma=kernel["gamma"],
            random_state=self.seed,
        )  # , use_offset=True)

        # IMPORTANT: here self.X is only used to get the number of features!!
        rff.fit(self.X)
        return rff

    def _get_clients_examples(self, X, y, n, iidness, seed):
        if iidness == "iid":
            return noniid.iid(X, y, n, seed=seed)
        elif iidness == "quantity":
            return noniid.quantity_skew(X, y, n, seed=seed)
        elif iidness == "covariate":
            return noniid.covariate_shift_cluster(X, y, n, seed=seed)
        else:
            raise ValueError(f"Unknown iidness {iidness}")

    def init_clients(self, iidness):
        """
        Initialize the clients, dividing the data into n_clients equal parts.
        It also saves the original SVMs of the clients, so that they can be used to compare
        the clients after and before the FedSVM algorithm.
        """

        if self.clients:
            return

        self.console.rule(f"[yellow bold]Initialization[/]")

        X, y = shuffle(self.X, self.y, random_state=self.seed)
        assignments = self._get_clients_examples(
            X, y, self.n_clients, iidness, self.seed
        )
        clients = []
        for i in track(range(self.n_clients), description="Initializing clients"):
            self.console.log(f"[red]\nInitializing client {i}")
            client = self.client_class(
                i,
                X[assignments[i]],
                y[assignments[i]],
                self.rff,
                client_eps=self.clients_eps,
                C=self.C,
                kernel=self.kernel,
                device=self.device,
            )
            clients.append(client)

        self.original_clients_svms = [deepcopy(client.svc) for client in clients]
        self.clients = clients

    def run(self, n_rounds=10, iidness="iid"):
        """
        The core of the Server. It runs for n_rounds, receiving the support vectors from the clients
        and sending them back so that each one receives the support vectors from all the others.
        When the average dot product between the clients' w vectors is below a certain threshold,
        the algorithm is considered converged and the server stops.
        """

        self.init_clients(iidness)

        with Progress(transient=True) as progress:
            srv_task = progress.add_task("[red]Server", total=n_rounds)
            clt_send = progress.add_task("[green]Client send", total=len(self.clients))
            clt_recv = progress.add_task("[blue]Client recv", total=len(self.clients))

            cnt_total_svs = 0
            for i in range(n_rounds):
                progress.update(srv_task, advance=1)
                progress.reset(clt_send)
                progress.reset(clt_recv)
                self.console.rule(f"[yellow bold]Round[/] {i+1}", style=".")
                svs_clients = []
                labels_clients = []
                # SERVER RECEIVES SVs FROM CLIENTS
                cnt_svs = 0
                for j, client in enumerate(self.clients):
                    progress.update(clt_send, advance=1)
                    self.console.log(
                        f"[red bold][Server][/] << Receiving SVs from client {j}"
                    )
                    svs, labels = client.send_svs()
                    svs_clients.append(svs)
                    labels_clients.append(labels)
                    cnt_svs += len(svs)

                self.console.log(
                    f"[red bold][Server][/] << Received {cnt_svs} SVs from clients"
                )

                cnt_total_svs += cnt_svs
                if cnt_svs == 0:
                    self.console.rule("Algorithm converged")
                    self.converged = True
                    break

                # SERVER SENDS SVs TO CLIENTS
                for c, client in enumerate(self.clients):
                    progress.update(clt_recv, advance=1)
                    svs_clients_c = svs_clients[:c] + svs_clients[c + 1 :]
                    labels_clients_c = labels_clients[:c] + labels_clients[c + 1 :]

                    count_svs_c = sum([len(svs) for svs in svs_clients_c])
                    self.console.log(
                        f"[red bold][Server][/] >> Sending {count_svs_c} SVs to client {c}"
                    )
                    cnt_total_svs += count_svs_c
                    client.receive_svs(svs_clients_c, labels_clients_c)

            self.console.log(
                f"[red bold][Server][/] Total number of SVs sent: {cnt_total_svs}"
            )

    def predict(self, X, client=0):
        """
        Return the predictions of the client-th client on the examples X.
        """

        return self.clients[client].predict(X, self.rff)

    def evaluate(self, X, y, client=None):
        """
        Return the accuracy, precision, recall and f1 score and auc of the client-th
        client on the examples X in a dictionary. If client is None, return the metrics for all clients.
        """

        if client is None:
            preds = [self.predict(X, client=i) for i in range(self.n_clients)]
            return evaluate_all(y, preds)
        else:
            return evaluate(y, self.predict(X, client=client))


def string_to_client_class(string):
    """
    Define the type of client to use based on the string parameter in the config file.
    This will be used to initialize the clients in the server.
    """

    if string == "FedSVMClientNoOptSD":
        return FedSVMClientNoOptSD
    elif string == "FedSVMClientNoOptMD":
        return FedSVMClientNoOptMD
    elif string == "FedSVMClientOptSD":
        return FedSVMClientOptSD
    elif string == "FedSVMClientOptMD":
        return FedSVMClientOptMD
    elif string == "FedSVMClientRandomMD":
        return FedSVMClientRandomMD
    else:
        raise ValueError(f"Unknown client class {string}")


def print_results_table(
    title, config, results, centralized_results, nsv_per_client, nsv_centralized
):
    table = Table(title=title)

    table.add_column("Description", style="bright_black")
    table.add_column("Num SVs", style="bright_black")
    metrics = list(results.keys())
    for metric in metrics:
        table.add_column(metric, style="bright_black")

    table.add_row(
        *(
            ["[bold blue]Centralized[/]", f"[blue]{nsv_centralized}[/]"]
            + [f"[cyan]{centralized_results[key][0]:.4f}[/]" for key in metrics]
        )
    )

    table.add_row(
        *["[bold green]Clients avgs[/]", f"[green]{np.mean(nsv_per_client):.2f}[/]"]
        + [f"[green]{np.mean(values):.4f}[/]" for values in results.values()]
    )

    for cn in range(config["n_clients"]):
        row = [
            f"Client {cn}",
            f"{nsv_per_client[cn]}",
        ] + [f"{results[metric][cn]:.4f}" for metric in metrics]
        table.add_row(*row)

    console.print(table)


def evaluate_results(
    server, X_train, X_test, y_train, y_test, config, config_path, force
):
    centr_kernel_config = deepcopy(config["kernel_config"])
    if config["kernel_config"]["kernel"] == "rff":
        centr_kernel_config["kernel"] = "rbf"

    exp_handler = ExpHandler(f"{os.path.dirname(config_path)}/results.json", config)
    exp_entry = exp_handler.new_entry()

    # Evaluation
    console.rule("[bold red]Evaluation[/]")

    results = server.evaluate(X_test, y_test)

    svc = SVC(**centr_kernel_config, C=config["C"])
    svc.fit(X_train, y_train)

    centralized_results = evaluate(y_test, svc.predict(X_test))

    nsv_per_client = [len(client.svc.support_vectors_) for client in server.clients]
    nsv_centralized = len(svc.support_vectors_)

    print_results_table(
        "Fed Clients Results",
        config,
        results,
        centralized_results,
        nsv_per_client,
        nsv_centralized,
    )

    exp_entry.add_result("client_results", results)
    exp_entry.add_result("centralized_svm_results", centralized_results)

    if server.rff is not None:
        console.log("[red]\nEvaluating centralized SVM-RFF")
        svc_rff = SVC(kernel="linear", C=config["C"])
        svc_rff.fit(server.rff.transform(X_train), y_train)

        Z_test = torch.tensor(server.rff.transform(X_test))

        centralized_rff_results = evaluate(y_test, svc_rff.predict(Z_test))
        for metric, values in centralized_rff_results.items():
            console.log(f"Centralized with RFF {metric}: {values}")

        exp_entry.add_result("centralized_rff_results", centralized_rff_results)
    else:
        Z_test = torch.tensor(X_test)

    preds = [
        server.original_clients_svms[i].predict(Z_test)
        for i in range(config["n_clients"])
    ]

    evals = evaluate_all(y_test, preds)
    exp_entry.add_result("original_client_results", evals)

    nsv_per_client = [len(svc.support_vectors_) for svc in server.original_clients_svms]

    print_results_table(
        "Original Clients Results",
        config,
        evals,
        centralized_results,
        nsv_per_client,
        nsv_centralized,
    )

    exp_handler.add_entry(exp_entry, force=force)

    console.save_html(f"{os.path.dirname(config_path)}/log.html")
    rich.print(f"[bold green]Log also saved in {os.path.dirname(config_path)}/log.html")


def run(
    config_path: Annotated[str, typer.Argument(help="Path to the config file")],
    device: Annotated[
        str,
        typer.Option(
            help="Device to use (cpu, cuda, mps). If None the device is auto-detected"
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(help="Force learning even if a model has been already saved"),
    ] = False,
    log_on_file: Annotated[
        bool,
        typer.Option(
            help='If true, the log entries will be stored in the "log.txt" file within the same directory as the configuration file.'
        ),
    ] = True,
):
    """
    Test the FedSVM algorithm using the configuration file at config_path.
    Most of the configuration file parameters should be self-explanatory.
    \n\n
    The client_class parameter can be one of the following:\n
        - FedSVMClientNoOptSD: No optimisation, single delta\n
        - FedSVMClientNoOptMD: No optimisation, multiple deltas\n
        - FedSVMClientOptSD: Optimisation, single delta\n
        - FedSVMClientOptMD: Optimisation, multiple deltas\n
        - FedSVMClientOptSDKernel: Optimisation, single delta, kernelized\n
    """
    global console

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    with open(config_path) as f:
        config = json.load(f)

    if log_on_file:
        rich.reconfigure(
            file=open(f"{os.path.dirname(config_path)}/log.txt", "w"), record=True
        )
    else:
        rich.reconfigure(record=True)

    console = rich.get_console()

    assert not (
        config["kernel_config"]["kernel"] in ["rbf", "poly"]
        and config["client_class"] in ["FedSVMClientNoOptSD", "FedSVMClientOptSD"]
    ), "Config error: cannot use rbf or poly kernel with single delta clients"

    console.rule("[bold red]Starting Experiment[/]")
    console.log(f"[green]Using device[/]: {device}")
    console.log("[green]Configuration[/]")
    console.log(config)

    X_train, X_test, y_train, y_test = load_dataset(config)

    server = FedSVMServer(
        X_train,
        y_train,
        config["n_clients"],
        n_random_features=config["n_random_features"],
        server_eps=config["server_eps"],
        client_eps=config["client_eps"],
        C=config["C"],
        client_class=string_to_client_class(config["client_class"]),
        kernel=dict(config["kernel_config"]),
        seed=config["seed"],
        device=device,
    )

    # Federated SVM simulation
    console.rule("[bold red]Federated SVM[/]")

    server.run(config["n_rounds"], config["iidness"])

    evaluate_results(
        server, X_train, X_test, y_train, y_test, config, config_path, force
    )


if __name__ == "__main__":
    try:
        typer.run(run)
    except SystemExit:
        pass
    except Exception as e:
        # Usar traceback estándar en lugar de rich para evitar problemas de encoding
        import traceback
        traceback.print_exc()
        sys.exit(1)
